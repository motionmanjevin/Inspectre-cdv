import os
import time
import queue
import threading
import sqlite3
from datetime import datetime
from typing import List, Tuple

import cv2
import streamlit as st
from PIL import Image
import numpy as np
from tqdm import tqdm

# Transformers / BLIP-2
import torch
from transformers import (
    Blip2Processor,
    Blip2ForConditionalGeneration,
    AutoProcessor,
    AutoModelForVision2Seq,
)

# Gemini
import google.generativeai as genai

# -------------------------
# ======= CONFIG ==========
# -------------------------
DB_FILE = "video_logs.db"
DEFAULT_FRAME_INTERVAL = 1.0  # seconds between captions
MAX_LOG_CONTEXT = 120  # max number of recent logs to include when querying Gemini
GPU = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# ======= DB HELPERS ======
# -------------------------
def init_db(db_path=DB_FILE):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        """CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            human_time TEXT,
            caption TEXT
        )"""
    )
    conn.commit()
    return conn

DB_CONN = init_db()

def insert_log(timestamp: float, caption: str, conn=DB_CONN):
    cur = conn.cursor()
    human_time = datetime.utcfromtimestamp(timestamp).isoformat() + "Z"
    cur.execute("INSERT INTO logs (timestamp, human_time, caption) VALUES (?, ?, ?)",
                (timestamp, human_time, caption))
    conn.commit()

def fetch_logs(limit: int = 200, conn=DB_CONN) -> List[Tuple[float, str, str]]:
    cur = conn.cursor()
    cur.execute("SELECT timestamp, human_time, caption FROM logs ORDER BY timestamp ASC LIMIT ?", (limit,))
    return cur.fetchall()

def clear_logs(conn=DB_CONN):
    cur = conn.cursor()
    cur.execute("DELETE FROM logs")
    conn.commit()

# -------------------------
# ======= MODEL HELPERS ===
# -------------------------
@st.cache_resource(show_spinner=False)
def load_blip_model(choice: str):
    """
    choice: one of
      - 'Salesforce/blip2-flan-t5-xl' (heavy)
      - 'Salesforce/blip2-flan-t5-base' (smaller)
      - 'Salesforce/blip-image-captioning-base' (BLIP-1 small, CPU-friendly)
    """
    if "blip2" in choice:
        processor = Blip2Processor.from_pretrained(choice)
        # for big models use float16 on GPU, otherwise float32
        dtype = torch.float16 if GPU == "cuda" else torch.float32
        model = Blip2ForConditionalGeneration.from_pretrained(choice, torch_dtype=dtype).to(GPU)
        return processor, model
    else:
        # fallback BLIP-1 style
        processor = AutoProcessor.from_pretrained(choice)
        model = AutoModelForVision2Seq.from_pretrained(choice).to(GPU)
        return processor, model

def caption_with_blip(processor, model, pil_img: Image.Image, question: str = "Describe what is happening in this image.") -> str:
    # works for both BLIP-2 and BLIP-1 variants (basic usage)
    inputs = processor(images=pil_img, text=question, return_tensors="pt").to(GPU)
    gen = model.generate(**inputs, max_new_tokens=64)
    # decode - processor may not always have tokenizer; use typical pattern:
    try:
        caption = processor.tokenizer.decode(gen[0], skip_special_tokens=True)
    except Exception:
        # fallback: convert tensor to list of ints then join tokens (not ideal)
        caption = gen[0].cpu().numpy().tolist()
        caption = str(caption)
    return caption

# -------------------------
# ======= CAPTURE WORKER ==
# -------------------------
class CaptureWorker(threading.Thread):
    def __init__(self, frame_queue: queue.Queue, processor, model, frame_interval=1.0):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.processor = processor
        self.model = model
        self.frame_interval = frame_interval
        self.running = threading.Event()
        self.last_process_time = 0.0

    def start_worker(self):
        self.running.set()
        if not self.is_alive():
            self.start()

    def stop_worker(self):
        self.running.clear()

    def run(self):
        # Takes frames from queue and captions them respecting frame_interval
        while True:
            # block for a short time so we can exit gracefully
            try:
                frame_ts, frame = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                if not self.running.is_set():
                    time.sleep(0.1)
                    continue
                else:
                    continue

            # rate-limit captioning
            now = time.time()
            if now - self.last_process_time < self.frame_interval:
                # skip if too close (drop frame)
                continue

            try:
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                caption = caption_with_blip(self.processor, self.model, pil_img)
            except Exception as e:
                caption = f"[caption error: {e}]"

            insert_log(frame_ts, caption)
            self.last_process_time = time.time()

            # if stopped, break loop (thread continues to live but will pause)
            if not self.running.is_set():
                time.sleep(0.1)

# -------------------------
# ======= GEMINI HELPERS ==
# -------------------------
def configure_gemini_from_key(key: str):
    genai.configure(api_key=key)

def query_gemini_for_question(question: str, max_logs: int = MAX_LOG_CONTEXT) -> str:
    """
    Build a compact context from recent logs and query Gemini.
    """
    logs = fetch_logs(limit=max_logs)
    if not logs:
        return "No logs available yet."

    # Keep logs short — include only recent N and truncate long captions
    def short(c, n=300):
        return c if len(c) <= n else c[:n-1] + "…"

    context_lines = [f"[{int(ts)}] {ht} — {short(caption)}" for ts, ht, caption in logs[-max_logs:]]
    context_text = "\n".join(context_lines)

    prompt = (
        "You are an assistant analyzing log entries from a live camera. Each entry has a UNIX "
        "timestamp and an ISO time and a short caption of what was seen.\n\n"
        f"Logs:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer using the logs where relevant. Mention timestamps or approximate times if helpful."
    )

    # choose model; user may pick other Gemini variant — mirror earlier recommendation
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    return response.text

# -------------------------
# ======= STREAMLIT UI ====
# -------------------------
st.set_page_config(page_title="Live Video → Gemini Assistant", layout="wide")
st.title("📹 Live Video + Gemini Chat (Real-time captioning & logs)")

# Left panel: controls + preview
col_left, col_right = st.columns([2, 1])

with col_right:
    st.header("Controls / Status")
    # Gemini API key
    api_key_input = st.text_input(
        "Gemini API Key (or set environment variable GEMINI_API_KEY)",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
    )
    if api_key_input:
        try:
            configure_gemini_from_key(api_key_input)
            st.success("Gemini configured.")
        except Exception as e:
            st.error(f"Gemini config error: {e}")

    # model choice for captioning
    model_choice = st.selectbox(
        "Caption model (pick smaller if you lack VRAM)",
        options=[
            "Salesforce/blip-image-captioning-base",
            "Salesforce/blip2-flan-t5-base",
            "Salesforce/blip2-flan-t5-xl",
        ],
        index=0,
    )

    frame_interval = st.number_input(
        "Seconds between captions (frame interval)",
        min_value=0.1,
        max_value=10.0,
        value=float(DEFAULT_FRAME_INTERVAL),
        step=0.1,
    )

    st.markdown("**Actions**")
    start_button = st.button("Start live capture")
    stop_button = st.button("Stop capture")
    clear_button = st.button("Clear logs")
    download_button = st.button("Download logs (CSV)")

    # Show last processed time & total logs
    logs = fetch_logs(limit=100000)
    if logs:
        last_ts, last_iso, _ = logs[-1]
        processed_since = datetime.utcfromtimestamp(logs[0][0]).isoformat() + "Z"
        st.write("✅ Logs:", len(logs))
        st.write(f"Last processed frame at (unix): {last_ts:.2f}  —  {last_iso}")
    else:
        st.write("No logs yet.")

    if clear_button:
        clear_logs()
        st.success("Cleared logs.")
        st.experimental_rerun()

    if download_button:
        import pandas as pd
        df = pd.DataFrame(fetch_logs(limit=100000), columns=["timestamp", "human_time", "caption"])
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="video_logs.csv", mime="text/csv")

with col_left:
    st.header("Live Camera Feed")
    vid_placeholder = st.empty()
    info_placeholder = st.empty()
    progress_placeholder = st.progress(0)

    # Prepare frame queue and worker in session state
    if "frame_queue" not in st.session_state:
        st.session_state.frame_queue = queue.Queue(maxsize=8)
    if "worker" not in st.session_state or st.session_state.get("model_choice") != model_choice:
        # (re)load model if new choice
        try:
            processor, model = load_blip_model(model_choice)
            st.session_state.processor = processor
            st.session_state.model = model
            st.session_state.model_choice = model_choice
            st.session_state.worker = CaptureWorker(
                frame_queue=st.session_state.frame_queue,
                processor=processor,
                model=model,
                frame_interval=frame_interval,
            )
            st.session_state.capture_device = None
            st.success(f"Loaded model: {model_choice} (device={GPU})")
        except Exception as e:
            st.error(f"Could not load model {model_choice}: {e}")

    # update worker interval if changed
    if "worker" in st.session_state:
        st.session_state.worker.frame_interval = frame_interval

    # Start/Stop camera capture
    if start_button:
        # open camera if not opened
        if st.session_state.get("capture_device") is None:
            cap = cv2.VideoCapture(0)
            st.session_state.capture_device = cap
        else:
            cap = st.session_state.capture_device

        if not st.session_state.worker.running.is_set():
            st.session_state.worker.start_worker()
        st.success("Capture started (background captioning running).")

    if stop_button:
        if st.session_state.get("capture_device") is not None:
            try:
                st.session_state.capture_device.release()
            except Exception:
                pass
            st.session_state.capture_device = None
        if "worker" in st.session_state:
            st.session_state.worker.stop_worker()
        st.warning("Capture stopped.")

    # main preview loop (non-blocking)
    preview_run = st.checkbox("Show live preview (update every 0.2s)", value=True)

    # Display loop: read a frame from camera and show
    last_preview_time = 0
    try:
        cap = st.session_state.get("capture_device", None)
        if cap is None:
            # show placeholder image
            vid_placeholder.info("Camera not running. Click 'Start live capture' to open webcam.")
        else:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                vid_placeholder.error("Could not read from camera.")
            else:
                # show
                vid_placeholder.image(frame, channels="BGR", use_column_width=True)
                now_ts = time.time()
                # push into queue for captioning if worker is running
                if st.session_state.worker.running.is_set():
                    try:
                        st.session_state.frame_queue.put_nowait((now_ts, frame.copy()))
                    except queue.Full:
                        # drop frame to keep up
                        pass

                # update progress UI: get last processed timestamp
                all_logs = fetch_logs(limit=2000)
                if all_logs:
                    first_ts = all_logs[0][0]
                    last_ts = all_logs[-1][0]
                    # estimate progress relative to elapsed capture time (if capture started)
                    # if capture just running, show fraction as (last - first) normalized to some cap (e.g., 600s)
                    elapsed = last_ts - first_ts
                    cap_seconds = 600.0  # cap display at 10 minutes for progress bar ratio
                    ratio = min(elapsed / cap_seconds, 1.0)
                    progress_placeholder.progress(ratio)
                    info_placeholder.metric("Last processed (UTC)", datetime.utcfromtimestamp(last_ts).isoformat() + "Z", delta=f"{int(elapsed)}s processed")
                else:
                    progress_placeholder.progress(0.0)
                    info_placeholder.write("Waiting for first captions...")
    except Exception as e:
        st.error(f"Preview loop error: {e}")

# -------------------------
# ======= CHAT INTERFACE ==
# -------------------------
st.markdown("---")
st.header("💬 Chat with the Video Assistant (Gemini)")

with st.form("chat_form", clear_on_submit=False):
    user_question = st.text_input("Ask a question about the footage (e.g., 'What happened between 10:00 and 10:05 UTC?')", "")
    include_recent = st.slider("Include how many recent log entries as context", min_value=10, max_value=500, value=MAX_LOG_CONTEXT, step=10)
    submitted = st.form_submit_button("Ask")

if submitted:
    # ensure Gemini configured
    configured = False
    try:
        # genai has an internal config field; check if configured via environment or earlier input
        # we still try to configure from env var if not set by text input
        if api_key_input:
            configure_gemini_from_key(api_key_input)
            configured = True
        else:
            env_key = os.environ.get("GEMINI_API_KEY", None)
            if env_key:
                configure_gemini_from_key(env_key)
                configured = True
    except Exception as e:
        st.error(f"Gemini configuration error: {e}")

    if not configured:
        st.error("Gemini is not configured. Provide API key in the control panel or set GEMINI_API_KEY env var.")
    else:
        with st.spinner("Querying Gemini with logs..."):
            try:
                answer = query_gemini_for_question(user_question, max_logs=include_recent)
                st.write("**Gemini:**")
                st.write(answer)
            except Exception as e:
                st.error(f"Gemini query error: {e}")

st.markdown("### Recent logs (most recent 50)")
recent = fetch_logs(limit=50)
for ts, ht, caption in recent[-50:]:
    st.write(f"- [{int(ts)} | {ht}] — {caption}")
