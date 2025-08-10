# **Inspectre – AI-Powered Hospital Intelligence for Korle Bu**

> Turning scattered patient data and unstructured video into real-time, actionable insights.

---

## **Overview**

**Inspectre** is an AI-driven hospital intelligence platform that combines **computer vision**, **real-time data integration**, and **natural language querying** to transform how healthcare teams monitor patients, track surgical events, and manage outpatient flow.


This adaptation resulted in **three specialized modules**:

1. **AI-Powered Patient Tracking & Risk Flagging**
    
2. **AI-Assisted Surgical Registry & Longitudinal Case Documentation**
    
3. **AI-Assisted OPD Scheduling & Patient Flow Monitoring**
    

---

## **The Main Concept**

The **core idea** of Inspectre is to provide a **single pane of glass** where staff can:

- Monitor live feeds from any ward or surgical theater.
    
- Detect and get alerted about critical events automatically.
    
- Search historical footage and patient events using plain language queries.
    
- Make **data-driven decisions** instantly.
    

This is powered by:

- **AI-based video analytics** (movement, posture, collapse, inactivity detection).
    
- **Automated metadata generation** for every detected event.
    
- **Structured medical documentation** from unstructured visual/audio data.
    
- **Cross-system integration** with hospital databases.
    

---

## **Korle Bu’s Challenges**

During initial needs assessment at KBTH, three **major pain points** were identified:

### **Challenge 1 – Fragmented Patient Monitoring**

- Inpatients are scattered across multiple wards, each with separate monitoring setups.
    
- No unified, real-time dashboard for doctors, nurses, and specialists to track all patients’ statuses.
    
- Critical changes (falls, inactivity, deterioration) can go unnoticed until it’s too late.
    ![[Inspectre - AI-Powered Healthcare Dashboard - Google Chrome 8_10_2025 11_22_59 PM.png]]

### **Challenge 2 – Unstructured Surgical Documentation**

- Surgeries are recorded, but footage is long, unindexed, and lacks structured notes.
    
- Reviewing past cases for training or audits is time-consuming.
    
- Finding a specific event in a surgery (e.g., start of tumor removal) requires manual scrubbing.
    ![alt text](<Inspectre - AI-Powered Healthcare Dashboard - Google Chrome 8_10_2025 11_23_51 PM.png>)!

### **Challenge 3 – Outpatient Congestion & Poor Flow Tracking**

- OPD waiting areas get overcrowded with no live tracking of wait times or patient vulnerabilities.
    
- Vulnerable patients (elderly, disabled, pregnant) can end up waiting too long without priority.
    
- No real-time analytics to optimize staffing or patient routing.
    ![[Inspectre - AI-Powered Healthcare Dashboard - Google Chrome 8_10_2025 11_24_04 PM 1.png]]

---

## **Inspectre’s Fitted Solutions**

Inspectre’s base capabilities were **branched into specialized modules** for each problem:

---

### **AI-Powered Patient Tracking & Risk Flagging**

**Problem Fit:**  
Centralizes patient status monitoring with **risk-level detection** and **event-triggered alerts**.

**How It Works:**

- Live video feeds from each bed are analyzed in real-time.
    
- AI assigns **risk badges** (Low/Medium/High) based on activity, posture, vitals (if integrated).
    
- Alerts trigger for falls, prolonged inactivity, unusual movement patterns.
    
- All staff can view the same **real-time dashboard** filtered by ward, patient, or risk type.
    

---

### **AI-Assisted Surgical Registry & Case Documentation**

**Problem Fit:**  
Transforms raw surgical video into **searchable, structured records**.

**How It Works:**

- Surgery videos get **AI-generated phase markers** (Prep, Incision, Closure).
    
- Critical events (e.g., bleeding, tumor removal start) are timestamped.
    
- Generates **structured summaries** with procedure type, duration, complications, and key events.
    
- Search using plain language: _“All craniotomies with complications in 2024”_.
    

---

### **AI-Assisted OPD Scheduling & Patient Flow**

**Problem Fit:**  
Enables **real-time patient flow management** to reduce wait times and prioritize care.

**How It Works:**

- Live OPD cameras generate **heatmaps** of crowding.
    
- Tracks check-in times and calculates **current wait time** per patient.
    
- Highlights vulnerable patients waiting too long.
    
- Provides **operational insights** (e.g., busiest hours, staffing recommendations).
    

---

## **Why Inspectre is a Perfect Fit for KBTH**

- **Centralized Intelligence** → All three modules feed into one secure, unified platform.
    
- **Proactive Alerts** → Staff are notified before small issues become emergencies.
    
- **Time Savings** → From auto-documenting surgeries to instantly locating patients at risk.
    
- **Training Value** → Indexed surgical footage becomes a goldmine for skill development.
    
- **Data-Driven Management** → Decisions based on live metrics, not guesswork.
    

---

## **Future Potential**

While Inspectre at KBTH starts with these three modules, its architecture allows:

- **Integration with EHR systems** for full patient records.
    
- **Predictive analytics** for patient deterioration.
    
- **Automated triage recommendations** based on historical patterns.

# VIDEO DEMONSTRATION

![[Project 3.mp4]]
---


## 📌 Project Highlights

- 📹 Live camera input processing
- 🔍 Object & scene detection using [CLIP](https://huggingface.co/openai/clip-vit-base-patch32)
- 🧠 Video understanding using **Video-LLaVA-style architecture** (multi-modal memory + LLM reasoning)
- 📚 Memory system built on **FAISS** for fast vector search
- 🤖 Natural language QA with Hugging Face's Falcon-7B-Instruct

---
# Inspectre — Full Architecture Overview

# High-level Architecture (components & flow)

```
[Cameras / Streams] --> [Ingest Layer] --> [Edge Preprocessing]
                                      \
                                       --> [Lightweight Segment Encoder] --> [Event Detector / Classifier]
                                                                      \
                                                                       --> [Event Summarizer] --> [Vector Store / Event DB] <--+
                                                                                                                             |
[User / UI / API] <-- [Query Router] <-- [Retriever (ANN: FAISS/Milvus/Chroma)] <-- [Re-ranker / Filter] <-- [LLM Reasoner] <---+
                                                                                                                             |
[Alerting / Automation] <---------------------------------------------------------------------------------------------------+
```

### Components

1. **Ingest Layer**
    
    - Handles RTSP/RTMP/ONVIF/GStreamer inputs, stream health checks, multi-camera management.
        
    - Tech: GStreamer, FFmpeg, OpenCV for prototyping.
        
2. **Edge Preprocessing** (optional but recommended)
    
    - Motion detection, frame sampling, ROI cropping, low-res feature extraction to save bandwidth.
        
    - Deploys on an edge device (Jetson, Coral, Raspberry Pi + TPU) or on-prem NUC.
        
3. **Lightweight Segment Encoder**
    
    - Fast vision encoder that converts short clips or keyframes into embeddings. (e.g., MobileNet variants, distilled ViT/VideoMAE-Small)
        
    - Runs at high throughput; extracts per-segment embeddings + keyframe pointers.
        
4. **Event Detector & Classifier**
    
    - Detects discrete events (enter/exit, object pick up, fall, interaction) and generates structured event logs with timestamps.
        
    - Can use lightweight action recognition or custom classifiers.
        
5. **Event Summarizer & NLP Normalizer**
    
    - Converts event labels + metadata into short natural-language sentences (for LLM context and human readability).
        
    - Example entry: `{time: "2025-08-10T12:01Z", event: "person_enters", desc: "Man in blue shirt entered room", bbox: [..], clip_ref: "S3://.."}`
        
    - Optionally runs ASR on audio to capture speech events.
        
6. **Long-term Memory: Vector Store + Event DB**
    
    - **Vector DB** (FAISS, Milvus, Chroma, Pinecone) stores embeddings for semantic retrieval.
        
    - **Event DB** (Postgres/TimescaleDB) stores structured events, metadata, clip locations, and indices.
        
    - Versioned pointers to original clips in object storage (S3, MinIO) for playback.
        
7. **Retriever + Re-ranker**
    
    - Retriever: ANN search on vector DB to get top-k relevant events/clips for a query.
        
    - Re-ranker: lightweight cross-encoder or heuristic filter to improve precision (time-window constraints, camera id, person-id, confidence thresholds).
        
8. **LLM Reasoner / QA Layer**
    
    - Receives retrieved events + short clips (or summaries) and composes final, human-level answers.
        
    - Uses retrieval-augmented generation (RAG) patterns, prompt templates, and hallucination mitigation (cite timestamps + clip refs).
        
    - Can run local LLMs (Mistral/Phi-style) or API LLMs (OpenAI, Anthropic) depending on privacy/latency constraints.
        
9. **Alerting & Automation**
    
    - Rule engine that triggers alerts / webhooks / automated actions (e.g., PTZ tracking, siren, notify staff).
        
    - Integrates with Slack, SMS, PagerDuty, or local dashboards.
        
10. **UI/API**
    

- Web UI for live view, timeline, semantic search, and question box.
    
- REST/gRPC API for programmatic access, integrations, and mobile clients.
    

11. **Mgmt & Observability**
    

- Metrics (Prometheus), logs (ELK/Vector), tracing (Jaeger), model metrics, and data drift detection.
    

---

# Data model (example)

**Event record**

```json
{
  "id": "evt_0001",
  "camera_id": "cam_01",
  "start_time": "2025-08-10T12:01:03Z",
  "end_time": "2025-08-10T12:01:10Z",
  "event_label": "person_enters",
  "description": "Person in blue shirt entered living room carrying a bag",
  "embedding_id": "vec_0001",
  "clip_ref": "s3://inspectre-clips/cam_01/2025-08-10/001.mp4",
  "bbox": [x,y,w,h],
  "confidence": 0.93,
  "faces": [{"id":"face_12", "embedding":"facevec_12"}]
}
```

**Vector DB entry**

- `id` -> `embedding` -> `metadata` (event id, time, camera id, short text summary)
    

---

# 4 — Query flow (example)

1. User asks: “When did anyone pick up a red cup in the kitchen today?”
    
2. **Query Router** parses: intent = `find_event`, keywords = `pick up`, `red cup`, `kitchen`, time window = `today`.
    
3. **Retriever**: semantic search over vector DB + filter by camera/area tags + time constraint.
    
4. **Re-ranker**: score and keep top-k matches.
    
5. **LLM Reasoner**: composes an answer referencing timestamps and providing clip links, e.g.,
    
    > “At 13:12:05 on camera cam_kitchen — a person wearing a white shirt picked up a red cup. Clip: [link].”
    
6. Optionally play back the clip or jump to time-coded frames in UI.
    

---



## Testing 

```bash
python chk2.py #this is for running the camera module for detection (webcam by default)
python dist_checking.py #to run frame captioning on lightweight BLIP
python process_checking #to run frame captioning using heavyweight BLIP
python summirizer.py #to run whole scene inference 
#after recorded 

```



---



## 🔍 Improvements over standard  Video-LLaVA architecture

| Component            | This System                        | Video-LLaVA Equivalent          |
|---------------------|------------------------------------|---------------------------------|
| Visual Perception    | CLIP object/scene tagging          | ViT/BLIP2 video encoder         |
| Event Abstraction    | Tag-based event builder            | Multimodal decoder (Q-Former)   |
| Long-term Memory     | FAISS vector store                 | Memory stack in LLaVA           |
| LLM Agent            | Falcon-7B-instruct pipeline        | LLaVA decoder (ChatGPT-4o like) |
| Input Type           | Live stream from webcam            | Video snippets / multi-modal    |

---


## Extending This System

we are working on:
- Replacing mock embeddings with **CLIP sentence embeddings**
- Swapping Falcon-7B with **Ollama** or **Mistral-7B** if you want local inference
- Adding FastAPI for a browser-based chat UI
- Incorporating full video scene description models like **VideoBLIP** or **VideoLLaVA**

---

## 📣 Acknowledgments

- [Video-LLaVA Paper](https://arxiv.org/abs/2403.08016)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [FAISS](https://github.com/facebookresearch/faiss)

---
