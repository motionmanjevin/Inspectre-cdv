
This system allows you to capture **live video**, **extract meaningful events**, store them in memory, and ask **natural language questions** about anything that has happened — just like a human assistant watching a video feed.

## 📌 Project Highlights

- 📹 Live camera input processing
- 🔍 Object & scene detection using [CLIP](https://huggingface.co/openai/clip-vit-base-patch32)
- 🧠 Video understanding using **Video-LLaVA-style architecture** (multi-modal memory + LLM reasoning)
- 📚 Memory system built on **FAISS** for fast vector search
- 🤖 Natural language QA with Hugging Face's Falcon-7B-Instruct

---

## 🏗️ Architecture Overview 

### 🔄 Streaming Pipeline (Video-LLaVA Style)

1. **Live Capture**: Frames streamed in real-time via webcam (OpenCV)
2. **Perceptual Encoding**: Frames converted to feature embeddings via CLIP (mimicking the visual encoder in Video-LLaVA)
3. **Event Abstraction**: Events are inferred from tags (e.g., “Person, cup, phone” → “Person picked up phone”)
4. **Vector Storage**: Events are embedded (mock or real) and stored in a searchable FAISS vector index
5. **LLM Agent**: Falcon-7B-Instruct (or any Hugging Face LLM) reasons over past events using retrieval + generation

---

## 🔧 Requirements

Install all required dependencies using:

```bash
pip install -r requirements.txt
```

Make sure you have:
- Python 3.8+
- An NVIDIA GPU (recommended for inference)
- Hugging Face model cache (some downloads may take time)

---

## 🚀 Running the System

```bash
python main.py
```

The system will:
1. Start capturing frames from webcam
2. Detect objects using CLIP
3. Build a semantic memory of visual events
4. Enter QA mode so you can ask:  
   > "What objects were detected earlier?"  
   > "Did someone pick up a cup?"  
   > "What happened after a person entered?"

---

## 💬 Example Workflow

```bash
Ask a question (or type 'exit' to quit): What did the person do?
Answer: Detected: person, phone — Possibly picked up phone
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

## 📁 Project Structure

```
.
├── main.py                  # Entry point
├── requirements.txt
├── README.md
├── video/
│   ├── capture.py           # OpenCV-based video stream
│   └── processor.py         # Frame tagger (CLIP)
├── embedding/
│   ├── event_extractor.py   # Tag-to-event abstraction
│   └── vector_store.py      # FAISS-based memory store
├── llm/
│   ├── query_agent.py       # LLM-based QA over event memory
│   └── prompts.py           # QA prompt templates
└── utils/
    └── logger.py            # Optional logging utility
```

---

## 🛠️ Extending This System

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

## 📬 Contact

Built by ChatGPT for Kevin Anaba 💡. Customizable for surveillance, smart monitoring, or research. Let me know if you want deployment-ready Docker + GUI!
