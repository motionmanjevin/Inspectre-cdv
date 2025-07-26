# Entry point for the real-time video QA system
from video.capture import VideoStream
from video.processor import FrameProcessor
from embedding.event_extractor import EventExtractor
from embedding.vector_store import VectorMemory
from llm.query_agent import QuestionAnswerAgent

import time

def main():
    print("[INFO] Starting Real-Time Video QA System...")

    stream = VideoStream()
    processor = FrameProcessor()
    extractor = EventExtractor()
    memory = VectorMemory()
    qa_agent = QuestionAnswerAgent(memory)

    print("[INFO] System ready. Processing live stream...")
    print("[INFO] To stop the stream, press Ctrl+C.")

    try:
        for frame in stream.capture():
            tags = processor.process(frame)
            if tags:
                event = extractor.extract(tags)
                memory.store(event)
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n[INFO] Video stream stopped.")

    print("[INFO] Entering QA mode. Ask questions about what happened.")
    while True:
        try:
            query = input("\nAsk a question (or type 'exit' to quit): ")
            if query.strip().lower() in ['exit', 'quit']:
                break
            answer = qa_agent.answer(query)
            print("\nAnswer:", answer)
        except KeyboardInterrupt:
            break

if __name__ == '__main__':
    main()