# Entry point for the real-time Video QA system
import time
import signal
import sys

from video.capture import VideoStream
from video.processor import FrameProcessor
from embedding.event_extractor import EventExtractor
from embedding.vector_store import VectorMemory
from llm.query_agent import QuestionAnswerAgent
from utils.logger import logger


# Graceful shutdown handler
def signal_handler(sig, frame):
    logger.info("Shutting down Real-Time Video QA System...")
    sys.exit(0)


def main():
    logger.info("Starting Real-Time Video QA System...")

    # Register Ctrl+C signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize components
    try:
        stream = VideoStream()
        processor = FrameProcessor()
        extractor = EventExtractor()
        memory = VectorMemory()
        qa_agent = QuestionAnswerAgent(memory)
    except Exception as e:
        logger.critical(f"Initialization failed: {e}")
        sys.exit(1)

    logger.info("System ready. Processing live stream...")
    logger.info("Press Ctrl+C to stop the stream.")

    try:
        start_time = time.time()
        frame_count = 0

        for frame in stream.capture():
            frame_count += 1
            try:
                tags = processor.process(frame)
                if tags:
                    event = extractor.extract(tags)
                    memory.store(event)
                    logger.debug(f"Stored event: {event}")
            except Exception as e:
                logger.error(f"Error processing frame: {e}")

            if frame_count % 30 == 0:  # Log FPS every 30 frames
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                logger.info(f"Processed {frame_count} frames ({fps:.2f} FPS)")

            time.sleep(0.05)  # Control frame rate (20 fps approx)

    except KeyboardInterrupt:
        logger.info("Video stream stopped by user.")

    # QA Mode
    logger.info("Entering QA mode. Ask questions about what happened.")
    while True:
        try:
            query = input("\nAsk a question (or type 'exit' to quit): ")
            if query.strip().lower() in ['exit', 'quit']:
                logger.info("Exiting QA mode.")
                break
            answer = qa_agent.answer(query)
            logger.info(f"Answer: {answer}")
        except KeyboardInterrupt:
            logger.info("QA mode interrupted.")
            break
        except KeyboardInterrupt:
            logger.info("QA mode intercepted.")
            break
        except Exception as e:
            logger.error(f"Error during QA: {e}")


if __name__ == '__main__':
    main()
