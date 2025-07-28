import cv2

class VideoStreamCUDA:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)  # Better backend for GPU decoding

        if not self.cap.isOpened():
            raise Exception("Could not open video source")

        # Try enabling hardware acceleration (depends on your GPU support in OpenCV build)
        self.gpu_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.gpu_available:
            print(f"CUDA detected: {cv2.cuda.getDevice()}")
            self.gpu_frame = cv2.cuda_GpuMat()
        else:
            print("CUDA not available, running on CPU.")

    def capture(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # If GPU is available, upload frame to GPU for faster processing
            if self.gpu_available:
                self.gpu_frame.upload(frame)
                frame = self.gpu_frame.download()  # Download back if needed
                # If just for display, you can keep it on GPU to save time

            yield frame

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
