import cv2
from image_capture import start_capture
import queue, time


class FramesThreadBody:
    def __init__(self, capture, max_queue_length=2):
        self.process = True
        self.frames_queue = queue.Queue()
        self.capture = capture
        self.max_queue_length = max_queue_length

    def __call__(self):
        while self.process:
            if self.frames_queue.qsize() > self.max_queue_length:
                time.sleep(0.1)
                continue
            has_frames, frames = self.capture.get_frames()
            if not has_frames and self.frames_queue.empty():
                self.process = False
                break
            if has_frames:
                self.frames_queue.put(frames)


class MultiCapture:
    def __init__(self, sources, loop):
        assert sources
        self.captures = []
        self.transforms = []
        self.fps = []
        for src in sources:
            capture = start_capture(src, loop)
            self.captures.append(capture)
            self.fps.append(capture.fps())

    def add_transform(self, t):
        self.transforms.append(t)

    def get_frames(self):
        frames = []
        for capture in self.captures:
            frame = capture.read()
            if frame is not None:
                for t in self.transforms:
                    frame = t(frame)
                frames.append(frame)

        return len(frames) == len(self.captures), frames

    def get_num_sources(self):
        return len(self.captures)

    def get_fps(self):
        return self.fps


class NormalizerCLAHE:
    def __init__(self, clip_limit=.5, tile_size=16):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                     tileGridSize=(tile_size, tile_size))

    def __call__(self, frame):
        for i in range(frame.shape[2]):
            frame[:, :, i] = self.clahe.apply(frame[:, :, i])
        return frame