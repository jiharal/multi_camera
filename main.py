from threading import Thread
from multicamera import MultiCapture, FramesThreadBody, NormalizerCLAHE
import time, sys, queue, cv2

import numpy as np


def check_pressed_keys(key):
    if key == 32:  # Pause
        while True:
            key = cv2.waitKey(0)
            if key == 27 or key == 32 or key == 13:  # enter: resume, space: next frame, esc: exit
                break
    else:
        key = cv2.waitKey(1)
    return key


def get_target_size(frame_sizes,
                    vis=None,
                    max_window_size=(1920, 1080),
                    stack_frames='vertical',
                    **kwargs):
    if vis is None:
        width = 0
        height = 0
        for size in frame_sizes:
            if width > 0 and height > 0:
                if stack_frames == 'vertical':
                    height += size[1]
                elif stack_frames == 'horizontal':
                    width += size[0]
            else:
                width, height = size
    else:
        height, width = vis.shape[:2]


    if stack_frames == 'vertical':
        target_height = max_window_size[1]
        target_ratio = target_height / height
        target_width = int(width * target_ratio)
    elif stack_frames == 'horizontal':
        target_width = max_window_size[0]
        target_ratio = target_width / width
        target_height = int(height * target_ratio)
    return target_width, target_height


def visualize_multicam_detections(frames,
                                  max_window_size=(1920, 1080),
                                  stack_frames='horizontal'):
    assert stack_frames in ['vertical', 'horizontal']
    vis = None
    for i, frame in enumerate(frames):
        if vis is not None:
            if stack_frames == 'vertical':
                vis = np.vstack([vis, frame])
            elif stack_frames == 'horizontal':
                vis = np.hstack([vis, frame])
        else:
            vis = frame
    # print(len(frames))
    target_width, target_height = get_target_size(frames, vis, max_window_size,
                                                  stack_frames)
    
    vis = cv2.resize(vis, (target_width, target_height))

    return vis


if __name__ == "__main__":
    print("Program started")
    sources = [
        "/home/thinkbook/Videos/cctv/cctv4.mp4",
        "/home/thinkbook/Videos/cctv/cctv3.mp4"
    ]
    capture = MultiCapture(sources=sources, loop=False)

    capture.add_transform(
        NormalizerCLAHE(clip_limit=1.0, tile_size=8)
    )

    thread_body = FramesThreadBody(capture,
                                   max_queue_length=len(capture.captures) * 2)
    frames_thread = Thread(target=thread_body)
    frames_thread.start()

    frames_read = False
    set_output_frames = False

    prev_frames = thread_body.frames_queue.get()

    key = -1
    frame_number = 0
    # stack_frames='vertical'
    # max_window_size=(1920, 1080)

    while thread_body.process:
        key = check_pressed_keys(key)
        if key == 27:
            break
        start = time.perf_counter()
        try:
            frames = thread_body.frames_queue.get_nowait()
            frames_read = True
        except queue.Empty:
            frames = None

        if frames is None:
            continue
        frame_number += 1
        # print(prev_frames)
        vis = visualize_multicam_detections(prev_frames)
        prev_frames, frames = frames, prev_frames
        # for frame in frames:
        #     print(len(frame))
        #     cv2.resize(frame, (1000, 1000))
        cv2.imshow("video", vis)

    thread_body.process = False
    frames_thread.join()
