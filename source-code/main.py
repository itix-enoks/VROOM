import sys
import time
import cv2 as cv
import pantilthat as pth
import numpy as np
import os
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from picamera2 import Picamera2
import algorithms.frame_difference as proc_naive
import algorithms.colored_frame_difference as proc_color


FRAME_WIDTH = 1332
FRAME_HEIGHT = 990

PREVIEW_MAIN_FRAME = False
PREVIEW_PROC_FRAME = True

RECORDING_ID = time.strftime('%y%m%d%H%M%S', time.gmtime())
OUTPUT_DIR = os.path.join("output_frames", RECORDING_ID)
os.makedirs(OUTPUT_DIR, exist_ok=True)


class SharedObject:
    y_measure: float = 1
    is_exit: bool = False
    frame = None
    frame_buffer = []


class CameraStream:
    def __init__(self, shared_obj, width=1332, height=990):
        self.shared = shared_obj
        self.width = width
        self.height = height
        self.picam2 = Picamera2()
        self._configure()
        self.thread = Thread(target=self._update_frames, daemon=True)
        self.frame_count = 0

    def _configure(self):
        fastmode = self.picam2.sensor_modes[0]
        config = self.picam2.create_preview_configuration(
            sensor={'output_size': fastmode['size'], 'bit_depth': fastmode['bit_depth']},
            controls={"FrameDurationLimits": (8333, 8333)}
        )
        self.picam2.configure(config)

    def start(self):
        self.picam2.start()
        time.sleep(1)
        self.thread.start()

    def _update_frames(self):
        while not self.shared.is_exit:
            frame = cv.cvtColor(self.picam2.capture_array("main"), cv.COLOR_BGR2RGB)
            rotated = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
            self.shared.frame = rotated
            self.frame_count += 1
            self.shared.frame_buffer.append(frame)

    def stop(self):
        self.picam2.stop()


def tilt(shared_obj, angle=-20):
    while True:
        if shared_obj.is_exit:
            sys.exit(0)
        pth.pan(0)
        pth.tilt(angle)
        time.sleep(5e-3)


def run_tasks_in_parallel(tasks):
    with ThreadPoolExecutor() as executor:
        running_tasks = [executor.submit(task) for task in tasks]
        for running_task in running_tasks:
            running_task.result()


COLOR_HUES = {
    "Red": 0,
    "Green": 60,
    "Blue": 120,
    "Cyan": 90,
    "Magenta": 150,
    "Yellow": 30,
    "Amber": 15,
    "Chartreuse": 45,
    "Spring Green": 75
    "Azure": 105,
    "Violet": 135,
    "Rose": 165
}


def process(shared_obj):
    prev_gray = None
    prev_time = time.time_ns()
    diff_time = 0
    frame_per_sec = 0
    frame_cnt_in_sec = 0
    is_one_sec_passed = False

    frame_buffer = []

    try:
        while True:
            if shared_obj.frame is None:
                continue
            curr_color = shared_obj.frame
            curr_gray = cv.cvtColor(curr_color, cv.COLOR_RGB2HSV)

            if prev_gray is None:
                output = curr_color
            else:
                _, output, _ = proc_color.process_frames(camera_prev_gray, current_gray_frame, current_frame, COLOR_HUES["Rose"], hue_tolerance=10)

            prev_gray = curr_gray
            frame_cnt_in_sec += 1
            curr_time = time.time_ns()
            diff_time += (curr_time - prev_time) / 1e6

            if is_one_sec_passed:
                cv.putText(output, f"FPS: {frame_per_sec}", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv.putText(output, f"FPS: (WAITING...)", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if int(diff_time) >= 1000:
                frame_per_sec = frame_cnt_in_sec
                frame_cnt_in_sec = 0
                diff_time = 0
                is_one_sec_passed = True

            prev_time = curr_time

            if PREVIEW_MAIN_FRAME:
                cv.imshow(f'[{RECORDING_ID}] [Live] Actual Frame', curr_color)

            if PREVIEW_PROC_FRAME:
                output = cv.cvtColor(output, cv.COLOR_RGB2BGR)
                cv.imshow(f'[{RECORDING_ID}] [Live] Processed Frame', output)

            if cv.waitKey(1) & 0xFF == ord('q'):
                shared_obj.is_exit = True
                for i, frame in enumerate(shared_obj.frame_buffer):
                    filename = os.path.join(OUTPUT_DIR, f"frame_{i:06d}.png")
                    cv.imwrite(filename, frame)

                print(f"[INFO] Saved {len(shared_obj.frame_buffer)} frames to: {OUTPUT_DIR}")
                break

    finally:
        cv.destroyAllWindows()


if __name__ == "__main__":
    shared_obj = SharedObject()
    camera = CameraStream(shared_obj)
    camera.start()
    run_tasks_in_parallel([lambda: process(shared_obj), lambda: tilt(shared_obj)])
    camera.stop()
