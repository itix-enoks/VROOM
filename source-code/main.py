import sys
import time

import cv2 as cv
import pantilthat as pth
import numpy as np

from picamera2 import Picamera2
from algorithms.background_subtraction import process_frames
from concurrent.futures import ThreadPoolExecutor


FRAME_WIDTH = 1332
FRAME_HEIGHT = 990

PREVIEW_MAIN_FRAME = False
PREVIEW_PROC_FRAME = True


class SharedObject(object):
    y_measure: float = 1
    is_exit: bool = False


def tilt(shared_obj, angle=-20):
    import pantilthat as pth
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


def process(shared_obj):
    recording_id = time.strftime('%y%m%d%H%M%S', time.gmtime())
    picam2 = Picamera2()

    fastmode = picam2.sensor_modes[0]
    cfg = picam2.create_preview_configuration(
        sensor={'output_size': fastmode['size'], 'bit_depth': fastmode['bit_depth']},
        controls={"FrameDurationLimits": (8333, 8333)}
    )
    picam2.configure(cfg)
    picam2.start()

    prev_gray = None
    prev_time = time.time_ns()
    diff_time = 0

    frame_per_sec = 0
    frame_cnt_in_sec = 0

    is_one_sec_passed = False

    try:
        while True:
            curr_color = cv.rotate(picam2.capture_array("main"), cv.ROTATE_90_CLOCKWISE)
            curr_gray = cv.cvtColor(curr_color, cv.COLOR_BGR2GRAY)

            if prev_gray is None:
                output = curr_color.copy()
            else:
                output = process_frames(prev_gray, curr_gray, curr_color)

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
                cv.imshow(f'[{recording_id}] [Live] Actual Frame', curr_color)

            if PREVIEW_PROC_FRAME:
                cv.imshow(f'[{recording_id}] [Live] Processed Frame', output)

            if cv.waitKey(1) & 0xFF == ord('q'):
                shared_obj.is_exit = True
                break

    finally:
        picam2.stop()
        cv.destroyAllWindows()


if __name__ == "__main__":
    shared_obj = SharedObject()
    run_tasks_in_parallel([lambda: process(shared_obj), lambda: tilt(shared_obj)])
    
