import cv2 as cv
import time

from picamera2 import Picamera2
from algorithms.background_subtraction import process_frames


FRAME_WIDTH = 1332
FRAME_HEIGHT = 990

PREVIEW_MAIN_FRAME = False
PREVIEW_PROC_FRAME = True


def main():
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
            curr_color = picam2.capture_array("main")
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
                break
    finally:
        picam2.stop()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
