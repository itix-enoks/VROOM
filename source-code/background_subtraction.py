from picamera2 import Picamera2
import cv2
import numpy as np
import time

# Parameters
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480
MIN_AREA_THRESH = 400
TEST_CURR_FRAME = False
TEST_PROC_FRAME = True
OUTPUT_FILENAME = 'tracked_output_follow.mp4'
OUTPUT_FPS      = 120

def remove_contained_bboxes(boxes):
    check_array = np.array([True, True, False, False])
    keep = list(range(len(boxes)))
    for i in keep[:]:
        for j in range(len(boxes)):
            if i != j and np.all((np.array(boxes[j]) >= np.array(boxes[i])) == check_array):
                try:
                    keep.remove(j)
                except ValueError:
                    continue
    return keep

def non_max_suppression(boxes, scores, threshold):
    boxes = boxes[np.argsort(scores)[::-1]]
    order = remove_contained_bboxes(boxes)
    keep = []

    while order:
        i = order.pop(0)
        keep.append(i)
        new_order = []
        for j in order:
            xx1 = max(boxes[i][0], boxes[j][0])
            yy1 = max(boxes[i][1], boxes[j][1])
            xx2 = min(boxes[i][2], boxes[j][2])
            yy2 = min(boxes[i][3], boxes[j][3])

            inter_area = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            area_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
            area_j = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
            union = area_i + area_j - inter_area

            iou = inter_area / union if union != 0 else 0

            if iou <= threshold:
                new_order.append(j)
        order = new_order

    return boxes[keep]

def process_frames(prev_gray, curr_gray, curr_color):
    diff = cv2.absdiff(curr_gray, prev_gray)
    _, bw = cv2.threshold(diff, 5, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = curr_color.copy()
    detections = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > MIN_AREA_THRESH:
            detections.append([x, y, x + w, y + h, area])

    boxes = [det[:4] for det in detections]
    scores = [det[4] for det in detections]

    keep = non_max_suppression(np.array(boxes), np.array(scores), 0)

    for box in keep:
        x1, y1, x2, y2 = box
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return output

def main():
    picam2 = Picamera2()

    cfg = picam2.create_video_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
        controls={"FrameRate": OUTPUT_FPS},
        buffer_count=6
    )
    picam2.configure(cfg)

    picam2.set_controls({
        "FrameDurationLimits": (8333, 8333),
        "NoiseReductionMode": 0,
        "AeEnable": False,
        "AwbEnable": False
    })

    picam2.start()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        OUTPUT_FILENAME,
        fourcc,
        OUTPUT_FPS,
        (FRAME_WIDTH, FRAME_HEIGHT)
    )

    prev_gray = None
    prev_time = time.time()

    try:
        while True:
            curr_color = picam2.capture_array("main")
            curr_gray = cv2.cvtColor(curr_color, cv2.COLOR_BGR2GRAY)

            if prev_gray is None:
                output = curr_color.copy()
            else:
                output = process_frames(prev_gray, curr_gray, curr_color)

            prev_gray = curr_gray
            writer.write(output)

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time

            cv2.putText(output, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if TEST_CURR_FRAME:
                cv2.imshow("Raw", curr_color)
            if TEST_PROC_FRAME:
                cv2.imshow("Processed", output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        picam2.stop()
        writer.release()
        cv2.destroyAllWindows()
        print(f"INFO: saved tracked video to '{OUTPUT_FILENAME}'")

if __name__ == "__main__":
    main()
