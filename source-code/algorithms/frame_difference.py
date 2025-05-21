import cv2 as cv
import numpy as np


def process_frames(prev_gray, curr_gray, curr_color):
    diff = cv.absdiff(curr_gray, prev_gray)
    _, bw = cv.threshold(diff, 5, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    bw = cv.morphologyEx(bw, cv.MORPH_OPEN, kernel)
    bw = cv.morphologyEx(bw, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    output = curr_color.copy()
    detections = []

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        area = w * h

        MIN_AREA_THRESH = 400
        if area > MIN_AREA_THRESH:
            detections.append([x, y, x + w, y + h, area])

    boxes = [det[:4] for det in detections]
    scores = [det[4] for det in detections]

    keep = _non_max_suppression(np.array(boxes), np.array(scores), 0)

    for box in keep:
        x1, y1, x2, y2 = box
        cv.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return output


def _remove_contained_bboxes(boxes):
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


def _non_max_suppression(boxes, scores, threshold):
    boxes = boxes[np.argsort(scores)[::-1]]
    order = _remove_contained_bboxes(boxes)
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
