import cv2 as cv
import numpy as np

def process_frames(prev_gray, curr_gray, curr_color, target_hue, hue_tolerance, s_min, s_max, v_min, v_max):
    # Convert the current color frame to HSV
    hsv = cv.cvtColor(curr_color, cv.COLOR_BGR2HSV)

    # Visualize HSV image for debugging purposes (optional)
    # cv.imshow("HSV Image", hsv)  # Uncomment to see the HSV image

    # Define the target color range in HSV (you can adjust the tolerance if needed)
    lower_bound = np.array([target_hue - hue_tolerance, s_min, v_min])  # Lower bound for target color
    upper_bound = np.array([target_hue + hue_tolerance, s_max, v_max])  # Upper bound for target color

    # Create a mask for the target color (binary image)
    mask = cv.inRange(hsv, lower_bound, upper_bound)

    # Visualize the mask to see if the target color is correctly captured (optional)
    # cv.imshow("Mask", mask)  # Uncomment to see the binary mask

    # Perform morphological operations to clean up the binary mask
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # Find contours based on the mask (which highlights the target color)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Initialize output image to draw detected boxes
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

    # Apply Non-Maximum Suppression to remove overlapping boxes
    keep = _non_max_suppression(np.array(boxes), np.array(scores), 0)

    # Draw bounding boxes on the output image for the detected color regions
    # NOTE: we only return non-None values if there is only *one* box detected
    if len(keep) != 1:
        return None, output, mask

    x1, y1, x2, y2 = keep[0]
    y_measured = (y1 + y2) / 2
    output = cv.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw red rectangle for the detected object

    return y_measured, output, mask # Output is just image with bounded rectangles for previewing purposes


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
