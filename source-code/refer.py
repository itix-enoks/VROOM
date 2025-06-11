import os
import sys
import cv2
import time
import json

from algorithms.colored_frame_difference import process_frames


def main():
    images_dir = "./training-set/moving-camera/"
    
    instance_reference_xys_map = {
        "35mm/1.1": [], "35mm/2.1": [], "35mm/2.2": [],
        "25mm/1.1": [], "25mm/1.2": [], "25mm/1.3": [],
        "25mm/2.1": [], "25mm/2.2": []
    }

    for inst_idx, (instance, xys) in enumerate(instance_reference_xys_map.items(), 1):
        frames = load_frames(images_dir + instance)
        
        for frame_idx, frame in enumerate(frames, 1):
            display = frame.copy()
            status = f"{inst_idx}/{len(instance_reference_xys_map)} {instance} {frame_idx}/{len(frames)}"
            cv2.putText(display, status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            cv2.imshow("Reference: click on the center of the ball", display)
            cv2.setMouseCallback("Reference: click on the center of the ball", 
                               on_click, (instance, instance_reference_xys_map, frame_idx))
            
            if cv2.waitKey(60000) in (ord('c'), 27):
                continue

        with open(f"{replace_last(instance, "/", "_")}.json", 'w') as f:
            json.dump({instance: xys}, f)


def on_click(event, x, y, flags, param):
    global _user_input_lock
    if event == cv2.EVENT_LBUTTONDOWN:
        _user_input_lock = True

        instance = param[0]
        xys = param[1][param[0]]

        clicked_point = (x, y)
        xys.append((param[2], clicked_point))

        print(f"Reference for {instance} at {param[2]} in JSON: {(param[2], xys[len(xys) - 1])}")


def load_frames(folder_path):
    frame_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    frame_files = [f for f in frame_files if os.path.splitext(f)[1].lower() in image_extensions]

    frame_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

    frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(folder_path, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            frames.append(frame)
        else:
            print(f"Warning: Could not load image {frame_file}")

    return frames


def replace_last(source_string, replace_what, replace_with):
    head, _sep, tail = source_string.rpartition(replace_what)
    return head + replace_with + tail


if __name__ == "__main__":
    main()