import json
import os

import numpy as np
import cv2 as cv
from itertools import product

from algorithms.colored_frame_difference import process_frames


def replace_last(source_string, replace_what, replace_with):
    head, _sep, tail = source_string.rpartition(replace_what)
    return head + replace_with + tail


def load_frames(folder_path):
    frame_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    frame_files = [f for f in frame_files if os.path.splitext(f)[1].lower() in image_extensions]
    frame_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(folder_path, frame_file)
        frame = cv.imread(frame_path)
        if frame is not None:
            frames.append(frame)
        else:
            print(f"Warning: Could not load image {frame_file}")

    return frames


def load_expected_positions(json_path, key):
    with open(replace_last(json_path, "/", "_"), "r") as f:
        data = json.load(f)
    return {int(frame): pos for frame, pos in data[key]}


def compute_cost(expected_dict, frames_data,
                 h_range, h_tol_range,
                 s_min_range, s_max_range,
                 v_min_range, v_max_range):

    best_params = None
    min_cost = float("inf")
    MISSING_Y_PENALTY = 1000

    for h, htol, smin, smax, vmin, vmax in product(
            h_range, h_tol_range, s_min_range, s_max_range, v_min_range, v_max_range):

        if smax <= smin or vmax <= vmin:
            continue  # skip invalid ranges

        cost = 0
        count = 0
        for frame_idx, expected_pos in expected_dict.items():
            i = frame_idx - 1
            if i <= 0 or i >= len(frames_data):
                continue

            prev_frame = frames_data[i - 1]
            curr_frame = frames_data[i]

            prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
            curr_gray = cv.cvtColor(curr_frame, cv.COLOR_BGR2GRAY)

            print(f"Trying Params (hue, hue_tolerance, min_saturation, max_saturation, min_value, max_value: {h, htol, smin, smax, vmin, vmax}")
            y_measured, _, _ = process_frames(
                prev_gray, curr_gray, curr_frame,
                h, htol, smin, smax, vmin, vmax
            )

            y_expected = expected_pos[1]
            if y_measured is None:
                cost += MISSING_Y_PENALTY ** 2
            else:
                cost += (y_measured - y_expected) ** 2

            count += 1

        if count > 0:
            avg_cost = cost / count
            if avg_cost < min_cost:
                min_cost = avg_cost
                best_params = (h, htol, smin, smax, vmin, vmax)
                print(f"Found better Params (hue, hue_tolerance, min_saturation, max_saturation, min_value, max_value: {best_params}")
    return best_params, min_cost


instance = "35mm/2.2"
images_path = "./training-set/moving-camera/" + instance
json_path = f"./training-set/Batch 1/{instance}.json"

expected = load_expected_positions(json_path, instance)
frames_data = load_frames(images_path)
h_range = range(0, 180, 10)
h_tol_range = range(0, 21, 1)
s_min_range = range(0, 255, 30)
s_max_range = range(1, 256, 30)
v_min_range = range(0, 255, 30)
v_max_range = range(1, 256, 30)

best_hsv, best_cost = compute_cost(
    expected, frames_data, 
    h_range, h_tol_range, 
    s_min_range, s_max_range, 
    v_min_range, v_max_range)

print(f"Final best Params (hue, hue_tolerance, min_saturation, max_saturation, min_value, max_value: {best_hsv}, with cost: {best_cost}")
