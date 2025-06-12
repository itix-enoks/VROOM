import json
import os

import numpy as np
import cv2 as cv
from itertools import product

from algorithms.color_filter import process_frames
from util import load_frames


def _replace_last(source_string, replace_what, replace_with):
    head, _sep, tail = source_string.rpartition(replace_what)
    return head + replace_with + tail


def _load_expected_positions(json_path, key):
    with open(_replace_last(json_path, "/", "_"), "r") as f:
        data = json.load(f)
    return {int(frame): pos for frame, pos in data[key]}


def compute_cost(expected_dict, frames_data,
                 h_range, h_tol_range,
                 s_min_range, s_max_range,
                 v_min_range, v_max_range):

    best_params = None
    min_cost = float("inf")

    for h, htol, s_min, s_max, v_min, v_max in product(
        h_range, h_tol_range, s_min_range, s_max_range, v_min_range, v_max_range
    ):
        if s_min >= s_max or v_min >= v_max:
            continue  # skip invalid intervals

        cost = 0
        total_frames = 0

        for frame_idx, expected_pos in expected_dict.items():
            frame = frames_data[frame_idx]
            y_measured, _, _ = process_frames(
                frame, h, htol, s_min, s_max, v_min, v_max
            )

            y_expected = expected_pos[1]
            if y_measured is None:
                # Penalize missed detections heavily
                cost += 5000
            else:
                cost += (y_measured - y_expected) ** 2

            total_frames += 1

        avg_cost = cost / total_frames if total_frames > 0 else float("inf")

        if avg_cost < min_cost:
            min_cost = avg_cost
            best_params = (h, htol, s_min, s_max, v_min, v_max)

    return best_params, min_cost


instances = [
    "35mm/1.1",
    "35mm/2.1",
    "35mm/2.2",

    "25mm/1.1",
    "25mm/1.2",
    "25mm/1.3",
    "25mm/2.1",
    "25mm/2.2"
]

# There is someting wrong here, if I increase the tolerance, the H is reduced. It does not work for a meter
# DO IN THESIS DEFINETELY: Prove that including full range of S and V is computationaly impossible with big O notation
for instance in instances:
    images_path = "./training-set/moving-camera/" + instance
    json_path = f"./training-set/Batch 1/{instance}.json"

    expected = _load_expected_positions(json_path, instance)
    frames_data = load_frames(images_path)
    h_range = range(0, 180, 6)
    h_tol_range = range(2, 20, 4)
    s_min_range = range(30, 120, 30)
    s_max_range = range(150, 256, 50)
    v_min_range = range(30, 120, 30)
    v_max_range = range(150, 256, 50)


    best_out, best_cost = compute_cost(
        expected, frames_data, 
        h_range, h_tol_range,
        s_min_range, s_max_range, v_min_range, v_max_range
    )

    print(f"Final best Params for instance {instance} (hue, hue_tolerance): {best_out}, with cost: {best_cost}")