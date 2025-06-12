import os
import sys
import cv2
import time

from algorithms.color_filter import process_frames
from util import load_frames


test_instance_color_map = {
    "/35mm/1.1": (162, 6 , 90, 250, 30, 250),  # cost=612.5735294117648
    "/35mm/2.1": (162, 18, 30, 200, 30, 250),  # cost=774.3382352941177
    "/35mm/2.2": (6  , 10, 60, 250, 30, 250),  # cost=2697.4895833333335
    
    "/25mm/1.1": (60 , 10, 30, 250, 90, 200),  # cost=1895.35
    "/25mm/1.2": (156, 14, 30, 250, 30, 250),  # cost=1832.0714285714287
    "/25mm/1.3": (72 , 14, 30, 250, 30, 150),  # cost=1538.0
    "/25mm/2.1": (6  , 6,  90, 250, 30, 150),  # cost=2566.0208333333335
    "/25mm/2.2": (54 , 18, 30, 250, 30, 150),  # cost=595.6428571428571
}

recording_id = time.strftime('%y%m%d%H%M%S', time.gmtime())
dir_root = f"/Users/enoks/Downloads/{recording_id}/"
os.makedirs(dir_root, exist_ok=True)
for index, (test_key, test_val) in enumerate(sorted(list(test_instance_color_map.items()))):
    folder_path = "./training-set/moving-camera" + test_key
    frames_list = load_frames(folder_path)

    for i in range(0, len(frames_list)):
        frame =  frames_list[i]

        processor_output = process_frames(
            frame,
            test_val[0],
            test_val[1]
            )

        y, output_image, mask = processor_output

        cv2.imshow(f"[{test_key}] Colored frame differencing", output_image)
        cv2.imwrite(dir_root + f"frame_{index:06d}.jpg", frame)
        cv2.waitKey(1000)
