import os
import sys
import cv2
import time

import algorithms.colored_frame_difference


def load_frames(folder_path):
    """
    Load all image frames from a folder into a sorted list.

    Args:
        folder_path (str): Path to the folder containing the frames

    Returns:
        list: Sorted list of frames (as numpy arrays)
    """
    # Get all files in the folder
    frame_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Filter for image files (you can add more extensions if needed)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    frame_files = [f for f in frame_files if os.path.splitext(f)[1].lower() in image_extensions]

    # Sort the files numerically (assuming they're named as frame_001.jpg, frame_002.jpg, etc.)
    frame_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

    # Load each image and store in list
    frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(folder_path, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            frames.append(frame)
        else:
            print(f"Warning: Could not load image {frame_file}")

    return frames


color_hues = {
    "Red": 10,            # Red at 0 hue (also spans to 180)
    "Green": 60,         # Green at 60 hue
    "Blue": 120,         # Blue at 120 hue
    "Cyan": 90,          # Cyan at 90 hue
    "Magenta": 150,      # Magenta at 150 hue
    "Yellow": 30,        # Yellow at 30 hue
    "Amber": 15,         # Amber at 15 hue
    "Chartreuse": 45,    # Chartreuse at 45 hue
    "Spring Green": 75,  # Spring Green at 75 hue
    "Azure": 105,        # Azure at 105 hue
    "Violet": 135,       # Violet at 135 hue
    "Rose": 165          # Rose at 165 hue
}

test_instance_color_map = {
    "/35mm/1.1": (160, 5, 50, 220, 30, 250),
    "/35mm/2.1": (150, 20, 130, 220, 130, 250),
    "/35mm/2.2": (20, 20, 30, 190, 30, 160),

    # "/25mm/1.1": color_hues["Green"],
    # "/25mm/1.2": color_hues["Rose"],
    # "/25mm/1.3": color_hues["Green"],
    # "/25mm/2.1": color_hues["Red"],
    # "/25mm/2.2": color_hues["Green"],
}

recording_id = time.strftime('%y%m%d%H%M%S', time.gmtime())
dir_root = f"/Users/enoks/Downloads/{recording_id}/"
os.makedirs(dir_root, exist_ok=True)
for index, (test_key, test_val) in enumerate(test_instance_color_map.items()):
    folder_path = "./training-set/moving-camera" + test_key
    frames_list = load_frames(folder_path)

    for i in range(1, len(frames_list)):
        frame_1, frame_2 = frames_list[i-1:i+1]


        processor_output = algorithms.colored_frame_difference.process_frames(frame_1,
                                                                              frame_2,
                                                                              frame_2,
                                                                              test_val[0],
                                                                              test_val[1],
                                                                              test_val[2],
                                                                              test_val[3],
                                                                              test_val[4],
                                                                              test_val[5])

        y, output_image, mask = processor_output

        cv2.imshow(f"[{test_key}] Colored frame differencing", output_image)
        cv2.imwrite(dir_root + f"frame_{index:06d}.jpg", frame_1)
        cv2.waitKey(1000)
