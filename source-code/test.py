import os
import sys
import cv2

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


# from algorithms.frame_difference import process_frames
# for i in range(1, len(frames_list)):
#     frame_1, frame_2 = frames_list[i-1:i+1]
#     frame_1_g = cv2.cvtColor(frame_1, cv2.COLOR_RGB2GRAY)
#     frame_2_g = cv2.cvtColor(frame_2, cv2.COLOR_RGB2GRAY)

#     out = process_frames(frame_1_g, frame_2_g, frame_2)
#     cv2.imshow("[Preview] Naive frame differencing", out)
#     cv2.waitKey(1000)


color_hues = {
    "Red": 0,            # Red at 0 hue (also spans to 180)
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
    "/35mm/1.1": color_hues["Rose"],
    "/35mm/2.1": color_hues["Rose"],
    "/35mm/2.2": color_hues["Red"],

    "/25mm/1.1": color_hues["Green"],
    "/25mm/1.2": color_hues["Rose"],
    "/25mm/1.3": color_hues["Green"],
    # "/25mm/2.1": color_hues["Red"],
    "/25mm/2.2": color_hues["Green"],
}

for test_key, test_val in test_instance_color_map.items():
    folder_path = "./training-set/moving-camera" + test_key
    frames_list = load_frames(folder_path)

    for i in range(1, len(frames_list)):
        frame_1, frame_2 = frames_list[i-1:i+1]

        # Call the process_frames function (it can return both the output and the mask)
        processor_output = algorithms.colored_frame_difference.process_frames(frame_1,
                                                                              frame_2,
                                                                              frame_2,
                                                                              target_hue=test_val,
                                                                              hue_tolerance=10)

        # The first element is the output image, the second is the mask
        y, output_image, mask = processor_output

        # Show the masked frame differencing (the mask)
        cv2.imshow("[Preview] Masked frame differencing", mask)

        # Show the colored frame differencing (the output image with tracked rectangles)
        cv2.imshow("[Preview] Colored frame differencing", output_image)

        # Wait for a key press, 1000 ms (1 second) between frames
        if cv2.waitKey(10000) == ord("q"):
            sys.exit(0)
