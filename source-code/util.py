import os
import cv2


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