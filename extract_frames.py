import cv2
import os

def extract_frames(video_path, output_folder, frame_interval):
    """
    Extract frames from a video every `frame_interval` frames and save them as images.

    :param video_path: Path to the input video file
    :param output_folder: Folder to save extracted frames
    :param frame_interval: Interval between frames to extract
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if frame_count % frame_interval == 0:
            # Construct filename and save frame
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames into '{output_folder}'.")

if __name__ == "__main__":
    video_path = "data/sequences/DJI_20251210134636_0001_S.mp4"      # Change to your video path
    output_folder = "data/frames"            # Change to desired output folder
    frame_interval = 30                 # Extract every Nth frame

    extract_frames(video_path, output_folder, frame_interval)
