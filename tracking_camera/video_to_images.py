import cv2
import os

# Function to extract frames from a video until reaching the desired frame count
def extract_frames(video_file, output_directory="unlabeled_images", step=20):
    cap = cv2.VideoCapture(video_file)

    # Get the video file's name without extension
    video_name = os.path.splitext(os.path.basename(video_file))[0]

    os.makedirs(output_directory, exist_ok=True)

    frame_count = 0
    saved_count = 0

    while True:
        # Jump directly to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame in the main folder
        output_file = os.path.join(output_directory, f"{video_name}_{frame_count}.jpg")
        cv2.imwrite(output_file, frame)
        saved_count += 1

        # Log every 50 frames to reduce slowdown
        if saved_count % 50 == 0:
            print(f"Saved {saved_count} frames from {video_name}...")

        frame_count += step

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    dir = 'rocket_videos'
    files = [f for f in os.listdir(dir)]
    for file in files:
        video_file = os.path.join(dir, file)
        extract_frames(video_file)