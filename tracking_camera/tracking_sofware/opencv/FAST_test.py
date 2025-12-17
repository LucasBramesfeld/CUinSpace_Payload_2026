import cv2
import numpy as np

# --- Video input ---
video_path = 'C:/Users/lucas/OneDrive/Desktop/Rocket Tracking Camera/rocket_videos/IMG_5934.mov'  # Change to your video path
cap = cv2.VideoCapture(video_path)

# --- FAST feature detector ---
fast = cv2.ORB_create(10000)
#fast.setThreshold(10)   # Sensitivity, adjust as needed
#fast.setNonmaxSuppression(True)  # Keep strongest features only

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for feature detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect FAST keypoints
    keypoints = fast.detect(gray, None)

    # Draw keypoints on the frame
    frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0))

    # Show frame
    cv2.imshow('FAST Features', frame_with_keypoints)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()