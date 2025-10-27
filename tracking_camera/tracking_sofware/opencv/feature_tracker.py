import cv2
import numpy as np

class FeatureTracker:
    def __init__(self, initial_box):
        self.detector = cv2.ORB_create(1000) 
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_box = initial_box
        self.prev_boxes = [initial_box]

    def get_matches(self, image):
        # Find the keypoints and descriptors with ORB
        kp1, des1 = self.detector.detectAndCompute(self.prev_boxes[-1], None)

        kp2, des2 = self.detector.detectAndCompute(image, None)
        # Find matches
        #matches = self.matcher.knnMatch(des1, des2, k=2)
        matches = self.matcher.match(des1, des2)

        # Find the matches there do not have a too high distance
        good = matches#sorted(matches, key = lambda x:x.distance)[:10]

        draw_params = dict(
            matchColor=-1,  # draw matches in green color
            singlePointColor=None,
            matchesMask=None,  # draw only inliers
            flags=2
        )

        img3 = None

        # Get the image points from the good matches
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])

        mean_x = int(np.mean(q2[:, 0]))
        mean_y = int(np.mean(q2[:, 1]))
            
        new_box = (mean_x - self.prev_box.shape[1] // 2,
                   mean_x + self.prev_box.shape[1] // 2,
                   mean_y - self.prev_box.shape[0] // 2,
                   mean_y + self.prev_box.shape[0] // 2)
        
        cv2.rectangle(image, (new_box[0], new_box[2]), (new_box[1], new_box[3]), (255, 0, 0), 2)
        img3 = cv2.drawMatches(self.prev_boxes[-1], kp1, image, kp2, good, None, **draw_params)

        self.prev_box = image[new_box[2]:new_box[3], new_box[0]:new_box[1]]
        self.prev_boxes.append(self.prev_box)

        return img3
    
def cut_box(img):
    h, w = img.shape[:2]  # image dimensions
    screen_w, screen_h = 1000, 500
    scale = min(screen_w / w, screen_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    rh, rw = h/new_h, w/new_w
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    clone = resized.copy()
    box = None
    drawing = {"status": False, "ix": -1, "iy": -1}

    def click_event(event, x, y, flags, param):
        nonlocal box, clone

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing["status"] = True
            drawing["ix"], drawing["iy"] = x, y

        elif event == cv2.EVENT_MOUSEMOVE and drawing["status"]:
            temp_img = clone.copy()
            cv2.rectangle(temp_img, (drawing["ix"], drawing["iy"]), (x, y), (0, 255, 0), 1)
            cv2.imshow(window, temp_img)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing["status"] = False
            x1, y1 = min(drawing["ix"], x), min(drawing["iy"], y)
            x2, y2 = max(drawing["ix"], x), max(drawing["iy"], y)
            box = ((int(x1*rw), int(y1*rh), int(x2*rw), int(y2*rh)))

            temp_img = clone.copy()
            cv2.rectangle(temp_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.imshow(window, temp_img)

    window = "Rocket Bounding Box Labeler"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, click_event)
    cv2.imshow(window, clone.copy())
    while True:
        key = cv2.waitKey(1) & 0xFF

        # Press enter to save
        if key == 13 and box is not None:
            # bounding box in YOLO format
            x1, y1, x2, y2 = box
            return img[y1:y2, x1:x2]

        # Press ESC to cancel
        if key == 8:
            break

if __name__ == "__main__":
    video_path = 'C:/Users/lucas/Downloads/Untitled video - Made with Clipchamp (18).mp4'
    cap = cv2.VideoCapture(video_path)

    tracker = None
    matches = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        #frame =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if tracker is None:
            box_img = cut_box(frame)
            tracker = FeatureTracker(box_img)
            continue
        try:
            matches = cv2.resize(tracker.get_matches(frame), (int(frame.shape[1]*0.5), int(frame.shape[0]*0.5)))
        except Exception as e:
            pass
        finally:
            cv2.imshow("Matches", matches)
            cv2.waitKey(33)

    cap.release()
    cv2.destroyAllWindows()