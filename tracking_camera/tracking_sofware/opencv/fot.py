import cv2
import numpy as np
import random

class FeatureTracker:
    def __init__(self, initial_frame, initial_box):
        self.detector = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.prev_image = initial_frame
        self.box = initial_box  # (x, y, w, h)

        # Detect keypoints/descriptors in the initial frame
        kp_all, des_all = self.detector.detectAndCompute(self.prev_image, None)

        # Separate points inside vs. outside the box
        self.kp_in, self.des_in, self.kp_out, self.des_out = self.split_keypoints_by_box(kp_all, des_all, self.box)

    def split_keypoints_by_box(self, keypoints, descriptors, box):
        x, y, w, h = box
        kp_in, des_in = [], []
        kp_out, des_out = [], []

        for i, kp in enumerate(keypoints):
            if x <= kp.pt[0] <= x + w and y <= kp.pt[1] <= y + h:
                kp_in.append(kp)
                if descriptors is not None:
                    des_in.append(descriptors[i])
            else:
                kp_out.append(kp)
                if descriptors is not None:
                    des_out.append(descriptors[i])

        des_in = np.array(des_in) if len(des_in) > 0 else None
        des_out = np.array(des_out) if len(des_out) > 0 else None

        return kp_in, des_in, kp_out, des_out
    
    def evaluate_box(self, box, old_box, kp_in, kp_out):
        x, y, w, h = box
        score = 0

        for kp in kp_in:
            if x <= kp.pt[0] <= x + w and y <= kp.pt[1] <= y + h:
                score += 1  # Reward inside points

        for kp in kp_out:
            if x <= kp.pt[0] <= x + w and y <= kp.pt[1] <= y + h:
                score -= 1  # Penalize outside points

        tau = 0.05 * (abs(old_box[0] - x) + abs(old_box[1] - y)) + 1 * (abs(old_box[2] - w) + abs(old_box[3] - h))

        return score - tau

    def update(self, new_frame, search_radius=100):
        kp, des = self.detector.detectAndCompute(new_frame, None)
        if des is None or len(kp) == 0:
            return self.prev_image  # nothing to update

        # Match with previous inside/outside descriptors
        matches_in = self.matcher.match(self.des_in, des) if self.des_in is not None else []
        matches_out = self.matcher.match(self.des_out, des) if self.des_out is not None else []

        # Filter out duplicate matches between in/out
        in_dist_map = {m.trainIdx: m.distance for m in matches_in}
        out_dist_map = {m.trainIdx: m.distance for m in matches_out}
        common_idxs = set(in_dist_map.keys()) & set(out_dist_map.keys())

        keep_in, keep_out = [], []
        for m in matches_in:
            if m.trainIdx not in common_idxs or in_dist_map[m.trainIdx] < 0.5 * out_dist_map[m.trainIdx]:
                keep_in.append(m)
        for m in matches_out:
            if m.trainIdx not in common_idxs or out_dist_map[m.trainIdx] >= 0.5 * in_dist_map[m.trainIdx]:
                keep_out.append(m)

        # Compute new keypoints positions for matched inside points
        if len(keep_in) > 0:
            pts_prev = np.array([self.kp_in[m.queryIdx].pt for m in keep_in])
            pts_new = np.array([kp[m.trainIdx].pt for m in keep_in])

            # Compute average motion
            delta = np.mean(pts_new - pts_prev, axis=0)
            dx, dy = delta

            # Optional: estimate scale (ratio of mean distances)
            if len(pts_prev) > 1:
                d_prev = np.linalg.norm(pts_prev - pts_prev.mean(axis=0), axis=1)
                d_new = np.linalg.norm(pts_new - pts_new.mean(axis=0), axis=1)
                scale = np.median(d_new / (d_prev + 1e-6))
                scale = max(0.9, min(1.1, scale))
            else:
                scale = 1.0

            # Update bounding box position and size
            x, y, w, h = self.box
            cx, cy = x + w / 2, y + h / 2
            new_w, new_h = w * scale, h * scale
            new_cx, new_cy = cx + dx, cy + dy
            new_box = (
                int(new_cx - new_w / 2),
                int(new_cy - new_h / 2),
                int(new_w),
                int(new_h)
            )
        else:
            new_box = self.box  # fallback if no matches

        # Evaluate box quality (optional)
        kp_in_matched = [kp[m.trainIdx] for m in keep_in]
        kp_out_matched = [kp[m.trainIdx] for m in keep_out]
        score = self.evaluate_box(new_box, self.box, kp_in_matched, kp_out_matched)

        # Update internal state
        self.prev_image = new_frame
        self.box = new_box
        self.kp_in, self.des_in, self.kp_out, self.des_out = self.split_keypoints_by_box(kp, des, new_box)
        

def label_box(img):
    h, w = img.shape[:2]  # image dimensions
    screen_w, screen_h = 800, 800
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
            return (x1, y1, x2 - x1, y2 - y1)

        # Press ESC to cancel
        if key == 8:
            break

if __name__ == "__main__":
    video_path = 'C:/Users/lucas/Downloads/Untitled video - Made with Clipchamp (18).mp4'
    cap = cv2.VideoCapture(video_path)

    tracker = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        #frame =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if tracker is None:
            box = label_box(frame)
            tracker = FeatureTracker(frame, box)
            continue
        
        
        match_img = tracker.update(frame)
        cv2.rectangle(frame, (tracker.box[0], tracker.box[1]), (tracker.box[0]+tracker.box[2], tracker.box[1]+tracker.box[3]), (0, 255, 0), 2)

        frame = cv2.resize(frame, (int(frame.shape[1]*0.5), int(frame.shape[0]*0.5)))
        cv2.imshow("Matches", frame)
        cv2.waitKey(200)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()