import os
import random
import cv2

def label_image(image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]  # image dimensions

    screen_w, screen_h = 1300, 700
    scale = min(screen_w / w, screen_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    h, w = new_h, new_w
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
            box = (x1, y1, x2, y2)

            temp_img = clone.copy()
            cv2.rectangle(temp_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.imshow(window, temp_img)

    window = "Rocket Bounding Box Labeler"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, click_event)
    cv2.imshow(window, clone.copy())
    while True:
        key = cv2.waitKey(1) & 0xFF

        # Press ENTER to save
        if key == 13 and box is not None:
            os.makedirs("dataset/images", exist_ok=True)
            os.makedirs("dataset/labels", exist_ok=True)

            # bounding box in YOLO format
            x1, y1, x2, y2 = box
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            base_name = os.path.splitext(os.path.basename(image_path))[0]

            def save(img_to_save, suffix, flip=False):
                """Helper to save image + label"""
                out_img_path = f"dataset/images/{base_name}{suffix}.jpg"
                cv2.imwrite(out_img_path, img_to_save)
                out_label_path = f"dataset/labels/{base_name}{suffix}.txt"
                with open(out_label_path, "w") as f:
                    if flip:
                        f.write(f"0 {1 - x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")
                    else:
                        f.write(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n")

            # Original
            save(img, "", flip=False)

            # Flipped origina
            flipped = cv2.flip(img, 1)
            save(flipped, "_flip", flip=True)

            # Contrast variations
            contrast_levels = [0.5, 1.5]  # darker, brighter
            for i, alpha in enumerate(contrast_levels, start=1):
                adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
                save(adjusted, f"_contrast{i}", flip=False)

                # Flipped version of contrast image
                adjusted_flipped = cv2.flip(adjusted, 1)
                save(adjusted_flipped, f"_contrast{i}_flip", flip=True)

            print(f"Saved 6 images (original, flip, contrast + flips) for {base_name} \t bounding box ceneter:{(x_center,y_center)}")
            break

        # Press ESC to cancel
        if key == 8:
            break



# Directory
dir = 'unlabeled_images'
files = [f for f in os.listdir(dir)]

while True:
    random_file = random.choice(files)
    random_file_path = os.path.join(dir, random_file)

    box = label_image(random_file_path)

cv2.destroyAllWindows()