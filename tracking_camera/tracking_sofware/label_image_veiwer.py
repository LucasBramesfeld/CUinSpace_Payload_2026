import os
import cv2

def draw_bounding_box(image, box_info):
    h, w = image.shape[:2]
    xc,yc = box_info[0]*w,box_info[1]*h
    boxw, boxh = box_info[2]*w, box_info[3]*h
    box = (int(xc-boxw/2),int(yc-boxh/2)),(int(xc+boxw/2),int(yc+boxh/2))
    cv2.rectangle(image, box[0], box[1], (0, 255, 0), 1)
    cv2.imshow("test",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def compare_bounding_box(image, true_box_info, pred_box_info):
    h, w = image.shape[:2]
    # Draw true box
    xc,yc = true_box_info[0]*w,true_box_info[1]*h
    boxw, boxh = true_box_info[2]*w, true_box_info[3]*h
    box = (int(xc-boxw/2),int(yc-boxh/2)),(int(xc+boxw/2),int(yc+boxh/2))
    cv2.rectangle(image, box[0], box[1], (0, 255, 0), 1)
    # Draw predicted box
    xc,yc = pred_box_info[0]*w,pred_box_info[1]*h
    boxw, boxh = pred_box_info[2]*w, pred_box_info[3]*h
    box = (int(xc-boxw/2),int(yc-boxh/2)),(int(xc+boxw/2),int(yc+boxh/2))
    cv2.rectangle(image, box[0], box[1], (0, 0, 255), 1)

    cv2.imshow("test",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    image_dir = 'dataset/images'
    images = [f for f in os.listdir(image_dir)]
    label_dir = 'dataset/labels'
    labels = [f for f in os.listdir(label_dir)]


    test_index = 1445
    test_image = cv2.imread(os.path.join(image_dir, images[test_index]))
    h, w = test_image.shape[:2]
    with open(os.path.join(label_dir, labels[test_index]), "r") as file:
        loc = file.read().split()[1:5]
        loc = [float(x) for x in loc]

    draw_bounding_box(test_image,loc)