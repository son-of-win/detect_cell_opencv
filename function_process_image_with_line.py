import cv2
import numpy as np

def get_box_text(image_path, x_origin, y_origin):
    img = cv2.imread(image_path)
    image = cv2.threshold(~img, 128, 255, cv2.THRESH_BINARY)[1]
    height, weight = image.shape[:2]
    x = []
    y = []
    for i in range(5,height):
        for j in range(5,weight):
            if image[i][j][0] == 0 and image[i][j][1] == 0 and image[i][j][2] == 0:
                y.append(i)
                x.append(j)
    if len(x) > 0 and len(y) > 0: 
        xmin = min(x) - 3 + x_origin
        xmax = max(x) + x_origin
        ymin = min(y) - 3 + y_origin
        ymax = max(y) + y_origin
    else:
        xmin = xmax = ymin = ymax = -1
    return xmin, xmax, ymin, ymax

def detect_line(file_path, min_line_width):
    image = cv2.imread(file_path)
    height, width = image.shape[:2]
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold, img_bin = cv2.threshold(gray_scale, 150, 250, cv2.THRESH_BINARY)
    img = img_bin.copy()
    img_bin = ~img_bin
    # cv2.imwrite('temp/first_process.png', img)
    ### min size = min line width
    kernel_h = np.ones((1, min_line_width), np.uint8)
    kernel_v = np.ones((min_line_width, 1), np.uint8)

    img_bin_hor = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel_h)
    img_bin_ver = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel_v)
    img_bin_final = img_bin_hor | img_bin_ver
    final_kernel = np.ones((3,3), np.uint8)
    img_bin_final = cv2.dilate(img_bin_final, final_kernel, iterations=1)

    boxes = []
    # cv2.imwrite('temp/dilate.png', img_bin_final)
    ret, label, stats, centroids = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)
    ROI_number = 0
    for x, y, w, h, area in stats[1:]:
        if h > 50 or w > 50 and h != height and w != width:
            ROI = img_bin[y:y+h, x:x+w]
            cv2.imwrite('temp/ROI_{}.png'.format(ROI_number), ROI)
            xmin , xmax, ymin, ymax = get_box_text('temp/ROI_{}.png'.format(ROI_number), x,y)
            if xmin != -1 and xmax != -1 and ymin != -1 and ymax != -1:
                cv2.circle(image, (xmin,ymin), radius=3, color=(124, 252, 0), thickness=-1)
                cv2.circle(image, (xmax,ymin), radius=3, color=(181, 27, 14), thickness=-1)
                boxes.append((xmin, ymin, xmax - xmin , ymax - ymin))
            # result = draw_box(image, xmin, xmax, ymin, ymax)
            ROI_number += 1
    return boxes