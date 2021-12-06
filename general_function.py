import cv2
import numpy as np
from xml.dom import minidom

def draw_key(img, xmins, xmaxs, ymins, ymaxs):
    n = len(xmins)

    for i in range(0, n):
        xmin = int(xmins[i].firstChild.data)
        xmax = int(xmaxs[i].firstChild.data)
        ymin = int(ymins[i].firstChild.data)
        ymax = int(ymaxs[i].firstChild.data)
        color = (124,252,0)
        img = cv2.line(img, (xmin, ymin), (xmax, ymin), color=color, thickness=1)
        img = cv2.line(img, (xmin, ymin), (xmin, ymax), color=color, thickness=1)
        img = cv2.line(img, (xmax, ymin), (xmax, ymax), color=color, thickness=1)
        img = cv2.line(img, (xmin, ymax), (xmax, ymax), color=color, thickness=1)

def cal_iou(x1, x2, y1, y2, xmins, xmaxs, ymins, ymaxs):
    n = len(xmins)

    for i in range(0, n):
        xmin = int(xmins[i].firstChild.data)
        xmax = int(xmaxs[i].firstChild.data)
        ymin = int(ymins[i].firstChild.data)
        ymax = int(ymaxs[i].firstChild.data)

        x_left = max(x1, xmin)
        x_right = min(x2, xmax)
        y_top = max(y1, ymin)
        y_bottom = min(y2, ymax)

        if x_right < x_left or y_bottom < y_top:
            iou = 0
        else:
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (xmax - xmin) * (ymax - ymin)

            iou = intersection_area / float(area1 + area2 - intersection_area)

        if iou > 0.5:
            return True

    return False

def draw_box(img, xmin, xmax, ymin, ymax):
    img = cv2.line(img, (xmin, ymin), (xmin, ymax), color=(255, 0, 0), thickness=1)
    img = cv2.line(img, (xmin, ymin), (xmax, ymin), color=(255, 0, 0), thickness=1)
    img = cv2.line(img, (xmin, ymax), (xmax, ymax), color=(255, 0, 0), thickness=1)
    img = cv2.line(img, (xmax, ymin), (xmax, ymax), color=(255, 0, 0), thickness=1)
    # cv2.imwrite('box1.png',img)
    return img

def get_key_xml(file_path):
    table_data = minidom.parse(file_path)
    # table_data = minidom.parse('test_xml/' + filename[:-4] + 'xml')
    xmins = table_data.getElementsByTagName('xmin')
    xmaxs = table_data.getElementsByTagName('xmax')
    ymins = table_data.getElementsByTagName('ymin')
    ymaxs = table_data.getElementsByTagName('ymax')
    return xmins, xmaxs, ymins, ymaxs

def caculate_accuracy(text_boxes, xmins, xmaxs, ymins, ymaxs):
    cnt = 0
    for box in text_boxes:
        x = box[0] + 3
        y = box[1]
        w = box[2] - 10
        h = box[3] 
        if cal_iou(x, x + w, y, y + h, xmins, xmaxs, ymins, ymaxs):
            cnt += 1
    return float(cnt / len(xmins))




