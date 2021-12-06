import cv2
import numpy as np
import matplotlib.pyplot as plt
from general_function import *
from function_process_image_with_line import *
from function_process_image_without_line import *
dir_path = 'test_images/'
def first_method_process(filename, list_low_accuracy):
    filename = filename.strip()
    # path = 'train_images/' + filename.strip() + '.png'
    file_path = dir_path + filename.strip() + '.png'
    xml_path = dir_path + filename.strip() + '.xml'
    img = cv2.imread(file_path)
    
    text_boxes = get_contour_boxes(img, remove_convert= False, morph_size=(10,2))
    xmins, xmaxs, ymins, ymaxs = get_key_xml(xml_path)
    accuracy_1 = caculate_accuracy(text_boxes,xmins, xmaxs, ymins, ymaxs)

    if(accuracy_1 < 0.6):
        text_boxes_2 = get_contour_boxes(img, remove_convert= True, morph_size=(10,2))
        accuracy_2 = caculate_accuracy(text_boxes_2,xmins, xmaxs, ymins, ymaxs)
        if(accuracy_2 < 0.6):
            list_low_accuracy.write(filename + "/" + str(accuracy_1) + "/" + str(accuracy_2) + "\n")
        else:
            print(filename + ":" + str(accuracy_2))
    else: 
        print(filename + ":" + str(accuracy_1))

# f = open('train_images.txt','r')
f = open('test_images.txt', 'r')
list_files = f.readlines()
f.close()
list_low_accuracy = open('list_low_accuracy.txt','w')
for filename in list_files:
    first_method_process(filename, list_low_accuracy)
list_low_accuracy.close()

## detect cell with line
improve_images = open('list_low_accuracy.txt','r')
list_improve_images = improve_images.readlines()
improve_images.close()
for image in list_improve_images:
    name_image, acc1, acc2 = image.split("/")
    path = dir_path + name_image.strip() + ".png"
    xml_path = dir_path + name_image.strip() + ".xml"
    boxes = detect_line(path,15)
    xmins, xmaxs, ymins, ymaxs = get_key_xml(xml_path)
    cnt = 0
    for box in boxes:
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3] 
            if cal_iou(x, x + w, y, y + h, xmins, xmaxs, ymins, ymaxs):
                cnt += 1
                # draw_box(img, x, x + w, y, y + h)
    accuracy_3 = float(cnt / len(xmins))
    print(name_image.strip() + ":" + str(max((float(acc1), float(acc2), accuracy_3))))

