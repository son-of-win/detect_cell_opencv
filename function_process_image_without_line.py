import cv2

def pre_process_image(img, morph_size=(10,2)):
    # chuyển đổi ảnh sang màu xám
    pre = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # chuyển ảnh sang binary áp dụng thuật toán phân ngưỡng ảnh OTSU
    pre = cv2.threshold(pre, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU )[1]
    # threshold = 128 là giá trị ngưỡng, những điểm ảnh có giá trị > 250 sẽ được gán lại bằng 255
    # xử lí hình thái học của ảnh
    # sử dụng phép giãn nở dilate để tăng kích thước của đối tượng trong ảnh
    copy_img = pre.copy()

    # tạo kernel matrix cho dilate
    kernel_maxtrix = cv2.getStructuringElement(cv2.MORPH_RECT, morph_size)
    copy_img = cv2.dilate(~copy_img, kernel=kernel_maxtrix, anchor=(-1, -1),iterations= 1) # iteration: số lần lặp lại của kernel trên ảnh
    pre = ~copy_img
    
    # cv2.imwrite('first_preprocess.png', pre)
    return pre

def remove_horizontal_line(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))

    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
    horizontal_kernel, iterations=2)

    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 2)

    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))

    result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel,
    iterations=1)
    return result

def remove_vertical_line(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))

    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
    vertical_kernel, iterations=2)

    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 2)

    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 1))

    result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel,
    iterations=1)
    return result

def find_text_boxes(pre_images, min_text_height_limit=2, max_text_height_limit=100, min_text_weight_limit = 1):
    contours, hierachy = cv2.findContours(pre_images, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    # tạo đường bounding box dựa trên các thông số về text size
    boxes = []
    for contour in contours:
        box = cv2.boundingRect(contour)
        height = box[3]
        weight = box[2]
        if min_text_height_limit < height < max_text_height_limit and weight >= min_text_weight_limit:
            boxes.append(box)

    return boxes

def get_contour_boxes(img, remove_convert, morph_size = (10,2)):
    if remove_convert:
        ver_img = remove_vertical_line(img)
        hor_img = remove_horizontal_line(ver_img) 
        pre_processed = pre_process_image(hor_img,morph_size) 
    else:
        hor_img = remove_horizontal_line(img)
        ver_img = remove_vertical_line(hor_img)
        pre_processed = pre_process_image(ver_img,morph_size)
    text_boxes = find_text_boxes(pre_processed)
    return text_boxes