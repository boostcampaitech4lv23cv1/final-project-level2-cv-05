import os
import glob
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from itertools import combinations

'''
label_path = './bootcamp/dataset/labels/'
label_list = sorted(os.listdir(label_path))

# sessions 리스트에 session 단위로
sessions = []
prefixes = [str(i) + '_' for i in range(1, 13)] + ['3rd Run_']
for prefix in prefixes:
    session = [(label[:-4] + '.jpg', label) for label in label_list if label.startswith(prefix)]
    sessions.append(session)

# 시각화 함수
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(124)]
'''

def draw_box(box, image, color, class_number=None):
    x1, x2, y1, y2 = [int(c) for c in box]
    thickness = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=thickness, lineType=cv2.LINE_AA)
    if class_number is not None:
        cv2.putText(image, str(class_number), (int(x1*0.98), int(y1*0.98)), cv2.FONT_ITALIC, 2, color, 2, cv2.LINE_AA)

        
def draw_bboxes(image, labels, colors=None):
    """
    image : 이미지
    labels : label [클래스, min_x, min_y, width, height] 들이 담겨있는 리스트
    """
    if colors is None:
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(124)]
    height, width, channels = image.shape
    for cl, x, y, w, h in labels:
        cl = int(cl)
        x = float(x) * width
        y = float(y) * height
        w = float(w) * width
        h = float(h) * height
        x1, x2 = round(x - w / 2), round(x + w / 2)
        y1, y2 = round(y - h / 2), round(y + h / 2)
        draw_box([x1, x2, y1, y2], image, colors[cl], cl)
        
def get_label(path):
    with open(path) as f:
        r = csv.reader(f, delimiter=' ')
        label_list = list(r)
    return label_list

def get_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_img_label(img_path, label_path):
    img = get_img(img_path)
    label = get_label(label_path)
    return img, label

def get_box_coord(label, width, height):
    """
    x1 : min_x
    x2 : max_x
    y1 : min_y
    y2 : max_y
    """
    cl, x, y, w, h = label
    cl = int(cl)
    x = float(x) * width
    y = float(y) * height
    w = float(w) * width
    h = float(h) * height
    x1, x2 = round(x - w / 2), round(x + w / 2)
    y1, y2 = round(y - h / 2), round(y + h / 2)
    return x1, x2, y1, y2

def box_coord_to_yolo_label(cl, box, width, height):
    x1, x2, y1, y2 = box
    x = (x1 + x2) / 2 / width
    y = (y1 + y2) / 2 / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return [cl, x, y, w, h]

def crop_image(img, box, center_crop=False):
    x1, x2, y1, y2 = box
    if center_crop:
        # center crop 시, 기존 w, h의 84%에 해당하는 길이 즉, 0.84*0.84=70% center 영역 crop
        center_x, center_y = int((x1+x2)/2), int((y1+y2)/2)
        w, h = x2-x1, y2-y1
        center_crop_w, center_crop_h = 0.84*w, 0.84*h 
        center_crop_w_2, center_crop_h_2 = int(center_crop_w/2), int(center_crop_h/2)
        cropped_img = img[center_y-center_crop_h_2:center_y+center_crop_h_2, center_x-center_crop_w_2:center_x+center_crop_w_2]
        if cropped_img.size==0:  # ceter crop 된 영역이 너무 작아서, 빈 리스트인 경우 center crop x -> 일반 crop
            cropped_img = img[y1:y2, x1:x2]
    else:
        cropped_img = img[y1:y2, x1:x2]
        
    return cropped_img

def resize(src, src_area, dst_area, dst_w, dst_h):
    if src_area == 0 or dst_area ==0:  # bbox 매우 작아서 area 0이면 pass
        return None
    if src_area > dst_area:
        resized = cv2.resize(src, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(src, (dst_w, dst_h), interpolation=cv2.INTER_CUBIC)
    # elif src_area < dst_area:
    #     resized = cv2.resize(src, (dst_w, dst_h), interpolation=cv2.INTER_CUBIC)
    # else:
    #     resized = src
        
    return resized

def resize_paste(src, src_area, dst, dst_area, dst_w, dst_h, dst_box):
    resized = resize(src, src_area, dst_area, dst_w, dst_h)
    if resized is None:
        return None
    pasted_image = dst.copy()
    pasted_image[dst_box[2]:dst_box[3], dst_box[0]:dst_box[1]] = resized
    
    return pasted_image


def zero_padding(src, src_w, src_h, dst_w, dst_h):
    shift_x = (dst_w - src_w) / 2
    shift_y = (dst_h - src_h) / 2
    m = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    dsize = (dst_w, dst_h)
    padded = cv2.warpAffine(src, m, dsize)
    return padded


def padding_paste(src, src_w, src_h, dst, dst_w, dst_h, dst_box):
    padded = zero_padding(src, src_w, src_h, dst_w, dst_h)
    dst[dst_box[2]:dst_box[3], dst_box[0]:dst_box[1]] = padded


def common_paste(src, src_area, src_w, src_h, dst, dst_area, dst_w, dst_h, dst_box):
    if src_area > dst_area:
        w1 = (src_w - dst_w) // 2
        w2 = src_w - dst_w - w1
        h1 = (src_h - dst_h) // 2
        h2 = src_h - dst_h - h1
        x1, x2 = dst_box[0] - w1, dst_box[1] + w2
        y1, y2 = dst_box[2] - h1, dst_box[3] + h1
    elif src_area < dst_area:
        w1 = (dst_w - src_w) // 2
        w2 = dst_w - src_w - w1
        h1 = (dst_h - src_h) // 2
        h2 = dst_h - src_h - h1
        x1, x2 = dst_box[0] + w1, dst_box[1] - w2
        y1, y2 = dst_box[2] + h1, dst_box[3] - h2
    else:
        x1, x2, y1, y2 = dst_box
    dst[y1:y2, x1:x2] = src
    return [x1, x2, y1, y2]

#fit paste 추가
def fit_paste(src, src_area, src_w, src_h, dst, dst_area, dst_w, dst_h, dst_box):
    src_slope = src_h/src_w
    dst_slope = dst_h/dst_w
    result = dst.copy()
    
    if (src_slope < 1 and dst_slope > 1) or (src_slope > 1 and dst_slope < 1):
        src = cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE)
        temp = src_w
        src_w = src_h
        src_h = temp
        
    if dst_w/src_w > dst_h/src_h:
        src_w = dst_w
        src_h *= dst_w/src_w
        src_h = int(src_h)
    else:
        src_w *= dst_h/src_h
        src_w = int(src_w)
        src_h = dst_h
    
    dst_img_h, dst_img_w, _ = dst.shape
    
    x = (dst_box[0]+dst_box[1])//2
    y = (dst_box[2]+dst_box[3])//2
    x1 = max(0,x - src_w//2)
    y1 = max(0,y - src_h//2)
    x2 = min(dst_img_w, x1 + src_w)
    y2 = min(dst_img_h, y1 + src_h)
    result[y1:y2, x1:x2] = cv2.resize(src, (x2-x1,y2-y1))
    
    return result, [x1,x2,y1,y2]

def crop_paste(src_img, src_box, dst_img, dst_box, paste_method='resize', center_crop=False):
    """
    src_img의 box를 crop해서 dst_img의 box 위치에 paste
    
    src_img : crop할 image
    src_box : crop할 src_img의 bbox
    dst_img : paste할 image
    dst_box : paste할 dst_img의 bbox
    paste_method : 'resize', 'padding', 'paste'
        'resize' : crop된 img1의 box를 img2의 box와 동일한 크기로 resize
        'padding' : img2의 box가 더 클 때 남는 부위를 zero padding, 작을 때는 'paste'와 동일
        'paste' : 그냥 붙여넣는다.
    center_crop : 기존 box 영역의 약 70%에 해당하는 center 영역을 crop, default:False
    """
    #paste_method fit 추가됨
    if paste_method not in ['resize', 'fit']:
        raise ValueError("invalid paste_method")

    
    # return할 label 값. crop할 박스의 class + paste될 box
    ret_box = src_box[:1] + dst_box[1:]
    
    # 각 이미지의 height, width 값
    src_img_height, src_img_width, _ = src_img.shape
    dst_img_height, dst_img_width, _ = dst_img.shape
    
    # resize 
    src_img = resize(src_img, src_img_height*src_img_width, dst_img_height*dst_img_width, dst_img_width, dst_img_height)
    src_img_height, src_img_width, _ = src_img.shape
    
    # min_x, max_x, min_y, max_y 값으로 변환
    temp_dst_box = dst_box.copy()
    src_box = get_box_coord(src_box, src_img_width, src_img_height)
    dst_box = get_box_coord(dst_box, dst_img_width, dst_img_height)

    # box의 width와 height 값
    src_box_w, src_box_h = src_box[1] - src_box[0], src_box[3] - src_box[2]
    dst_box_w, dst_box_h = dst_box[1] - dst_box[0], dst_box[3] - dst_box[2]
    
    # box의 크기 
    src_box_area = src_box_w * src_box_h
    dst_box_area = dst_box_w * dst_box_h
  
    if src_box_area==0 or dst_box_area==0:
        return dst_img, temp_dst_box

    # image crop
    cropped_img = crop_image(src_img, src_box, center_crop)
    
    if paste_method == 'resize':
        pasted_image = resize_paste(cropped_img, src_box_area, dst_img, dst_box_area, dst_box_w, dst_box_h, dst_box)
    elif paste_method == 'fit':
        pasted_image, modified_box = fit_paste(cropped_img, src_box_area, src_box_w, src_box_h, 
                                    dst_img, dst_box_area, dst_box_w, dst_box_h, dst_box)
        ret_box = box_coord_to_yolo_label(ret_box[0], modified_box, dst_img_width, dst_img_height)
    # elif paste_method == 'padding' and src_box_area < dst_box_area:
    #     padding_paste(cropped_img, src_box_w, src_box_h, dst_img, dst_box_w, dst_box_h, dst_box)
    # else:
    #     modified_box = common_paste(cropped_img, src_box_area, src_box_w, src_box_h, 
    #                                 dst_img, dst_box_area, dst_box_w, dst_box_h, dst_box)
    #     ret_box = box_coord_to_yolo_label(ret_box[0], modified_box, dst_img_width, dst_img_height)
    return pasted_image, ret_box

def IoU(box1, box2):
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

def calculate_iou_in_list(bbox_list):
    """
    input: [[cls, x, y, w, h], [cls, x, y, w, h] ...] 
    
    output: {box_idx:[max iou], ...}
    """
    x1y1x2y2_box_list = []
    for bbox in bbox_list:
        x1, x2, y1, y2 = get_box_coord(bbox, width=1920, height=1080)
        x1y1x2y2_box = [x1, y1, x2, y2]
        x1y1x2y2_box_list.append(x1y1x2y2_box)
    
    iou_dict = {idx:[] for idx in range(len(x1y1x2y2_box_list))}
    box_combination = list(combinations(list(range(len(x1y1x2y2_box_list))), 2))
    
    for combi in box_combination:
        bbox1, bbox2 = x1y1x2y2_box_list[combi[0]], x1y1x2y2_box_list[combi[1]]
        iou = IoU(bbox1, bbox2)
        iou_dict[combi[0]].append(iou)
        iou_dict[combi[1]].append(iou)
    
    for key in iou_dict.keys():
        iou_dict[key] = max(iou_dict[key])
        
    return iou_dict

def vertical_cut_mix(dst_img, dst_label, src_img, src_label, index, cuts, num_pieces):
    src_img_height, src_img_width, _ = src_img.shape
    dst_img_height, dst_img_width, _ = dst_img.shape
    
    cut_start, cut_end = cuts[index], cuts[index+1]
        
    src_x1, src_x2 = int(src_img_width*cut_start), int(src_img_width*cut_end)
    dst_x1, dst_x2 = int(dst_img_width*cut_start), int(dst_img_width*cut_end)
    
    if src_x1==src_x2 or dst_x1==dst_x2:
        return dst_img, dst_label
    
    dst_img[:,dst_x1:dst_x2] = cv2.resize(src_img[:,src_x1:src_x2], (dst_x2-dst_x1,dst_img_height))
    
    add_label = []
    
    for src_bbox in src_label:
        x1,x2,y1,y2 = get_box_coord(src_bbox, dst_img_width, dst_img_height)
        if x2-x1==0: continue
        
        ix1, ix2 = max(x1,dst_x1), min(x2,dst_x2)
        iratio = (ix2-ix1)/(x2-x1)
        if iratio > 0.5:
            add_label.append(box_coord_to_yolo_label(src_bbox[0], [ix1,ix2,y1,y2], dst_img_width, dst_img_height))
    
    add_label = np.array(add_label,dtype=np.float32)
    
    if len(add_label) != 0:
        dst_label = np.concatenate((dst_label,add_label))
            
    return dst_img, dst_label