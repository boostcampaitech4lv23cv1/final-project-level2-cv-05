import random
import numpy as np
import cv2
from itertools import combinations


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


def crop_image(img, box):
    x1, x2, y1, y2 = box
    cropped_img = img[y1:y2, x1:x2]
    return cropped_img


def resize(src, src_area, dst_area, dst_w, dst_h):
    if src_area == 0 or dst_area ==0:  # bbox 매우 작아서 area 0이면 pass
        return None
    if src_area > dst_area:
        resized = cv2.resize(src, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(src, (dst_w, dst_h), interpolation=cv2.INTER_CUBIC)        
    return resized


def resize_paste(src, src_area, dst, dst_area, dst_w, dst_h, dst_box, inplace=False):
    resized = resize(src, src_area, dst_area, dst_w, dst_h)
    if resized is None:
        return None
    pasted_image = dst if inplace else dst.copy()
    pasted_image[dst_box[2]:dst_box[3], dst_box[0]:dst_box[1]] = resized
    return pasted_image


def zero_padding(src, src_w, src_h, dst_w, dst_h):
    shift_x = (dst_w - src_w) / 2
    shift_y = (dst_h - src_h) / 2
    m = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    dsize = (dst_w, dst_h)
    padded = cv2.warpAffine(src, m, dsize)
    return padded


def padding_paste(src, src_w, src_h, dst, dst_w, dst_h, dst_box, inplace=False):
    padded = zero_padding(src, src_w, src_h, dst_w, dst_h)
    pasted_image = dst if inplace else dst.copy()
    pasted_image[dst_box[2]:dst_box[3], dst_box[0]:dst_box[1]] = padded
    return pasted_image


def common_paste(src, src_area, src_w, src_h, dst, dst_area, dst_w, dst_h, dst_box, inplace=False):
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
    pasted_image = dst if inplace else dst.copy()
    pasted_image[y1:y2, x1:x2] = src
    return pasted_image, [x1, x2, y1, y2]


def crop_paste(src_img, src_box, dst_img, dst_box, paste_method='resize', inplace=False):
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
    """
    assert paste_method == 'resize'
    if paste_method not in ['resize', 'padding', 'paste']:
        raise ValueError("paste_method : 'resize'")

    # return할 label 값. crop할 박스의 class + paste될 box
    ret_box = src_box[:1] + dst_box[1:]

    # 각 이미지의 height, width 값
    src_img_height, src_img_width, _ = src_img.shape
    dst_img_height, dst_img_width, _ = dst_img.shape

    # resize
    src_img = resize(src_img, src_img_height*src_img_width, dst_img_height*dst_img_width, dst_img_width, dst_img_height)
    src_img_height, src_img_width, _ = src_img.shape

    # min_x, max_x, min_y, max_y 값으로 변환
    src_box = get_box_coord(src_box, src_img_width, src_img_height)
    dst_box = get_box_coord(dst_box, dst_img_width, dst_img_height)

    # box의 width와 height 값
    src_box_w, src_box_h = src_box[1] - src_box[0], src_box[3] - src_box[2]
    dst_box_w, dst_box_h = dst_box[1] - dst_box[0], dst_box[3] - dst_box[2]

    # box의 크기 
    src_box_area = src_box_w * src_box_h
    dst_box_area = dst_box_w * dst_box_h

    if src_box_area == 0 or dst_box_area == 0:  # bbox 매우 작아서 area 0인 경우
        return None, ret_box

    # image crop
    cropped_img = crop_image(src_img, src_box)

    if paste_method == 'resize':
        pasted_image = resize_paste(cropped_img, src_box_area, dst_img, dst_box_area, 
                                    dst_box_w, dst_box_h, dst_box, inplace=inplace)
    elif paste_method == 'padding' and src_box_area < dst_box_area:
        pasted_image = padding_paste(cropped_img, src_box_w, src_box_h, dst_img, 
                                     dst_box_w, dst_box_h, dst_box, inplace=inplace)
    else:
        pasted_image, modified_box = common_paste(cropped_img, src_box_area, src_box_w, src_box_h, 
                                                  dst_img, dst_box_area, dst_box_w, dst_box_h, 
                                                  dst_box, inplace=False)
        ret_box = box_coord_to_yolo_label(ret_box[0], modified_box, dst_img_width, dst_img_height)
    return pasted_image, ret_box


def crop_mix(src_img, src_box, dst_img, dst_box, alpha=0.2, method='mixup'):
    """
    src_img의 box를 crop해서 dst_img의 box와 mix
    src_img : crop할 image
    src_box : crop할 src_img의 bbox
    dst_img : paste할 image
    dst_box : paste할 dst_img의 bbox
    alpha : 베타 분포 parameter
    method : cutmix 또는 mixup
    """
    assert method in ['mixup', 'cutmix']

    lam = np.random.beta(alpha, alpha)
    ret_box = dst_box
    mixed_box = src_box[:1] + dst_box[1:]

    # 각 이미지의 height, width 값
    src_img_height, src_img_width, _ = src_img.shape
    dst_img_height, dst_img_width, _ = dst_img.shape

    # resize
    src_img = resize(src_img, src_img_height*src_img_width, dst_img_height*dst_img_width, dst_img_width, dst_img_height)
    src_img_height, src_img_width, _ = src_img.shape

    # min_x, max_x, min_y, max_y 값으로 변환
    src_box = get_box_coord(src_box, src_img_width, src_img_height)
    dst_box = get_box_coord(dst_box, dst_img_width, dst_img_height)

    # box의 width와 height 값
    src_box_w, src_box_h = src_box[1] - src_box[0], src_box[3] - src_box[2]
    dst_box_w, dst_box_h = dst_box[1] - dst_box[0], dst_box[3] - dst_box[2]

    # box의 크기
    src_box_area = src_box_w * src_box_h
    dst_box_area = dst_box_w * dst_box_h

    if src_box_area == 0 or dst_box_area == 0:
        return None, None, None, None

    # image crop
    cropped_img1 = crop_image(src_img, src_box)
    cropped_img2 = crop_image(dst_img, dst_box)
    cropped_img1 = resize(cropped_img1, src_box_area, dst_box_area, dst_box_w, dst_box_h)

    # box mix
    if method == 'mixup':
        cropped_img2 = cropped_img2 * lam + cropped_img1 * (1 - lam)
    elif method == 'cutmix':
        cx = np.random.uniform(0, dst_box_w)
        cy = np.random.uniform(0, dst_box_h)
        w = dst_box_w * np.sqrt(1 - lam)
        h = dst_box_h * np.sqrt(1 - lam)
        x0 = int(np.round(max(cx - w / 2, 0)))
        x1 = int(np.round(min(cx + w / 2, dst_box_w)))
        y0 = int(np.round(max(cy - h / 2, 0)))
        y1 = int(np.round(min(cy + h / 2, dst_box_h)))

        cropped_img2[y0:y1, x0:x1, :] = cropped_img1[y0:y1, x0:x1, :]

    mixed_image = dst_img.copy()
    mixed_image[dst_box[2]:dst_box[3], dst_box[0]:dst_box[1]] = cropped_img2

    return mixed_image, ret_box, mixed_box, lam


def box_out(dst_img, dst_box):

    # 각 이미지의 height, width 값
    dst_img_height, dst_img_width, _ = dst_img.shape

    # min_x, max_x, min_y, max_y 값으로 변환
    dst_box = get_box_coord(dst_box, dst_img_width, dst_img_height)

    # dst_box가 1보다 작으면
    if (dst_box[1] - dst_box[0]) * (dst_box[3] - dst_box[2]) == 0:
        return None

    # box out
    ret_img = dst_img.copy()
    x1, x2, y1, y2 = dst_box

    # ret_img[y1:y2, x1:x2] = 0
    ret_img[y1:y2, x1:x2] = [random.randint(64, 191) for _ in range(3)]
    return ret_img


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


def calculate_iou_in_list(bbox_list, width=1920, height=1080):
    """
    input: [[cls, x, y, w, h], [cls, x, y, w, h] ...] 

    output: {box_idx:[max iou], ...}
    """
    x1y1x2y2_box_list = []
    for bbox in bbox_list:
        x1, x2, y1, y2 = get_box_coord(bbox, width=width, height=height)
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
