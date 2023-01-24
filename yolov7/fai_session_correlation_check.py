import argparse
from os import listdir
from os.path import isfile, join
from collections import defaultdict
import numpy as np
import torch
import cv2


def process_batch(detections, labels, class_list):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0])).astype(bool)
    gt_classes = labels[:, 0:1]
    detection_classes = detections[:, 0]
    iou = box_iou(labels[:, 1:], detections[:, 1:5])
    correct_class = gt_classes == detection_classes
    same_session_fp, diff_session_fp = 0, 0
    # x = torch.where(iou >= iouv)
    x = torch.where((iou > 0) & correct_class)  # IoU > threshold and classes match
    xx = torch.where(iou > 0)                   # IoU > threshold

    mmatches = torch.stack(xx, 1).cpu().numpy()  # match using only iou [label, detect, iou]
    mm0, mm1 = mmatches.transpose().astype(int)

    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # match using iou and class [label, detect, iou]
        # if x[0].shape[0] > 1:
        #     matches = matches[matches[:, 2].argsort()[::-1]]
        #     matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        #     matches = matches[matches[:, 2].argsort()[::-1]]
        #     matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int)] = True

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i) and any(mm1 == i):  # false postives on ground truth
                    if str(int(dc)) not in class_list:
                        diff_session_fp += 1
                    else:
                        same_session_fp += 1
    else:  # no ground truth matches predictions
        for i, dc in enumerate(detection_classes):
            if any(mm1 == i):  # false postives on ground truth
                if str(int(dc)) not in class_list:
                    diff_session_fp += 1
                else:
                    same_session_fp += 1

    return torch.tensor(correct, dtype=torch.bool), same_session_fp, diff_session_fp


def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x, y, w, h) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # xywh to x1y1x2y2
    bb1 = box1.clone()
    bb2 = box2.clone()

    bb1[:, 2:4] = bb1[:, 2:4] + bb1[:, 0:2]
    bb2[:, 2:4] = bb2[:, 2:4] + bb2[:, 0:2]

    (a1, a2), (b1, b2) = bb1[:, None].chunk(2, 2), bb2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (box_area(bb1.T)[:, None] + box_area(bb2.T) - inter + eps)

    
def main(opt):
    """
    Caculate session correlation metric(Total/TP/FP/sameFP/diffFP)
    sameFP and diffFP is metrics to indicate how many false postive detections
    on the ground truth have same / different session compared to ground truth.
    Note that [FP >= sampFP + diffFP] because they are only counted when they
    are on the ground truth. FP on backgrounds are not counted.
    """

    # Get class list
    class_list = defaultdict(list)
    with open(opt.cls_path, 'r') as f_in:
        session = ''
        lines = f_in.readlines()[1:]
        for line in lines:
            line = line.rstrip().split(',')
            class_num = line[0]
            session = line[2] if line[2] else session

            class_list[session].append(class_num)

    # Get files list
    pred_path = opt.pred_path
    img_path = f'{opt.data_root}/images/'
    gt_path = f'{opt.data_root}/labels/'
    files = [f for f in listdir(pred_path) if isfile(join(pred_path, f))]  # prediction

    # Initialize counters
    cnt_tp, cnt_fp = 0, 0
    same_session_fp, diff_session_fp = 0, 0

    # Caculate metric for every detections
    for file in files:
        session = file.split('_')[0]
        with open(pred_path + file, 'r') as f:
            pred = f.readlines()
            pred = [[float(val) for val in p.split()] for p in pred]
            pred_lines = torch.tensor(pred)

        if not isfile(join(gt_path, file)):
            gt = []
        else:
            with open(gt_path + file, 'r') as f:
                gt = f.readlines()
                gt = [[float(val) for val in g.split()] for g in gt]
                # if len(gt) == 0: continue
                gt_lines = torch.tensor(gt)

        # Caculation is conducted only when ground truth exists
        if len(gt) > 0:
            accuracy, local_same_session_fp, local_diff_session_fp = process_batch(pred_lines, gt_lines, class_list[session])
            local_cnt_tp = sum(accuracy)
            local_cnt_fp = accuracy.shape[0] - sum(accuracy)

            cnt_tp += local_cnt_tp
            cnt_fp += local_cnt_fp
            same_session_fp += local_same_session_fp
            diff_session_fp += local_diff_session_fp
        # Every prediction is processed as false positive if there is no ground truth
        else:
            cnt_fp += pred_lines.shape[0]
            for p in pred:
                class_num = str(int(p[0]))
                if class_num not in class_list[session]:
                    diff_session_fp += 1
                else:
                    same_session_fp += 1
            local_cnt_tp = -1
            local_cnt_fp = -1
            local_same_session_fp = -1
            local_diff_session_fp = -1

        # Visualize false positive results
        if opt.visualizing and (local_diff_session_fp > 0 or local_same_session_fp > 0):
            img = cv2.imread(img_path + file[:-3] + 'jpg')
            H, W, _ = img.shape
            gt_color, pred_color = (255, 0, 0), (0, 255, 0)
            for g in gt:
                x1 = int((g[1] - g[3] / 2) * W)
                x2 = int((g[1] + g[3] / 2) * W)
                y1 = int((g[2] - g[4] / 2) * H)
                y2 = int((g[2] + g[4] / 2) * H)
                cv2.rectangle(img, (x1, y1), (x2, y2), gt_color, 3, 0)
                cv2.putText(img, f'{int(g[0])}', (int(g[1] * W), int(g[2] * H)), 0, 1, gt_color, 2, 0)
            for p in pred:
                x1 = int((p[1] - p[3] / 2) * W)
                x2 = int((p[1] + p[3] / 2) * W)
                y1 = int((p[2] - p[4] / 2) * H)
                y2 = int((p[2] + p[4] / 2) * H)
                cv2.rectangle(img, (x1, y1), (x2, y2), pred_color, 3, 0)
                cv2.putText(img, f'{int(p[0])}', (int(p[1] * W), int(p[2] * H)), 0, 1, pred_color, 2, 0)
                cv2.putText(img, f'{float(p[5]):0.2f}', (int((p[1] + 0.02) * W), int(p[2] * H)), 0, 0.5, (0, 0, 255), 2, 0)
            cv2.putText(img, f'TP: {local_cnt_tp}', (20, 30), 0, 1, (0, 255, 0), 2, 0)
            cv2.putText(img, f'FP: {local_cnt_fp}', (20, 70), 0, 1, (0, 255, 0), 2, 0)
            cv2.putText(img, f'FP_SAME: {local_same_session_fp}', (20, 150), 0, 1, (0, 255, 0), 2, 0)
            cv2.putText(img, f'FP_DIFF: {local_diff_session_fp}', (20, 190), 0, 1, (0, 255, 0), 2, 0)
            cv2.imshow(file, img)
            cv2.waitKey()
            cv2.destroyWindow(file)

    print(f"{' ' * 8}{'Total':>9}{'TP':>9}{'FP':>9}{'sameFP':>9}{'diffFP':>9}")
    print(f"{'=' * 54}")
    print(f'Results {cnt_tp.tolist() + cnt_fp.tolist():>9}{cnt_tp.tolist():>9}{cnt_fp.tolist():>9}{same_session_fp:>9}{diff_session_fp:>9}')


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default='./dataset/', help='data root folder path')
    parser.add_argument('--cls-path', type=str, default='./data/class_124.csv', help='class meta data path')
    parser.add_argument('--pred-path', type=str, default='./runs/test/validation_exp/labels/', help='prediction results path')
    parser.add_argument('--visualizing', type=bool, default=False, help='show false postive results with its confidence')
    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
