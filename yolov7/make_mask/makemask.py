import collections
from collections import defaultdict
import glob
import gc
import io
import json
import logging
import os
import random
import warnings

import imageio
from IPython.display import display, Javascript
from IPython.display import Image as IPyImage
import matplotlib
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import scipy.misc
from six import BytesIO
from skimage import color
from skimage import measure
from skimage import transform
from skimage import util
from skimage.color import rgb_colors
import tensorflow as tf
import tifffile as tiff 

import argparse
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_devices = tf.config.experimental.list_physical_devices('GPU')

# Define utilities

COLORS = ([rgb_colors.cyan, rgb_colors.orange, rgb_colors.pink,
           rgb_colors.purple, rgb_colors.limegreen , rgb_colors.crimson] +
          [(color) for (name, color) in color.color_dict.items()])
random.shuffle(COLORS)

logging.disable(logging.WARNING)


def read_image(path):
  """Read an image and optionally resize it for better plotting."""
  with open(path, 'rb') as f:
    img = Image.open(f)
    return np.array(img, dtype=np.uint8)

def read_json(path):
  with open(path) as f:
    return json.load(f)

def create_detection_map(annotations):
    """Creates a dict mapping IDs to detections."""
    
    ann_map = {}
    for image in tqdm(annotations['images']):
        tmp = image['id']
        for bbox in annotations['annotations']:
            if bbox['image_id'] == tmp:
                if image['file_name'] not in ann_map:
                    ann_map[image['file_name']] = [bbox]
                else:
                    tmp_1 = ann_map[image['file_name']]
                    tmp_1.append(bbox)
                    ann_map[image['file_name']] = tmp_1
                
    return  ann_map

def get_mask_prediction_function(model):
  """Get single image mask prediction function using a model."""

  @tf.function
  def predict_masks(image, boxes):
    height, width, _ = image.shape.as_list()
    batch = image[tf.newaxis]
    boxes = boxes[tf.newaxis]

    detections = model(batch, boxes)
    masks = detections['detection_masks']

    return reframe_box_masks_to_image_masks(masks[0], boxes[0],
                                             height, width)

  return predict_masks

def convert_boxes(boxes):
  xmin, ymin, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
  ymax = ymin + height
  xmax = xmin + width

  return np.stack([ymin/1080, xmin/1920, ymax/1080, xmax/1920], axis=1).astype(np.float32)



# Copied from tensorflow/models
def reframe_box_masks_to_image_masks(box_masks, boxes, image_height,
                                     image_width, resize_method='bilinear'):
  """Transforms the box masks back to full image masks.
  Embeds masks in bounding boxes of larger masks whose shapes correspond to
  image shape.
  Args:
    box_masks: A tensor of size [num_masks, mask_height, mask_width].
    boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
           corners. Row i contains [ymin, xmin, ymax, xmax] of the box
           corresponding to mask i. Note that the box corners are in
           normalized coordinates.
    image_height: Image height. The output mask will have the same height as
                  the image height.
    image_width: Image width. The output mask will have the same width as the
                 image width.
    resize_method: The resize method, either 'bilinear' or 'nearest'. Note that
      'bilinear' is only respected if box_masks is a float.
  Returns:
    A tensor of size [num_masks, image_height, image_width] with the same dtype
    as `box_masks`.
  """
  resize_method = 'nearest' if box_masks.dtype == tf.uint8 else resize_method
  # TODO(rathodv): Make this a public function.
  def reframe_box_masks_to_image_masks_default():
    """The default function when there are more than 0 box masks."""
    def transform_boxes_relative_to_boxes(boxes, reference_boxes):
      boxes = tf.reshape(boxes, [-1, 2, 2])
      min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
      max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
      denom = max_corner - min_corner
      # Prevent a divide by zero.
      denom = tf.math.maximum(denom, 1e-4)
      transformed_boxes = (boxes - min_corner) / denom
      return tf.reshape(transformed_boxes, [-1, 4])

    box_masks_expanded = tf.expand_dims(box_masks, axis=3)
    num_boxes = tf.shape(box_masks_expanded)[0]
    unit_boxes = tf.concat(
        [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
    reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)

    # TODO(vighneshb) Use matmul_crop_and_resize so that the output shape
    # is static. This will help us run and test on TPUs.
    resized_crops = tf.image.crop_and_resize(
        image=box_masks_expanded,
        boxes=reverse_boxes,
        box_indices=tf.range(num_boxes),
        crop_size=[image_height, image_width],
        method=resize_method,
        extrapolation_value=0)
    return tf.cast(resized_crops, box_masks.dtype)

  image_masks = tf.cond(
      tf.shape(box_masks)[0] > 0,
      reframe_box_masks_to_image_masks_default,
      lambda: tf.zeros([0, image_height, image_width, 1], box_masks.dtype))
  return tf.squeeze(image_masks, axis=3)

def makeseg(image_path, detection_map, source, prediction_function):
    image_id = os.path.basename(image_path)

    if image_id not in detection_map:
        print(f'Image {image_path} is missing detection data.')
    elif len(detection_map[image_id]) == 0:
        print(f'There are no detected objects in the image {image_path}.')
    else:
        detections = detection_map[image_id]
        image = read_image(image_path)
        ids = np.array([det['id'] for det in detections])
        bboxes = np.array([det['bbox'] for det in detections])
        bboxes = convert_boxes(bboxes)
        masks = prediction_function(tf.convert_to_tensor(image),
                                    tf.convert_to_tensor(bboxes, dtype=tf.float32))
        
        contours_results = [measure.find_contours(mask.numpy(), 0.5) for mask in masks]
        count = 0
        for contours in contours_results:
            segmentations = []
            for contour in contours:
                contour = np.flip(contour, axis=1)
                segmentations.append(contour.ravel().tolist())
            if len(segmentations) != 0:
                source['annotations'][ids[count]-1]['segmentation'] = [0]
                source['annotations'][ids[count]-1]['segmentation'][0] = segmentations[0]
            count = count + 1
        
    return source

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonfile', type=str, default='./train_v4_pkt_sample.json', help='data root folder path')
    parser.add_argument('--imgpath', type=str, default='../custom_dataset/train/images/', help='class meta data path')
    return parser.parse_known_args()[0] if known else parser.parse_args()

def main(opt):

    print('========= loading model ============')
    model = tf.keras.models.load_model('deepmac_1024x1024_coco17/saved_model')
    print('========= processing model =========')
    prediction_function = get_mask_prediction_function(model)
    BOX_ANNOTATION_FILE = opt.jsonfile
    print('======= make detection_map =========')
    detection_map = create_detection_map(read_json(BOX_ANNOTATION_FILE))
    print('=========== make mask ==============')
    with open(BOX_ANNOTATION_FILE,'r') as f:
        source = json.load(f)

    source_len = len(source['images'])
    for i in tqdm(range(source_len), desc=f"iterate range {source_len}"):
        path = source['images'][i]['file_name']
        path_name = f"{opt.imgpath}{path}"
        source = makeseg(path_name, detection_map, source, prediction_function)

    with open(BOX_ANNOTATION_FILE,'w', encoding='utf-8') as make_file:
        json.dump(source, make_file, indent="\t")
    

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
