# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:39:16 2023

@author: Mohammad
"""

import numpy as np
import tensorflow as tf
# from tf.keras import keras
import torch
import sys
import os
import cv2 as cv
import random
from matplotlib import pyplot as plt
from models.experimental import attempt_load
from utils.general import check_img_size
from utils.torch_utils import select_device, TracedModel
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
import glob



weights = 'weights.pt'
device_id = 'cpu'
image_size = 640
trace = True

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)

device = select_device(device_id)
half = device.type != 'cpu'  # half precision only supported on CUDA
# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(image_size, s=stride)  # check img_size

if trace:
    model = TracedModel(model, device, image_size)

if half:
    model.half()  # to FP16
    
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

model.eval()

# selecting the source of raw images
source_image_paths = glob.glob('F:\AI projetcs/job/proj2/Input_images/' + '*.jpg')

for num_img in range(len(source_image_paths)):

    ###################
    source_image_path =   source_image_paths[num_img] 
    source_image = cv.imread(source_image_path)
    print(source_image.shape)
    # Padded resize
    img_size = 640
    stride = 32
    img = letterbox(source_image, img_size, stride=stride)[0]
    #########################
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
        
    with torch.no_grad():
        # Inference
        pred = model(img, augment=True)[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.25, 0.45, classes=0, agnostic=True)
    ##########################

    plate_detections = []
    det_confidences = []

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], source_image.shape).round()

            # Return results
            for *xyxy, conf, cls in reversed(det):
                coords = [int(position) for position in (torch.tensor(xyxy).view(1, 4)).tolist()[0]]
                plate_detections.append(coords)
                det_confidences.append(conf.item())
    ##########################
    print('111111111111111',plate_detections)

    def crop(image, coord):
        cropped_image = image[int(coord[1]):int(coord[3]), int(coord[0]):int(coord[2])]
        return cropped_image

    ###  cropping the detected plates
    cropped_image = crop(source_image, plate_detections[0])

    # saving the detected plates in the directory =>  ../outputs/detected_plates
    cv.imwrite("../outputs/detected_plates/plate_{}.jpg".format(num_img), cropped_image)
