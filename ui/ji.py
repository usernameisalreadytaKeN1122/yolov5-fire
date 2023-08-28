# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: yolov5-60
File Name: ji.py
Author: chenming
Create Date: 2021/11/18
Description：
-------------------------------------------------
"""
import argparse
import os
os.chdir("/home/chenming/scm/xianyu/det/yolov5-60/yolov5-60")
import sys
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import json
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox
import numpy as np


@torch.no_grad()
def init():
    weights = "runs/train/exp2/weights/best.pt"  # model.pt path(s)
    device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    half = False  # use FP16 half-precision inference
    dnn = False  # use OpenCV DNN for ONNX inference
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()
    print("模型加载完成!")
    return model


@torch.no_grad()
def process_image(handle=None, input_image=None, args=None, **kwargs):
    '''Do inference to analysis input_image and get output
    Attributes:
    handle: algorithm handle returned by init()
    input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR
    Returns: process result
    '''

    # Process image here
    fake_result = {
        'objects': []
    }

    # Padded resize
    net = handle
    device = ''
    device = select_device(device)
    half = False
    augment = True
    visualize = False
    img_size = (640, 640)
    img0 = input_image
    stride = net.stride
    names = net.names
    img = letterbox(img0, img_size, stride=stride)[0]
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    # pred
    im = torch.from_numpy(img).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    # Inference

    pred = net(im, augment=augment, visualize=visualize)
    conf_thres = 0.25
    iou_thres = 0.45
    classes = None
    max_det = 1000
    agnostic_nms = False
    save_crop = False

    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    for i, det in enumerate(pred):  # per image
        # seen += 1
        # if webcam:  # batch_size >= 1
        #     p, im0, frame = path[i], im0s[i].copy(), dataset.count
        #     s += f'{i}: '
        # else:
        #     p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
        im0 = img0.copy()

        # p = Path(p)  # to Path
        # save_path = str(save_dir / p.name)  # im.jpg
        # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
        # s += '%gx%g ' % im.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy() if save_crop else im0  # for save_crop
        annotator = Annotator(im0, line_width=3, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                # if save_txt:  # Write to file
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                c = int(cls)  # integer class
                label = f'{names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))

                xyxy = [x.cpu().numpy().item() for x in xyxy]
                conf = conf.cpu().numpy()
                name = names[c]
                # print(cls)
                # print(conf)
                # print(xyxy)
                # print(xywh)
                fake_result['objects'].append({
                    'xmin': int(xyxy[0]),
                    'ymin': int(xyxy[1]),
                    'xmax': int(xyxy[2]),
                    'ymax': int(xyxy[3]),
                    'name': str(name),
                    'confidence': float(conf)
                })
                # line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                # with open(txt_path + '.txt', 'a') as f:
                #     f.write(('%g ' * len(line)).rstrip() % line + '\n')
                # if save_img or save_crop or view_img:  # Add bbox to image
        #
        #
        # im0 = annotator.result()
        # # print(im0)
        # # if view_img:
        # cv2.imshow("test", im0)
        # cv2.waitKey(0)  # 1 millisecond
        # cv2.destroyAllWindows()
    return json.dumps(fake_result, indent=4)

# 现在的任务就是走通训练的流程，包括使用cpu进行训练和将日志文件以及模型输出到对应的位置中去
if __name__ == '__main__':
    """Test python api
    """
    img = cv2.imread('data/images/phone/phone_419.jpg')
    predictor = init()
    result = process_image(predictor, img)
    print(result)
