import random

import cv2
import numpy as np
import torch
from PIL import ImageFont, ImageDraw, Image

# -------------------------------------#
#       对单张图片进行预测
# -------------------------------------#
from dataset import MyDataset, test_root
from nets.yolo4_inference import YOLO4_inference

def detect_test_data_show(img, gt, boxes, labels, scores):
    from dataset import ID2CLASS
    label_color = np.asarray([(0, 255, 0), (255, 0, 0), (0, 0, 255)])
    gt_color = np.asarray([(0, 255, 255), (255, 0, 255), (255, 255, 0)])

    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font='simhei.ttf', size=10)
    if gt is not None:
        for id in range(gt.shape[0]):
            draw.rectangle(xy=(gt[id,0], gt[id,1], gt[id,2], gt[id,3]), fill=None, outline=tuple(gt_color[int(gt[id,4])]), width=2)

            txt = '{}:gt'.format(ID2CLASS[int(gt[id,4])])
            txt_size = draw.textsize(txt, font)
            draw.rectangle(xy=(gt[id, 0], gt[id, 1] - txt_size[1], gt[id, 0] + txt_size[0]+2, gt[id, 1]),
                           fill=tuple(gt_color[int(gt[id, 4])]), )
            draw.text((gt[id,0]+2, gt[id,1]-txt_size[1]), txt, fill=(0, 0, 0), font=font)
    if labels is not None:
        for id in range(len(labels)):
            top, left, bottom, right = boxes[id]
            draw.rectangle(xy=(left, top, right, bottom), fill=None, outline=tuple(label_color[labels[id]]), width=1)
            txt = '{}:{:.2f}'.format(ID2CLASS[labels[id]], scores[id])
            txt_size = draw.textsize(txt, font)
            draw.rectangle(xy=(left, top - txt_size[1], left + txt_size[0] + 2, top),
                           fill=tuple(label_color[labels[id]]), )
            draw.text((left + 2, top - txt_size[1]), txt, fill=(0, 0, 0), font=font)


    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    cv2.imshow("FaceMaskDetector", img)
    cv2.waitKey()

def detect_test_data(model_path='weights/face_mask_weights.pth', num=20):
    yolo_detector = YOLO4_inference(model_path=model_path)
    test_data = MyDataset(test_root)
    for i in random.sample(range(test_data.__len__()), num):
        img, gt = test_data.getRaw(i)
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        boxes, labels, scores = yolo_detector.predict(image)
        detect_test_data_show(image, gt, boxes, labels, scores)


def detect_image(img_path, model_path='weights/face_mask_weights.pth'):
    yolo_detector = YOLO4_inference(model_path=model_path)
    image = Image.open(img_path)
    boxes, labels, scores = yolo_detector.predict(image)
    detect_test_data_show(image, None, boxes, labels, scores)

def evaluate():
    pass


if __name__=="__main__":
    # detect_test_data()
    detect_image('test_imgs/maksssksksss0.png')