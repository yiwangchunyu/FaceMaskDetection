import argparse
import os
import pickle
import random

import cv2
import numpy as np
import lmdb
import torch
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from utils.display import show_img_cv2

raw_annotations_root='data/data.ncignore/annotations'
raw_images_root='data/data.ncignore/images'
CLASSID={'with_mask':0, 'without_mask':1, 'mask_weared_incorrect':2}
ID2CLASS=['Mask', 'NoMask', 'IncorrectMask']
train_root='data/train'
test_root='data/test'
num_class=3

class MyDataset(Dataset):
    def __init__(self, data_dir, input_shape=(300,300), train=True, mosaic=False, transform=None):
        self.transform = transform
        self.lmdbData=LmdbData(data_dir)
        self.mosaic=mosaic
        self.input_shape=input_shape

    def __len__(self):
        return self.lmdbData.len()

    def _rand(self,a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, img, label, jitter=.3, hue=.1, sat=1.5, val=1.5):
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # image.show()
        iw, ih = image.size
        h, w = self.input_shape

        #resize
        while True:
            new_ar = w / h * self._rand(1 - jitter, 1 + jitter) / self._rand(1 - jitter, 1 + jitter)
            scale = self._rand(.25, 2)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)
            # image.show()

            # place image
            dx = int(self._rand(0, w - nw))
            dy = int(self._rand(0, h - nh))
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image = new_image

            # flip image or not
            flip = self._rand() < .5
            if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

            # distort image
            hue = self._rand(-hue, hue)
            sat = self._rand(1, sat) if self._rand() < .5 else 1 / self._rand(1, sat)
            val = self._rand(1, val) if self._rand() < .5 else 1 / self._rand(1, val)
            x = rgb_to_hsv(np.array(image) / 255.)
            x[..., 0] += hue
            x[..., 0][x[..., 0] > 1] -= 1
            x[..., 0][x[..., 0] < 0] += 1
            x[..., 1] *= sat
            x[..., 2] *= val
            x[x > 1] = 1
            x[x < 0] = 0
            image_data = hsv_to_rgb(x) * 255  # numpy array, 0 to 1

            # correct boxes
            box=label.copy()
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            box_data = np.zeros((len(box), 5))
            box_data[:len(box)] = box
            # Image.fromarray(np.uint8(image_data)).show()
            if (box_data[:,:4]>0).any():
                return image_data, box_data
            else:
                continue

    def generate(self,idx):
        if self.mosaic == True:
            print('not available')
            exit(0)
        else:
            image,label=self.getRaw(idx)
            image_new, lable_new=self.get_random_data(image, label)
        # 在这里看一下图片标注是否正确
        # show_img_cv2(cv2.cvtColor(np.uint8(image_new), cv2.COLOR_RGB2BGR), lable_new)

        boxes = np.array(lable_new[:, :4], dtype=np.float32)
        # box归一化
        boxes[:, 0] = boxes[:, 0] / self.input_shape[1]
        boxes[:, 1] = boxes[:, 1] / self.input_shape[0]
        boxes[:, 2] = boxes[:, 2] / self.input_shape[1]
        boxes[:, 3] = boxes[:, 3] / self.input_shape[0]

        boxes = np.maximum(np.minimum(boxes, 1), 0)
        # 获得长宽
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        #获得中心位置
        boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
        boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2
        y = np.concatenate([boxes, lable_new[:, -1:]], axis=-1)
        img = np.array(image_new, dtype=np.float32)
        return img,y



    def getRaw(self,idx):
        image, label = self.lmdbData.get(idx)
        return image, label

    def __getitem__(self, idx):
        img,label=self.generate(idx)
        img = np.transpose(img / 255.0, (2, 0, 1))
        return torch.Tensor(img), torch.Tensor(label)


def my_collate(batch):
    data = [item[0].numpy() for item in batch]
    data = torch.FloatTensor(data)
    target = [item[1] for item in batch]

    return [data, target]

class LmdbData:
    def __init__(self,dir,size = 1):
        self.env = lmdb.open(dir, map_size=size*1024*1024) #size M
        self.next=0
        pass
    def len(self):
        txn = self.env.begin()
        len = int(txn.get('len'.encode()).decode())
        return len

    def get(self, id):
        txn = self.env.begin()
        x = pickle.loads(txn.get(('x_%d'%(id)).encode()))
        y = pickle.loads(txn.get(('y_%d'%(id)).encode()))
        return x,y
    def list(self):
        len=self.len()
        data=[]
        labels=[]
        for id in range(len):
            x,y=self.get(id)
            data.append(x)
            labels.append(y)
        return np.asarray(data), np.asarray(labels)

    def add(self,x,y):
        txn = self.env.begin(write=True)
        txn.put(('x_%d'%(self.next)).encode(), pickle.dumps(x))
        txn.put(('y_%d' % (self.next)).encode(), pickle.dumps(y))
        txn.put('len'.encode(), str(self.next+1).encode())
        txn.commit()
        self.next+=1

    def put(self,k,v):
        txn = self.env.begin(write=True)
        txn.put(k, v)
        txn.commit()




def get_annotations(path):
    from xml.etree.ElementTree import ElementTree
    tree = ElementTree(file=path)
    root = tree.getroot()
    ObjectSet=root.findall('object')
    data=[]
    for Object in ObjectSet:
        ObjName=Object.find('name').text
        classid=CLASSID[ObjName]

        BndBox=Object.find('bndbox')
        x1 = int(BndBox.find('xmin').text)
        y1 = int(BndBox.find('ymin').text)
        x2 = int(BndBox.find('xmax').text)
        y2 = int(BndBox.find('ymax').text)
        data.append([x1,y1,x2,y2,classid])
    return np.asarray(data)


def raw2lmdb():
    lmdbData_train=LmdbData(dir=train_root,size=500)
    lmdbData_test = LmdbData(dir=test_root, size=500)
    files = os.listdir(raw_images_root)
    files_train, files_test = train_test_split(files, test_size=0.2, random_state=2020)
    print('raw2lmbd train data....')
    for file in files_train:
        name=file.split('.')[0]
        file_path = os.path.join(raw_images_root,file)
        annotation_path = os.path.join(raw_annotations_root,name+'.xml')
        img = cv2.imread(file_path)
        label=get_annotations(annotation_path)
        lmdbData_train.add(img,label)
    print('raw2lmbd train data....')
    for file in files_test:
        name=file.split('.')[0]
        file_path = os.path.join(raw_images_root,file)
        annotation_path = os.path.join(raw_annotations_root,name+'.xml')
        img = cv2.imread(file_path)
        label=get_annotations(annotation_path)
        lmdbData_test.add(img,label)

def show(img, label):
    label_color = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
    print(img.shape)
    img_draw = img.copy()
    for id in range(label.shape[0]):
        print(label[id,4], label[id, 2]-label[id, 0],label[id, 3]-label[id, 1])
        img_draw = cv2.rectangle(img, (label[id, 0], label[id, 1]), (label[id, 2], label[id, 3]),
                                 label_color[label[id, 4]], 1)
        img_draw=cv2.putText(img_draw, ID2CLASS[label[id, 4]], (label[id, 0], label[id, 1]-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, label_color[label[id, 4]], 1)
    cv2.imshow('img', img_draw)
    cv2.waitKey(0)

def statics():
    train_data = MyDataset(train_root)
    test_data = MyDataset(test_root)
    boxes_cnt=[]
    image_shapes=[]
    boxes_shapes=[]
    class_cnt=[0,0,0]
    for i in range(train_data.__len__()):
        img, label = train_data.__getitem__(i)
        image_shapes.append(img.shape)
        boxes_cnt.append(label.shape[0])

        for j in range(label.shape[0]):
            class_cnt[label[j,0]]+=1
            boxes_shapes.append([label[j, 4] - label[j, 2], label[j, 3] - label[j, 1]])
    for i in range(test_data.__len__()):
        img, label = test_data.__getitem__(i)
        image_shapes.append(img.shape)
        boxes_cnt.append(label.shape[0])
        for j in range(label.shape[0]):
            class_cnt[label[j,0]]+=1
            boxes_shapes.append([label[j, 4] - label[j, 2], label[j, 3] - label[j, 1]])
    boxes_cnt_avg=np.average(boxes_cnt)
    image_shape_avg=np.average(image_shapes,axis=0)
    box_shape_avg=np.average(boxes_shapes,axis=0)
    print('train data length:', train_data.__len__(), '\ntest  data length:', test_data.__len__())
    print('image shape in average', image_shape_avg)
    print('box shape in average', box_shape_avg)
    print('average objects in a images:',boxes_cnt_avg)
    print('class count:', class_cnt)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw2lmdb", help="transform raw data to lmdb data", action="store_true")
    parser.add_argument("--stat", help="", action="store_true")
    parser.add_argument("--show_origin", help="", action="store_true")
    args = parser.parse_args()

    if args.raw2lmdb:
        raw2lmdb()
    if args.stat:
        statics()

    train_data = MyDataset(train_root)
    test_data = MyDataset(test_root)
    if args.show_origin:
        for i in random.sample(range(test_data.__len__()), 10):
            img, label = test_data.getRaw(i)
            show(img, label)
    train_data.__getitem__(0)


