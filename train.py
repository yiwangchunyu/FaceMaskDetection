import argparse

import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from dataset import MyDataset, train_root, test_root, my_collate
from nets.loss import YOLOLoss
from nets.yolo4 import YoloBody
from utils.display import plot_loss_curve

parser = argparse.ArgumentParser()
parser.add_argument('--input_shape', type=int, default=416, help='input image shape width=height')
parser.add_argument('--batch_size', type=int, default=8, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')
parser.add_argument('--nepoch', type=int, default=30, help='')
parser.add_argument('--train_data_root', type=str, default='data/train', help='')
parser.add_argument('--test_data_root', type=str, default='data/test', help='')
parser.add_argument('--gpu', type=bool, default=True)

args = parser.parse_args()
num_class=3
anchors=[
    [
        [459,401],
        [192,243],
        [142,110]
    ],
    [
        [72,146],
        [76,55],
        [36,75]
    ],
[
        [40,28],
        [19,36],
        [12,16]
    ],
]


def train():
    num_classes=3

    # 建立loss函数
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(np.reshape(anchors, [-1, 2]), num_classes, (args.input_shape, args.input_shape), 0.3, args.gpu))

    train_data = MyDataset(train_root,input_shape=(args.input_shape,args.input_shape))
    test_data = MyDataset(test_root,input_shape=(args.input_shape,args.input_shape))

    train_loader=DataLoader(train_data,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers, collate_fn=my_collate)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=my_collate)

    # 使用GPU
    if args.gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")


    # 创建模型
    model = YoloBody(len(anchors[0]), num_classes)
    model_path = "weights/yolov4_coco_pretrained_weights.pth"
    # model_path = "model_data/yolov4_maskdetect_weights0.pth"
    # 加快模型训练的效率
    print('Loading pretrained model weights.')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    net=model.to(device)

    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)

    train_losses=[]
    test_losses=[]
    min_loss=1e10
    for epoch in range(args.nepoch):
        train_loss=0
        for i, data in enumerate(train_loader):
            net.train()
            inputs, labels = data[0].to(device), data[1]
            optimizer.zero_grad()
            outputs = net(inputs)
            losses = []
            for j in range(3):
                loss_item = yolo_losses[j](outputs[j], labels)
                losses.append(loss_item[0])
            loss = sum(losses)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            train_loss+=loss.item()
            train_losses.append(loss.item())
            print("epoch:%d/%d, batch:%d/%d, train_loss:%f"%(epoch, args.nepoch, i, len(train_loader), loss.item()))
        train_loss/=len(train_loader)

        # test
        test_loss=0
        net.eval()
        for i_test, data_test in enumerate(test_loader):
            with torch.no_grad():
                inputs, labels = data_test[0].to(device), data_test[1]
                optimizer.zero_grad()
                outputs = net(inputs)
                losses = []
                for j in range(3):
                    loss_item = yolo_losses[j](outputs[j], labels)
                    losses.append(loss_item[0])
                loss = sum(losses)
                test_loss+=loss.item()
                print("epoch:%d/%d, batch:%d/%d, test_loss:%f" % (epoch, args.nepoch, i_test, len(test_loader), loss.item()))

        test_loss/=len(test_loader)
        test_losses.append(test_loss)

        if test_loss<min_loss:
            torch.save(net.state_dict(), 'weights/face_mask_weights.pth')
        print("epoch:%d/%d, train_loss:%f， test_loss:%f" % (epoch, args.nepoch, train_loss, test_loss))
    plot_loss_curve(train_losses,test_losses,len(train_loader))

if __name__=="__main__":
    train()