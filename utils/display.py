import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_img_cv2(img, label):
    from dataset import ID2CLASS
    label_color = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
    print(img.shape)
    img_draw = img.copy()
    label_draw = np.uint(label)
    for id in range(label_draw.shape[0]):
        print(label_draw[id,4], label_draw[id, 2]-label_draw[id, 0],label_draw[id, 3]-label_draw[id, 1])
        img_draw = cv2.rectangle(img_draw, (label_draw[id, 0], label_draw[id, 1]), (label_draw[id, 2], label_draw[id, 3]),
                                 label_color[label_draw[id, 4]], 1)
        img_draw=cv2.putText(img_draw, ID2CLASS[label_draw[id, 4]], (label_draw[id, 0], label_draw[id, 1]-2),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.3, label_color[label_draw[id, 4]], 1)
    cv2.imshow('img', img_draw)
    cv2.waitKey(0)


def plot_loss_curve(train_losses, test_losses, t):
    plt.plot(np.arange(len(train_losses)),train_losses,label='train_loss')
    plt.plot(np.arange(len(test_losses))*t, test_losses, label='test_loss')
    plt.xlabel('Iter')
    plt.savefig('loss.png',dpi=300)
    plt.savefig('loss.eps', dpi=300)