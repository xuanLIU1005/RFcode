import numpy
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from torch.utils.data.dataloader import default_collate
import sys
from collections import defaultdict
import json

torch.set_default_dtype(torch.float32)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import HIBERTools as hiber
import random

import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches


HIBER_CLASSES = (
    "bg", "human"
)


class HIBERDatasetSingleFrame(Dataset):

    def __len__(self):
        return len(self.ds)

    def __init__(self, data_dir, split):
        self.data_dir = data_dir
        self.categories = ["MULTI"]
        self.split = split
        self.ds = hiber.HIBERDataset(root=self.data_dir, categories=self.categories, mode=self.split)
        self.classes = {i: n for i, n in enumerate(HIBER_CLASSES, 1)}

    def __getitem__(self, idx):
        data = self.ds[idx]
        hor,ver = self.get_image(data)
        # visualize_channels(hor)
        # visualize_channels(ver)
        # print(hor.shape,ver.shape)
        hor_boxes = self.get_boxes(data)
        # visualize_combined(hor_boxes,hor)
        hor_boxes_wh = np.zeros_like(hor_boxes)
        hor_boxes_wh[:, 0] = (hor_boxes[:,0]+hor_boxes[:, 2])/2/4
        hor_boxes_wh[:, 1] = (hor_boxes[:,1] + hor_boxes[:, 3]) / 2/4
        hor_boxes_wh[:, 2] = (hor_boxes[:,2] - hor_boxes[:, 0]) /2
        hor_boxes_wh[:, 3] = (hor_boxes[:, 3] - hor_boxes[:, 1]) /2
        heatmap = generate_heatmap(256, 256, hor_boxes_wh[:,:2],sigma=2 )
        # plt.imshow(heatmap)
        # plt.show()
        return hor.astype(np.float32), ver.astype(np.float32), hor_boxes_wh.astype(np.float32), heatmap.astype(np.float32)


    def get_image(self, data):
        hor = data[0]
        ver = data[1]
        hor = cv2.resize(hor.transpose(1,2,0), (256, 256)).transpose(2, 0, 1)
        ver = cv2.resize(ver, (256, 256)).transpose(2, 0, 1)
        return hor, ver

    def get_boxes(self, data): # return xyxy
        hor_boxes = data[4]
        original_size = np.array([160, 200])  # height, width
        target_size = np.array([256, 256])  # height, width

        # 计算尺寸缩放因子
        scale_factors = target_size / original_size

        scaled_bounding_boxes = hor_boxes.copy()
        scaled_bounding_boxes[:, [0, 2]] *= scale_factors[1]  # 缩放 x 坐标
        scaled_bounding_boxes[:, [1, 3]] *= scale_factors[0]  # 缩放 y 坐标
        return scaled_bounding_boxes

class HIBERDatasetMultiFrame(Dataset):

    def __len__(self):
        return len(self.ds)

    def __init__(self, data_dir, split):
        self.data_dir = data_dir
        self.categories = ["MULTI"]
        self.split = split
        self.ds = json.load(open("dataset10_test.json"))
        self.classes = {i: n for i, n in enumerate(HIBER_CLASSES, 1)}

    def __getitem__(self, idx):
        data = self.ds[idx]
        horizontal_heatmap_path = data["horizontal_heatmap_path"]
        vertical_heatmap_path = data["vertical_heatmap_path"]
        hor = np.zeros((10,2,256,256))
        ver = np.zeros((10,2,256,256))
        for i in range(len(horizontal_heatmap_path)):
            hor_path = horizontal_heatmap_path[i]
            ver_path = vertical_heatmap_path[i]
            hor_i,ver_i = self.get_image(os.path.join(self.data_dir, hor_path), os.path.join(self.data_dir, ver_path))
            hor[i] = hor_i
            ver[i] = ver_i

        original_hor_boxes = np.array(data['boxes'])
        for i, box in enumerate(original_hor_boxes):
            original_hor_boxes[i] = self.get_boxes(original_hor_boxes[i]) # 左上右下
        hor_boxes_wh = np.zeros_like(original_hor_boxes)
        hor_boxes_wh[:,:, 0] = (original_hor_boxes[:,:,0]+original_hor_boxes[:,:, 2])/2 #中心点x
        hor_boxes_wh[:,:, 1] = (original_hor_boxes[:,:,1] + original_hor_boxes[:,:, 3]) / 2 #中心点y
        hor_boxes_wh[:,:, 2] = (original_hor_boxes[:,:,2] - original_hor_boxes[:,:, 0]) /2  #宽的一半（x）
        hor_boxes_wh[:,:, 3] = (original_hor_boxes[:,:, 3] - original_hor_boxes[:,:, 1]) /2 #高的一半（y）
        heatmap = np.zeros((10,64,64))
        for i,box in enumerate(hor_boxes_wh):
            hp = generate_heatmap(256, 256, box[:,:2], sigma = 1.5)
            # visualize_combined(original_hor_boxes[i], hor[i])
            # plt.imshow(hp)
            # plt.title(f"{hp.shape}")
            # plt.show()
            heatmap[i] = hp
        # visualize_channels(hor[1])
        # visualize_channels(ver[1])
        # print(original_hor_boxes[1])
        # visualize_combined(original_hor_boxes[1],hor[1])
        # exit()
        # plt.imshow(heatmap[1])
        # plt.show()
        hor_boxes_size = np.zeros_like(hor_boxes_wh) # (10, 2, 4)
        hor_boxes_size[:,:,0] = hor_boxes_wh[:,:, 0]/4 - original_hor_boxes[:,:,0]/4
        hor_boxes_size[:,:,1] = hor_boxes_wh[:,:, 1]/4 - original_hor_boxes[:,:,1]/4
        hor_boxes_size[:,:,2] = original_hor_boxes[:,:, 2]/4 - hor_boxes_wh[:,:,0]/4
        hor_boxes_size[:,:,3] = original_hor_boxes[:,:, 3]/4 - hor_boxes_wh[:,:,1]/4

        hor_offset = np.zeros((hor_boxes_size.shape[0],hor_boxes_size.shape[1],2)) #(10, 2, 2)
        hor_offset[:,:,0] = hor_boxes_wh[:,:, 0]/4 - hor_boxes_wh[:,:, 0]//4
        hor_offset[:,:,1] = hor_boxes_wh[:,:, 1]/4 - hor_boxes_wh[:,:, 1]//4

        original_pose3D = np.array(data['pose3D']) #(10, 2, 14, 3)

        return (hor.astype(np.float32), ver.astype(np.float32), hor_boxes_size.astype(np.float32),
                heatmap.astype(np.float32), hor_offset.astype(np.float32), hor_boxes_wh[:,:,:2].astype(np.float32),
                original_pose3D.astype(np.float32))


    def get_image(self, hor_path, ver_path):
        hor = np.load(hor_path)
        ver = np.load(ver_path)
        hor = cv2.resize(hor.transpose(1,2,0), (256, 256)).transpose(2, 0, 1)
        ver = cv2.resize(ver, (256, 256)).transpose(2, 0, 1)
        return hor, ver

    def get_boxes(self, hor_boxes): # return xyxy
        original_size = np.array([160, 200])  # height, width
        target_size = np.array([256, 256])  # height, width

        # 计算尺寸缩放因子
        scale_factors = target_size / original_size

        scaled_bounding_boxes = hor_boxes.copy()
        scaled_bounding_boxes[:, [0, 2]] *= scale_factors[1]  # 缩放 x 坐标
        scaled_bounding_boxes[:, [1, 3]] *= scale_factors[0]  # 缩放 y 坐标
        return scaled_bounding_boxes

def compute_heatmap(heatmap, gt_boxes):
    # 这里假设gt_boxes是[n, 4]的tensor，其中n是batch size，每个box是[x1, y1, x2, y2]
    for i in range(len(gt_boxes)):
        box = gt_boxes[i]
        width = box[2]
        height = box[3]
        center_x = box[0]
        center_y = box[1]
        # 将中心点坐标转换为热图的尺度
        # generate_heatmap(height, width, )
    return heatmap


def generate_heatmap(height, width, points, sigma=2):
    """ 生成高斯热图。

    Args:
    - height (int): 热图的高度。
    - width (int): 热图的宽度。
    - points (list of tuples): 每个点的 (x, y) 坐标列表。
    - sigma (float): 高斯核的标准差。

    Returns:
    - numpy.ndarray: 生成的热图。
    """
    heatmap = np.zeros((height//4, width//4), dtype=np.float32)
    for x, y in points:
        # 将原始坐标缩小四倍
        x_scaled = int(x / 4)
        y_scaled = int(y / 4)
        for i in range(height//4):
            for j in range(width//4):
                heatmap[i, j] += np.exp(-((j - x_scaled) ** 2 + (i - y_scaled) ** 2) / (2 * sigma ** 2))
    return heatmap




def visualize_channels(data):
    num_channels = data.shape[0]

    # 创建足够的子图来展示所有通道
    fig, axes = plt.subplots(1, num_channels, figsize=(6 * num_channels, 6))
    if num_channels == 1:
        axes = [axes]  # 确保axes是列表，即使只有一个子图

    # 为每个通道的数据创建一个图像
    for i, ax in enumerate(axes):
        img = ax.imshow(data[i], cmap='gray')
        ax.set_title(f'Channel {i + 1}')
        plt.colorbar(img, ax=ax)

    plt.show()


def visualize_combined(bbox_data, heatmap_data, frame_index=0):
    # 加载 bounding box 和 heatmap 数据
    # bbox_data = np.load(file_path_bbox)
    # heatmap_data = np.load(file_path_heatmap)

    num_channels = heatmap_data.shape[0]

    # 创建足够的子图来展示所有通道
    fig, axes = plt.subplots(1, num_channels, figsize=(6 * num_channels, 6))
    if num_channels == 1:
        axes = [axes]  # 确保axes是列表，即使只有一个子图

    # 获取指定帧的bounding box数据
    frame_data = bbox_data #[frame_index]

    # 为每个通道的数据创建一个图像并叠加bounding box
    for i, ax in enumerate(axes):
        img = ax.imshow(heatmap_data[i], cmap='gray')
        ax.set_title(f'Channel {i + 1}')
        plt.colorbar(img, ax=ax)  # 为每个图像添加色条

        # 绘制每个人的边界框
        for person_index, bbox in enumerate(frame_data):
            # 解析边界框坐标
            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            height = y_max - y_min

            # 创建一个矩形表示边界框
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min, f'Person {person_index + 1}', color='white', fontsize=12, ha='left', va='top',
                    bbox=dict(boxstyle="square,pad=0.3", facecolor='black', alpha=0.5))

    plt.show()

if __name__ == "__main__":
    root_path = '/Volumes/Xuan/'
    dataset = HIBERDatasetMultiFrame(root_path, 'train')
    hor, ver, hor_boxes_wh, heatmap = dataset[0]
    print(hor.shape, ver.shape, hor_boxes_wh.shape, heatmap.shape)

