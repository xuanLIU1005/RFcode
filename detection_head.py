import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DetectionHead, self).__init__()
        self.center_heatmap = nn.Conv2d(in_channels,1, kernel_size=1, stride=1)
        self.center_offset = nn.Conv2d(in_channels, 2, kernel_size=1, stride=1)  # x, y offsets
        self.box_size = nn.Conv2d(in_channels, 4, kernel_size=1, stride=1)  # width, height
        self.keypoint_features = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, out_channels, kernel_size=1, stride=1)
        )

    def forward(self, x):
        heatmap = self.center_heatmap(x)
        offset = self.center_offset(x)
        bsize = self.box_size(x)
        kpt_feats = self.keypoint_features(x)
        return heatmap, offset, bsize, kpt_feats
        #torch.Size([2, 1, 64, 64]) torch.Size([2, 2, 64, 64]) torch.Size([2, 2, 64, 64]) torch.Size([2, 128, 64, 64])


    '''def compute_heatmap(self, heatmap, gt_boxes):
        # 这里假设gt_boxes是[n, 4]的tensor，其中n是batch size，每个box是[x1, y1, x2, y2]
        for i in range(len(gt_boxes)):
            box = gt_boxes[i]
            width = box[2] - box[0]
            height = box[3] - box[1]
            center_x = (box[2] + box[0]) / 2
            center_y = (box[3] + box[1]) / 2
            # 将中心点坐标转换为热图的尺度
            draw_gaussian(heatmap[i], (center_x / self.down_ratio, center_y / self.down_ratio), width, height)
        return heatmap

# 这里需要实现一个高斯分布的函数来模拟heatmap的响应
def draw_gaussian(heatmap, center, width, height):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((height, width), sigma_x=width / 6, sigma_y=height / 6)
    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    torch.max(masked_heatmap, masked_gaussian, out=masked_heatmap)

def gaussian2D(shape, sigma_x, sigma_y):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return torch.from_numpy(h).float()'''

