from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn

from HRNet import get_pose_net
from detection_head import DetectionHead
from config import cfg
from config import update_config
import argparse
import torchvision.ops as ops

class MultiHeadAttentionEncoder(nn.Module):
    def __init__(self):
        super(MultiHeadAttentionEncoder, self).__init__()
        self.ln1 = nn.LayerNorm(128)
        self.atten = nn.MultiheadAttention(128, 8, dropout=0.1)
        self.ln2 = nn.LayerNorm(128)
        self.mlp = nn.Linear(128, 128)

    def forward(self, x):
        x1 = self.ln1(x)
        x1,_ = self.atten(x1, x1, x1) #torch.Size([56, 40, 128]) torch.Size([40, 56, 56])
        x = x1 + x
        x1 = self.ln2(x)
        x1 = self.mlp(x1)
        x = x1 + x
        return x

class PoseRegressionNetwork(nn.Module):
    def __init__(self, input_dim=128, output_dim=3):
        super(PoseRegressionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        # 输入的 x 维度: [batch_size, sequence_length, num_joints, feature_dim]
        x = torch.relu(self.fc1(x))  # 经过第一层全连接: [batch_size * sequence_length * num_joints, 128]
        x = torch.relu(self.fc2(x))  # 经过第二层全连接: [batch_size * sequence_length * num_joints, 64]
        x = self.fc3(x)  # 经过第三层全连接: [batch_size * sequence_length * num_joints, output_dim]
        return x


class Model(nn.Module):
    def __init__(self, cfg, is_train, **kwargs):
        super().__init__()
        self.backbone = get_pose_net(cfg, is_train=is_train)
        self.backbone.load_state_dict(torch.load('pose_hrnet_w48_256x192.pth', map_location='cpu'), strict=False)
        self.backbone.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.backbone.final_layer = nn.Conv2d(
            in_channels=336,
            out_channels=cfg['MODEL']['NUM_JOINTS'],
            kernel_size=cfg['MODEL']['EXTRA']['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if cfg['MODEL']['EXTRA']['FINAL_CONV_KERNEL'] == 3 else 0
        )
        self.detectionhead = DetectionHead(17, 128)
        self.k = 4
        self.proj = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1)
        self.embed = nn.Linear(4, 128)
        self.avgpool = nn.AdaptiveAvgPool1d(14)
        self.pos_embed = nn.Linear(14,128,bias=False).weight.T
        self.pos_embed2 = nn.Linear(10,128,bias=False).weight.T
        self.multihead_spatial_attention = nn.ModuleList()
        for i in range(3):
            self.multihead_spatial_attention.append(MultiHeadAttentionEncoder())
        self.multihead_temporal_attention = nn.ModuleList()
        for i in range(3):
            self.multihead_temporal_attention.append(MultiHeadAttentionEncoder())

        self.regression_head = PoseRegressionNetwork()



    def forward(self, hor, ver): # torch.Size([4, 10, 2, 256, 256]) torch.Size([4, 10, 2, 256, 256])
        batch_size = hor.size(0)
        T = hor.size(1)
        self.batch_size = batch_size
        self.T = T
        output = {}
        hor = hor.view(-1, 2, 256, 256)
        ver = ver.view(-1, 2, 256, 256)
        hor = self.backbone(hor) # torch.Size([40, 17, 64, 64])
        ver = self.backbone(ver) # torch.Size([40, 17, 64, 64])
        hor_heatmap, hor_offset, hor_bsize, hor_kpt_feats = self.detectionhead(hor)
        '''
        torch.Size([40, 1, 64, 64]) torch.Size([40, 2, 64, 64]) torch.Size([40, 4, 64, 64]) torch.Size([40, 128, 64, 64])
        '''
        output['hor_heatmap'] = hor_heatmap.view(batch_size, T, 1, 64, 64)
        # output['hor_offset'] = hor_offset.view(batch_size, T, 2, 64, 64)
        # output['hor_bsize'] = hor_bsize.view(batch_size, T, 4, 64, 64)
        '''
        torch.Size([4, 10, 1, 64, 64]) torch.Size([4, 10, 2, 64, 64]) torch.Size([4, 10, 4, 64, 64])
        '''
        ver_heatmap, ver_offset, ver_bsize, ver_kpt_feats = self.detectionhead(ver)

        e_input = self.multi_view_fusion_network(hor_heatmap, hor_offset, hor_bsize, hor_kpt_feats,ver_kpt_feats,output) # torch.Size([40, 4, 128, 14])
        e_input = e_input.view(batch_size,T,self.k,128,14)
        e_input = e_input.permute(0,1,2,4,3) #torch.Size([4, 10, 4, 14, 128])


        # 时间注意力
        e_input = e_input.permute(3,2,1,0,4).contiguous() #torch.Size([14, 4, 10, 4, 128])
        e_input = e_input.view(self.k*14,-1,128)
        mask = torch.randint(0, 2, (self.k*14,batch_size*T, 1)).bool()
        x_spatial = e_input.masked_fill(mask, 0) #torch.Size([56, 40, 128])
        PE_sp = self.pos_embed.unsqueeze(1).unsqueeze(1).repeat(1, self.k, batch_size*T, 1).view(self.k*14, batch_size*T, -1) #torch.Size([56, 40, 128])
        x_spatial = x_spatial+PE_sp
        for i in range(len(self.multihead_spatial_attention)):
            x_spatial = x_spatial + self.multihead_spatial_attention[i](x_spatial) #torch.Size([56, 40, 128])
        x_spatial = x_spatial.view(14,self.k,T,batch_size,128) #torch.Size([14, 4, 10, 4, 128])


        # 空间注意力
        x_temporal = x_spatial.permute(2, 1, 0, 3, 4).contiguous() #torch.Size([10, 4, 14, 4, 128])
        x_temporal = x_temporal.view(T*self.k,-1,128) #torch.Size([40, 56, 128])
        mask = torch.randint(0, 2, (self.k * T, batch_size * 14, 1)).bool()
        x_temporal= x_temporal.masked_fill(mask, 0)
        PE_sp = self.pos_embed2.unsqueeze(1).unsqueeze(1).repeat(1, self.k, batch_size * 14, 1).view(self.k * T,
                                                                                                   batch_size * 14, -1) #torch.Size([40, 56, 128])
        x_temporal = x_temporal + PE_sp
        for i in range(len(self.multihead_temporal_attention)):
            x_temporal = x_temporal + self.multihead_temporal_attention[i](x_temporal) #torch.Size([40, 56, 128])
        x_temporal = x_temporal.view(T, self.k, 14, batch_size, 128) #torch.Size([10, 4, 14, 4, 128])


        # 生成3D坐标的回归头
        x_temporal = x_temporal.permute(3, 0, 1, 2, 4).contiguous() #torch.Size([4, 10, 4, 14, 128])
        x_pose3d = self.regression_head(x_temporal) #torch.Size([4, 10, 4, 14, 3])
        output['x_pose3d'] = x_pose3d

        return output

    def multi_view_fusion_network(self,hor_heatmap, hor_offset, hor_bsize, hor_kpt_feats,ver_kpt_feats,output):
        topk_vals, topk_indices = torch.topk(hor_heatmap.view(hor_heatmap.size(0), -1), self.k)

        # 计算这些索引在原始热力图中的位置
        batch_indices = torch.arange(hor_heatmap.size(0)).view(-1, 1).repeat(1, self.k).flatten()
        spatial_indices = topk_indices.flatten()

        # 创建一个列表来存储所有物体的框 (x1, y1, x2, y2)
        bounding_boxes = []
        bounding_center = []
        bounding_offset = []
        bounding_bsize = []
        scores = []


        for batch_idx, spatial_idx in zip(batch_indices, spatial_indices):
            y = spatial_idx // hor_heatmap.size(3)
            x = spatial_idx % hor_heatmap.size(3)

            # 获取当前点的偏移量
            offset_x, offset_y = hor_offset[batch_idx, :, y, x]
            score = hor_heatmap[batch_idx, 0, y, x]
            scores.append(score)
            # 获取边界框大小
            # print(hor_bsize.shape)
            # exit()
            width = hor_bsize[batch_idx,0,y,x]+hor_bsize[batch_idx,2,y,x]
            height= hor_bsize[batch_idx,1, y, x] + hor_bsize[batch_idx,3, y, x]

            # 计算边界框的中心点
            center_x = (x + offset_x) * 4
            center_y = (y + offset_y) * 4
            bounding_center.append(torch.tensor((center_x, center_y)))

            # 计算边界框的左上角和右下角坐标
            x1 = center_x - width / 2
            y1 = center_y - height / 2
            x2 = center_x + width / 2
            y2 = center_y + height / 2

            # 添加到列表中
            bounding_boxes.append(torch.tensor((batch_idx, x1, y1, x2, y2)))
            # 处理偏移
            bounding_offset.append(torch.tensor((offset_x, offset_y)))
            bounding_bsize.append(torch.tensor((center_x/4 - x1/4, center_y/4 - y1/4, x2/4 - center_x/4, y2/4 - center_y/4)))

        bounding_boxes = torch.stack(bounding_boxes)
        bounding_offset = torch.stack(bounding_offset).view(self.batch_size,self.T,self.k,-1) #torch.Size([4, 10, 4, 2])
        bounding_bsize = torch.stack(bounding_bsize).view(self.batch_size,self.T,self.k,-1) #torch.Size([4, 10, 4, 4])
        bounding_center = torch.stack(bounding_center).view(self.batch_size,self.T,self.k,-1) #torch.Size([4, 10, 4, 2])
        scores = torch.stack(scores).view(self.batch_size,self.T,self.k) #torch.Size([4, 10, 4])


        output['hor_offset'] = bounding_offset
        output['hor_bsize'] = bounding_bsize
        output['hor_center'] = bounding_center
        output['scores'] = scores

        # 原图尺寸 (256, 256)，特征图尺寸 (64, 64)，缩放因子为 64 / 256 = 0.25
        bounding_boxes[:, 1:] *= 0.25

        # 使用 roi_align 提取特征
        # output_size 是输出特征的尺寸，这里假设我们想要每个 RoI 输出的尺寸为 (5, 5)
        output_size = (5, 5)
        aligned_features = ops.roi_align(hor_kpt_feats, bounding_boxes, output_size,
                                         spatial_scale=1)  # torch.Size([16, 128, 5, 5])
        aligned_features_hor = aligned_features.view(-1, self.k, 128, 5, 5).view(-1, 128, 25)  # torch.Size([16,128,25])
        aligned_features_hor = self.avgpool(aligned_features_hor) # torch.Size([16, 128, 14])
        # 提取vertical bounding box size
        bounding_boxes_ver = torch.zeros_like(bounding_boxes)  # torch.Size([16, 5])
        bounding_boxes_ver[:, 0] = bounding_boxes[:, 0]
        bounding_boxes_ver[:, 1] = bounding_boxes[:, 1]
        bounding_boxes_ver[:, 2] = 0
        bounding_boxes_ver[:, 3] = bounding_boxes[:, 3]
        bounding_boxes_ver[:, 4] = 64

        aligned_features = ops.roi_align(ver_kpt_feats, bounding_boxes_ver, output_size,
                                         spatial_scale=1)  # torch.Size([16, 128, 5, 5])
        aligned_features_ver = aligned_features.view(-1, self.k, 128, 5, 5).view(-1, 128, 25)  # torch.Size([16,128,25])
        aligned_features_ver = self.avgpool(aligned_features_ver) #torch.Size([16, 128, 14])

        # 映射到世界坐标
        aligned_features_hor = aligned_features_hor + self.embed(bounding_boxes[:, 1:] / 64).unsqueeze(-1).repeat(1, 1,
                                                                                                                  14)
        aligned_features_ver = aligned_features_ver + self.embed(bounding_boxes_ver[:, 1:] / 64).unsqueeze(-1).repeat(1,
                                                                                                                      1,
                                                                                                                      14)  # torch.Size([16, 128, 14])
        aligned_features_hor = self.proj(aligned_features_hor).view(-1, self.k, 128,14)
        aligned_features_ver = self.proj(aligned_features_ver).view(-1, self.k, 128,14)  # view之前torch.Size([16, 128, 14]) view之后torch.Size([4, 4, 128,14])
        cat_features = torch.stack([aligned_features_hor, aligned_features_ver],
                                   dim=2)  # torch.Size([4, 4, 2, 128,14])
        cat_features_exp = torch.exp(cat_features)
        weight = cat_features_exp / torch.sum(cat_features, dim=2, keepdim=True)  # torch.Size([4, 4, 2, 128,14])
        e_input = torch.sum(cat_features * weight, dim=2, keepdim=True).squeeze(
            2)  # squeeze前torch.Size([4, 4, 1, 128,14]) 后torch.Size([4, 4, 128,14])

        return e_input




if __name__ == '__main__':
    args = parse_args()

    update_config(cfg,args)

    model = Model(cfg, is_train=True)

    Output = model(torch.randn(2, 3, 256, 256))