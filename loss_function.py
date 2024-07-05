from torch import nn
import torch
from scipy.optimize import linear_sum_assignment
class LossFunction(nn.Module):
    def __init__(self, gamma=1):
        super(LossFunction, self).__init__()
        self.gamma = gamma
        self.centerloss_fn = nn.MSELoss(reduction='mean')
        self.offsetloss_fn = nn.L1Loss(reduction='mean')
        self.boxsizeloss_fn = nn.L1Loss(reduction='mean')
        self.pose_loss_fn = nn.MSELoss(reduction='sum')

    def forward(self, output_dict, gt_heatmap, gt_hor_boxes_wh, gt_hor_offset,gt_hor_center,gt_pose3d):
        hor_heatmap = output_dict['hor_heatmap']
        hor_offset = output_dict['hor_offset']
        hor_bsize = output_dict['hor_bsize']
        hor_center = output_dict['hor_center']
        scores = output_dict['scores']
        x_pose3d = output_dict['x_pose3d']

        '''
        hor_heatmap torch.Size([4, 10, 1, 64, 64])
        hor_offset torch.Size([4, 10, 4, 2]) 
        hor_bsize torch.Size([4, 10, 4, 4])
        hor_center torch.Size([4, 10, 4, 2])
        hor_scores torch.Size([4, 10, 4])
        x_pose3d torch.Size([4, 10, 4, 14, 3])
        gt_heatmap torch.Size([4, 10, 64, 64]) 
        gt_hor_boxes_wh torch.Size([4, 10, 2, 4])
        gt_hor_center torch.Size([4, 10, 2, 2])
        gt_pose3d torch.Size([4, 10, 2, 14, 3])
        '''

        center_loss = self.centerloss_fn(hor_heatmap.squeeze(2), gt_heatmap)
        total_offset_loss, total_size_loss = self.hungarian(hor_bsize, gt_hor_boxes_wh,
                                                                 hor_offset, gt_hor_offset, hor_center, gt_hor_center,scores)
        detection_loss = center_loss + total_offset_loss + total_size_loss
        pose_loss = self.compute_pose_loss(x_pose3d, gt_pose3d)
        total_loss = self.gamma * detection_loss + pose_loss

        return total_loss

    def hungarian(self, pred_boxes, gt_boxes, pred_offsets, gt_offsets, pred_centers, gt_centers, pred_scores):
        batch_size, num_frames, num_pred, _ = pred_boxes.size()
        _, _, num_gt, _ = gt_boxes.size()

        total_offset_loss = 0
        total_size_loss = 0

        for b in range(batch_size):
            for t in range(num_frames):

                pred_offset = pred_offsets[b, t]
                gt_offset = gt_offsets[b, t]

                pred_size = pred_boxes[b, t]
                gt_size = gt_boxes[b, t]
                pred_center = pred_centers[b, t]
                gt_center = gt_centers[b, t]
                pred_score = pred_scores[b, t]

                # 计算每个帧的匹配
                cost_matrix = self.compute_cost_matrix(pred_center, gt_center, pred_score)
                row_indices, col_indices = linear_sum_assignment(cost_matrix.cpu().detach().numpy())

                # 计算中心、偏移和尺寸的损失
                for row, col in zip(row_indices, col_indices):
                    total_offset_loss += self.offsetloss_fn(pred_offset[row], gt_offset[col])
                    total_size_loss += self.boxsizeloss_fn(pred_size[row], gt_size[col])

        # 归一化损失
        total_offset_loss /= (batch_size * num_frames)
        total_size_loss /= (batch_size * num_frames)
        return total_offset_loss, total_size_loss



    def compute_cost_matrix(self, pred_centers, gt_centers, pred_scores):
        num_pred = pred_centers.size(0)
        num_gt = gt_centers.size(0)
        cost_matrix = torch.zeros((num_pred, num_gt))

        for i in range(num_pred):
            for j in range(num_gt):
                cost_matrix[i, j] = -pred_scores[i] + torch.norm(pred_centers[i] - gt_centers[j])  # 使用欧几里得距离计算成本

        return cost_matrix

    def compute_pose_loss(self, pred_poses, gt_poses):
        batch_size, num_frames, num_pred, num_joints, _ = pred_poses.size()
        _, _, num_gt, _, _ = gt_poses.size()

        total_pose_loss = 0

        for b in range(batch_size):
            for t in range(num_frames):
                cost_matrix = self.compute_cost_matrix_pose(pred_poses[b, t], gt_poses[b, t])
                row_indices, col_indices = linear_sum_assignment(cost_matrix.cpu().detach().numpy())

                # 计算 3D pose 的损失
                for row, col in zip(row_indices, col_indices):
                    total_pose_loss += self.pose_loss_fn(pred_poses[b, t, row], gt_poses[b, t, col])

        # 归一化损失
        total_pose_loss /= (batch_size * num_frames * num_pred * num_joints)
        return total_pose_loss

    def compute_cost_matrix_pose(self, pred_poses, gt_poses):
        num_pred = pred_poses.size(0)
        num_gt = gt_poses.size(0)
        cost_matrix = torch.zeros((num_pred, num_gt))

        for i in range(num_pred):
            for j in range(num_gt):
                cost_matrix[i, j] = torch.norm(pred_poses[i] - gt_poses[j])  # 使用欧几里得距离计算成本

        return cost_matrix

def calculate_center(boxes):
    # 计算中心点 (cx, cy)
    centers = (boxes[..., :2] + boxes[..., 2:]) / 2
    return centers

def calculate_size(boxes):
    # 计算尺寸 (w, h)
    sizes = boxes[..., 2:] - boxes[..., :2]
    return sizes

def calculate_offset(boxes, centers):
    # 计算偏移量 (dx, dy)
    offsets = centers - boxes[..., :2]
    return offsets