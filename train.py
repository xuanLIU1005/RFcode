from hiberdataset import HIBERDatasetSingleFrame, HIBERDatasetMultiFrame
from torch.utils.data import DataLoader

import torch

from model import Model
import argparse

from config import cfg
from config import update_config
from loss_function import LossFunction

from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='experiments/coco/hrnet/w48_256x192_adam_lr1e-3.yaml',
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args

def main(cfg):
    dataset = HIBERDatasetMultiFrame('/Users/xuanxuan/Desktop/', 'train')
    data_loader = DataLoader(dataset, 4, shuffle=True, num_workers=0)
    model = Model(cfg, is_train=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    loss_fn = LossFunction()
    for epoch in range(1):
        model.train()
        loss_log = 0
        for hor, ver, hor_boxes_wh, heatmap, hor_offset, hor_center, pose3d in tqdm(data_loader):
            optimizer.zero_grad()
            output_dict = model(hor, ver)
            loss = loss_fn(output_dict, heatmap, hor_boxes_wh, hor_offset, hor_center, pose3d)
            loss.backward()
            optimizer.step()
            loss_log += loss.item()
            print(loss.item())
        loss_log = loss_log / len(data_loader)
        print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss_log))
        torch.save(model.state_dict(),f'save_model/baseline_model_{epoch}.pth')




if __name__ == '__main__':
    args = parse_args()
    update_config(cfg,args)
    main(cfg)

