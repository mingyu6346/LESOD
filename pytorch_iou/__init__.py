

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def _iou(pred, target, size_average = True):

    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
        Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
        IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b

class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):

        return _iou(pred, target, self.size_average)

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        :param alpha: 平衡正负样本的参数（可调节），默认为 1.0
        :param gamma: 调节因子 gamma，默认为 2.0
        :param reduction: 损失的返回方式，'none' | 'mean' | 'sum'，默认为 'mean'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        :param inputs: 模型的预测值，shape (N, 1) 或 (N,) (假设为logits)
        :param targets: 真实标签，shape (N,)，其中每个值为0或1
        :return: 计算后的 Focal Loss
        """
        # 将输入的 logits 转化为概率
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # 预测概率 p_t
        pt = torch.exp(-BCE_loss)

        # focal loss 的主要公式
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        # 根据 reduction 参数返回损失
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# 使用示例
if __name__ == "__main__":
    # 创建一个Focal Loss实例
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    # 生成模拟数据
    inputs = torch.tensor([0.2, 0.8, 0.6, 0.4])  # 假设logits
    targets = torch.tensor([0, 1, 1, 0], dtype=torch.float32)  # 真实标签

    # 计算Focal Loss
    loss = criterion(inputs, targets)
    print(f"Focal Loss: {loss.item()}")

