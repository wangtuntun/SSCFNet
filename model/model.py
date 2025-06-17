import torch
from torch import nn
from .SSCFNet.SSCFNet import SSCFNet as SSCFNetModel
from .SSCFNet.Config import get_config

class SSCFNet(nn.Module):

    def __init__(self, mode):
        super(SSCFNet, self).__init__()

        self.config_vit = get_config()
        self.mode = mode

        if self.mode == 'train':
            self.model = SSCFNetModel(config=self.config_vit, n_channels=1, mode='train', deepsuper=True)
        elif self.mode == 'test':
            self.model = SSCFNetModel(config=self.config_vit, n_channels=1, mode='test', deepsuper=True)
        else:
            raise ValueError("Unkown self.mode")
        
        self.cal_loss = nn.BCELoss(reduction='mean')#nn.BCELoss(size_average=True)
        
    def forward(self, img):
        return self.model(img)
    
    def loss(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                gt_mask = gt_masks[i]
                loss = self.cal_loss(pred, gt_mask)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        elif isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.cal_loss(pred, gt_masks)
                a.append(loss)
            loss_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5]
            return loss_total

        else:
            loss = self.cal_loss(preds, gt_masks)
            return loss


if __name__ == '__main__':
    model = SSCFNet(mode='train')
    input_tensor = torch.randn((1, 3, 256, 256))
    outputs = model(input_tensor)
    for output in outputs:
        print(type(output))
        print(output.shape)

