import torch
import torch.nn.functional as F

class FocalLoss(nn.modules.loss._WeightedLoss):

    def __init__(self, weight=None, gamma=1.5):

        super(FocalLoss, self).__init__(weight)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        return focal_loss
