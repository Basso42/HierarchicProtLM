import torch
import torch.nn.functional as F
import torch.nn as nn

#https://www.kaggle.com/code/thedrcat/focal-multilabel-loss-in-pytorch-explained
#https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
#Comment incluent-ils le gamma dans la somme de la BCE? Ils ne l'incluent pas?

class FocalLoss(nn.modules.loss._WeightedLoss):

    def __init__(self, weight=None, gamma=1.5):

        super(FocalLoss, self).__init__(weight)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.binary_cross_entropy_with_logits(input, target, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean() #loss average over batch size

        return focal_loss
