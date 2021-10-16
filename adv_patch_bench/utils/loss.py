import torch.nn as nn
import torch.nn.functional as F


class PixelwiseCELoss(nn.Module):
    def forward(self, logits, targets):
        loss = F.cross_entropy(logits, targets, reduction='none')
        return loss.mean((1, 2))


class TRADESLoss(nn.Module):
    def __init__(self, beta):
        super(TRADESLoss, self).__init__()
        self.beta = beta

    def forward(self, logits, targets):
        batch_size = logits.size(0) // 2
        cl_logits, adv_logits = logits[:batch_size], logits[batch_size:]
        cl_loss = F.cross_entropy(cl_logits, targets, reduction='mean')
        cl_probs = F.softmax(cl_logits, dim=1)
        adv_lprobs = F.log_softmax(adv_logits, dim=1)
        adv_loss = F.kl_div(adv_lprobs, cl_probs, reduction='batchmean')
        return cl_loss + self.beta * adv_loss


class KLDLoss(nn.Module):
    def __init__(self, reduction='none'):
        super(KLDLoss, self).__init__()
        assert reduction in ('none', 'mean', 'sum-non-batch')
        self.reduction = reduction

    def forward(self, cl_logits, adv_logits):
        cl_probs = F.softmax(cl_logits, dim=1)
        adv_lprobs = F.log_softmax(adv_logits, dim=1)
        if self.reduction in ('none', 'mean'):
            return F.kl_div(adv_lprobs, cl_probs, reduction=self.reduction)
        loss = F.kl_div(adv_lprobs, cl_probs, reduction='none')
        dims = tuple(range(1, loss.ndim))
        return loss.sum(dims)
