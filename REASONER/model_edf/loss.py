import warnings
from typing import Optional
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import sys
import torch

def to_one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    # if `dim` is bigger, add singleton dim at the end
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)

    sh = list(labels.shape)

    if sh[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")

    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels


class CrossEntropy_Loss(_Loss):
    def __init__(self,
                 softmax: bool = True,
                 ce_weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean',
                 epsilon: float = 1.0,
                 ) -> None:
        super().__init__()
        self.softmax = softmax
        self.reduction = reduction
        self.epsilon = epsilon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(ce_weight)
        self.ceweight = ce_weight
        self.cross_entropy = nn.CrossEntropyLoss(weight=torch.tensor(ce_weight).to(self.device), reduction='mean')
        #self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
                You can pass logits or probabilities as input, if pass logit, must set softmax=True
            target: if target is in one-hot format, its shape should be BNH[WD],
                if it is not one-hot encoded, it should has shape B1H[WD] or BH[WD], where N is the number of classes,
                It should contain binary values
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
       """

        if len(input.shape) - len(target.shape) == 1:
            target = target.unsqueeze(1).long()
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        # target not in one-hot encode format, has shape B1H[WD]
        #print(n_pred_ch)
        #print(n_target_ch)
        if n_pred_ch != n_target_ch:
            # squeeze out the channel dimension of size 1 to calculate ce loss
            self.ce_loss = self.cross_entropy(input, torch.squeeze(target, dim=1).long())
            # convert into one-hot format to calculate ce loss
            target = to_one_hot(target, num_classes=n_pred_ch)
        else:
            # # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
            self.ce_loss = self.cross_entropy(input, torch.argmax(target, dim=1))

        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        pt = (input * target).sum(dim=1)  # BH[WD]
        #print("pt",pt)
        #print("pt1", (input * target))
        #print("pt",pt.shape)
        #print("self.ce_loss",self.ce_loss.shape)
        poly_loss = self.ce_loss + self.epsilon * (1 - pt)
        poly_loss = self.ce_loss

        if self.reduction == 'mean':
            #polyl = torch.mean(poly_loss)  # the batch and channel average
            polyl = poly_loss
        elif self.reduction == 'sum':
            polyl = torch.sum(poly_loss)  # sum over the batch and channel dims
        elif self.reduction == 'none':
            # BH[WD]
            polyl = poly_loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return (polyl)

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, output, target):
        # cr = nn.CrossEntropyLoss(weight=torch.tensor(classes_weights).to(device))
        mse = nn.MSELoss(reduction="mean")
        return mse(output, target)
        #return torch.sum(mse(output, target))
"""
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, output, target):
        # cr = nn.CrossEntropyLoss(weight=torch.tensor(classes_weights).to(device))
        mse = nn.MSELoss(reduction="mean")
        return mse(output, target)
        #return torch.sum(mse(output, target))
"""

class Feature_Level_Consistency_Loss(_Loss):
    def __init__(self) -> None:
        super().__init__()

    def compute_joint(self, view1, view2):
        """Compute the joint probability matrix P"""

        bn, k = view1.size()
        assert (view2.size(0) == bn and view2.size(1) == k)

        #print(view1.shape)
        p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)
        #print(p_i_j.shape)
        p_i_j = p_i_j.sum(dim=0)
        #print(p_i_j.shape)
        p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
        #print(p_i_j.shape)
        # print(p_i_j.t().shape)
        # print(p_i_j.shape)
        p_i_j = p_i_j / p_i_j.sum()  # normalise
        #print(p_i_j.sum())
        #print(p_i_j.shape)
        return p_i_j

    def forward(self, view1, view2, lamb=0.0, EPS=sys.float_info.epsilon):
        """Contrastive loss for maximizng the consistency"""
        _, k = view1.size()
        p_i_j = self.compute_joint(view1, view2)
        assert (p_i_j.size() == (k, k))
        # print("p_i_j",p_i_j.shape)
        p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
        p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)
        # print("p_i", p_i.shape)
        # print("p_j", p_j.shape)
        #     Works with pytorch <= 1.2
        #     p_i_j[(p_i_j < EPS).data] = EPS
        #     p_j[(p_j < EPS).data] = EPS
        #     p_i[(p_i < EPS).data] = EPS

        # Works with pytorch > 1.2
        p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
        p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
        p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

        loss1 = - p_i_j * (torch.log(p_i_j) \
                           - (lamb + 1) * torch.log(p_j) \
                           - (lamb + 1) * torch.log(p_i))
        loss = -(p_i_j * (0.15 * torch.log(p_i_j) - torch.log(p_j * p_i)))

        loss = torch.sum(loss)
        #loss = torch.mean(loss)
        return loss

class MultVariateKLD(torch.nn.Module):
    def __init__(self, reduction):
        super(MultVariateKLD, self).__init__()
        self.reduction = reduction

    def forward(self, m1, m2): # var is standard deviation
        assert len(m1.shape) == 3 and len(m2.shape) == 3, \
            'Features should have shape(bs, len, d) not ' + str(m1.shape)
        var_m1, var_m2 = m1.var(dim=-1), m2.var(dim=-1)
        m1_, m2_ = m1.mean(-1), m2.mean(-1)
        mu1, mu2 = var_m1.type(dtype=torch.float64), var_m2.type(dtype=torch.float64)
        sigma_1 = m1_.type(dtype=torch.float64)
        sigma_2 = m2_.type(dtype=torch.float64)
        #print(sigma_1[0])
        sigma_diag_1 = torch.diag_embed(sigma_1, offset=0, dim1=-2, dim2=-1)#协方差矩阵
        #print(sigma_diag_1[0])
        sigma_diag_2 = torch.diag_embed(sigma_2, offset=0, dim1=-2, dim2=-1)
        #print(sigma_diag_2.shape)

        sigma_diag_2_inv = sigma_diag_2.inverse()
        #print(sigma_diag_2_inv.shape)

        # log(det(sigma2^T)/det(sigma1))
        term_1 = (sigma_diag_2.det() / sigma_diag_1.det()).log()
        # term_1[term_1.ne(term_1)] = 0

        # trace(inv(sigma2)*sigma1)
        term_2 = torch.diagonal((torch.matmul(sigma_diag_2_inv, sigma_diag_1)), dim1=-2, dim2=-1).sum(-1)

        # (mu2-m1)^T*inv(sigma2)*(mu2-mu1)
        term_3 = torch.matmul(torch.matmul((mu2 - mu1).unsqueeze(-1).transpose(2, 1), sigma_diag_2_inv),
                              (mu2 - mu1).unsqueeze(-1)).flatten()

        # dimension of embedded space (number of mus and sigmas)
        n = mu1.shape[1]

        # Calc kl divergence on entire batch
        kl = 0.5 * (term_1 - n + term_2 + term_3)
        #print(kl.shape)

        # Calculate mean kl_d loss
        if self.reduction == 'mean':
            kl_agg = torch.mean(kl)
        elif self.reduction == 'sum':
            kl_agg = torch.sum(kl)
        else:
            raise NotImplementedError(f'Reduction type not implemented: {self.reduction}')
        print(kl.shape)
        print(kl_agg)

        return kl_agg

class MultVariateKLD1(torch.nn.Module):
    def __init__(self, reduction):
        super(MultVariateKLD1, self).__init__()
        self.reduction = reduction

    def forward(self, m1, m2): # var is standard deviation
        assert len(m1.shape) == 3 and len(m2.shape) == 3, \
            'Features should have shape(bs, len, d) not ' + str(m1.shape)
        var_m1, var_m2 = m1.var(dim=-1), m2.var(dim=-1)
        var_m_avg =  (var_m1 + var_m2)/2
        m1_, m2_ = m1.mean(-1), m2.mean(-1)
        m_avg = (m1_ + m2_)/2
        #print(m1_.shape)
        mu1, mu2, mu_avg = var_m1.type(dtype=torch.float64), var_m2.type(dtype=torch.float64), var_m_avg.type(dtype=torch.float64)
        #print(mu1.shape)
        sigma_1 = m1_.type(dtype=torch.float64)
        sigma_2 = m2_.type(dtype=torch.float64)
        sigma_avg = m_avg.type(dtype=torch.float64)
        #print(sigma_1.shape)
        sigma_diag_1 = torch.diag_embed(sigma_1, offset=0, dim1=-2, dim2=-1)
        #print(sigma_diag_1.shape)
        #print(sigma_diag_1.shape)
        sigma_diag_2 = torch.diag_embed(sigma_2, offset=0, dim1=-2, dim2=-1)

        sigma_diag_avg = torch.diag_embed(sigma_avg, offset=0, dim1=-2, dim2=-1)
        #print(sigma_diag_2.shape)

        sigma_diag_2_inv = sigma_diag_2.inverse()

        sigma_diag_avg_inv = sigma_diag_avg.inverse()
        #print(sigma_diag_2_inv.shape)

        # log(det(sigma2^T)/det(sigma1))
        term_1_1 = (sigma_diag_avg.det() / sigma_diag_1.det()).log()
        # term_1[term_1.ne(term_1)] = 0

        # trace(inv(sigma2)*sigma1)
        term_2_1 = torch.diagonal((torch.matmul(sigma_diag_avg_inv, sigma_diag_1)), dim1=-2, dim2=-1).sum(-1)

        # (mu2-m1)^T*inv(sigma2)*(mu2-mu1)
        term_3_1 = torch.matmul(torch.matmul((mu_avg - mu1).unsqueeze(-1).transpose(2, 1), sigma_diag_avg_inv),
                              (mu_avg - mu1).unsqueeze(-1)).flatten()

        # dimension of embedded space (number of mus and sigmas)
        n1 = mu1.shape[1]

        ####################################################################
        # log(det(sigma2^T)/det(sigma1))
        term_1_2 = (sigma_diag_avg.det() / sigma_diag_2.det()).log()
        # term_1[term_1.ne(term_1)] = 0

        # trace(inv(sigma2)*sigma1)
        term_2_2 = torch.diagonal((torch.matmul(sigma_diag_avg_inv, sigma_diag_2)), dim1=-2, dim2=-1).sum(-1)

        # (mu2-m1)^T*inv(sigma2)*(mu2-mu1)
        term_3_2 = torch.matmul(torch.matmul((mu_avg - mu2).unsqueeze(-1).transpose(2, 1), sigma_diag_avg_inv),
                                (mu_avg - mu2).unsqueeze(-1)).flatten()

        # dimension of embedded space (number of mus and sigmas)
        n2 = mu2.shape[1]

        # Calc kl divergence on entire batch
        kl = 0.5* (0.5 * (term_1_1 - n1 + term_2_1 + term_3_1) + 0.5 * (term_1_2 - n2 + term_2_2 + term_3_2))
        #print(kl.shape)

        # Calculate mean kl_d loss
        if self.reduction == 'mean':
            kl_agg = torch.mean(kl)
        elif self.reduction == 'sum':
            kl_agg = torch.sum(kl)
        else:
            raise NotImplementedError(f'Reduction type not implemented: {self.reduction}')

        return kl_agg





