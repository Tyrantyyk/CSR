import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from torch.distributions import MultivariateNormal as MVN


class all_weight_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, beta=0.8, lambda_=10, epoch_all=300, start_confi=20):
        super(all_weight_loss, self).__init__()
        self.num_classes = num_classes
        self.num_examp = num_examp
        self.start_confi = start_confi
        self.target = torch.zeros(num_examp, self.num_classes).cuda()
        self.beta = beta
        self.lambda_ = lambda_
        self.epoch_all = epoch_all
        self.trans = nn.Parameter(torch.eye(num_classes, num_classes, dtype=torch.float32))

        self.u = nn.Parameter(torch.empty(num_examp, 1, dtype=torch.float32))
        self.v = nn.Parameter(torch.empty(num_examp, num_classes, dtype=torch.float32))

        self.init_param(mean=0.0, std=1e-8)

    def init_param(self, mean=0., std=1e-8):
        torch.nn.init.normal_(self.u, mean=mean, std=std)
        torch.nn.init.normal_(self.v, mean=mean, std=std)

    def forward(self, index, output, output_ema, label, epoch, confi_weight):
        eps = 1e-4
        T = self.trans
        T = (T - T.min()) / (T.max() - T.min())  # 归一化
        U_square = self.u[index] ** 2 * label
        V_square = self.v[index] ** 2 * (1 - label)

        U_square = torch.clamp(U_square, 0, 1)
        V_square = torch.clamp(V_square, 0, 1)

        y_pred = F.softmax(output_ema, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.data.detach()
        beta = (0.1 - self.beta) * epoch / self.epoch_all + 1
        # beta = 0.9 - self.beta * epoch / self.epoch_all
        self.target[index] = beta * self.target[index] + (1 - beta) * (y_pred_ / y_pred_.sum(dim=1, keepdim=True))

        original_prediction = F.softmax(output, dim=1)
        prediction = torch.clamp(
            original_prediction @ T + U_square - V_square.detach(), min=eps)

        prediction = F.normalize(prediction, p=1, eps=eps)
        prediction = torch.clamp(prediction, min=eps, max=1.0)

        prediction_u = torch.clamp(
            (original_prediction @ T).detach() + U_square - V_square.detach(), min=eps)

        prediction_u = F.normalize(prediction_u, p=1, eps=eps)
        prediction_u = torch.clamp(prediction_u, min=eps, max=1.0)

        prediction_f = torch.clamp(
            original_prediction @ T.detach() + U_square.detach() - V_square.detach(), min=eps)

        prediction_f = F.normalize(prediction_f, p=1, eps=eps)
        prediction_f = torch.clamp(prediction_f, min=eps, max=1.0)

        label_one_hot = self.soft_to_hard(original_prediction.detach())

        if epoch > self.start_confi:
            MSE_loss = torch.sum(
                torch.sum(F.mse_loss((label_one_hot + U_square - V_square), label, reduction='none'), dim=1) *
                confi_weight[index].cuda()) / len(label)
            MSE_loss += torch.mean(
                -torch.sum(label * torch.log(prediction_f), dim=-1) * (1 - confi_weight[index].cuda()))
            MSE_loss += torch.mean(-torch.sum(label * torch.log(prediction_u), dim=-1) * confi_weight[index].cuda())
        else:
            MSE_loss = F.mse_loss((label_one_hot + U_square - V_square), label, reduction='sum') / len(label)

        loss = torch.mean(-torch.sum(label * torch.log(prediction), dim=-1))

        loss += MSE_loss

        avg_prediction = torch.mean(prediction, dim=0)
        prior_distr = 1.0 / self.num_classes * torch.ones_like(avg_prediction)
        avg_prediction = torch.clamp(avg_prediction, min=eps, max=1.0)
        balance_kl = torch.mean(-(prior_distr * torch.log(avg_prediction)).sum(dim=0))
        loss += 0.1 * balance_kl

        return loss

    def soft_to_hard(self, x):
        with torch.no_grad():
            return (torch.zeros(len(x), self.num_classes)).cuda().scatter_(1, (x.argmax(dim=1)).view(-1, 1), 1)

class TruncatedLoss(nn.Module):
    def __init__(self, q=0.7, k=0.5, trainset_size=50000):
        super(TruncatedLoss, self).__init__()
        self.q = q
        self.k = k
        self.weight = torch.nn.Parameter(data=torch.ones(trainset_size, 1), requires_grad=False)

    def forward(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))

        loss = ((1 - (Yg ** self.q)) / self.q) * self.weight[indexes] - ((1 - (self.k ** self.q)) / self.q) * \
               self.weight[indexes]
        loss = torch.mean(loss)

        return loss

    def update_weight(self, logits, targets, indexes):
        p = F.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        Lq = ((1 - (Yg ** self.q)) / self.q)
        Lqk = np.repeat(((1 - (self.k ** self.q)) / self.q), targets.size(0))
        Lqk = torch.from_numpy(Lqk).type(torch.cuda.FloatTensor)
        Lqk = torch.unsqueeze(Lqk, 1)

        condition = torch.gt(Lqk, Lq)
        self.weight[indexes] = condition.type(torch.cuda.FloatTensor)


class linear_mapping_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, beta=1, lambda_=10):
        super(linear_mapping_loss, self).__init__()
        self.num_classes = num_classes
        self.num_examp = num_examp
        self.target = torch.zeros(num_examp, self.num_classes).cuda()
        self.beta = beta
        self.lambda_ = lambda_

        self.vector = nn.Parameter(torch.ones(num_examp, num_classes, dtype=torch.float32))
        self.u = nn.Parameter(torch.empty(num_examp, 1, dtype=torch.float32))

        self.init_param(mean=0.0, std=1e-8)

    def init_param(self, mean=0., std=1e-8):
        torch.nn.init.normal_(self.u, mean=mean, std=std)

    def forward(self, index, output, label):
        eps = 1e-4

        U = self.u[index]
        Vector = self.vector[index]


        U = torch.clamp(U, 0, 1)

        original_prediction = F.softmax(output, dim=1)
        prediction = torch.clamp(
            original_prediction * Vector + U, min=eps)

        prediction = F.normalize(prediction, p=1, eps=eps)
        prediction = torch.clamp(prediction, min=eps, max=1.0)

        loss = torch.mean(-torch.sum(label * torch.log(prediction), dim=-1))

        # U = self.u[index]
        # Vector = self.vector[index]
        # Vector = F.softmax(Vector+label*3, dim=-1)
        #
        # U = torch.clamp(U, 0, 1)
        #
        # original_prediction = F.softmax(output, dim=1)
        #
        # loss = torch.mean(-torch.sum(torch.log(original_prediction * Vector), dim=-1))

        return loss

    def soft_to_hard(self, x):
        with torch.no_grad():
            return (torch.zeros(len(x), self.num_classes)).cuda().scatter_(1, (x.argmax(dim=1)).view(-1, 1), 1)


class sop_loss_nov(nn.Module):
    def __init__(self, num_examp, num_classes=10, beta=1, lambda_=10, epoch_all=150):
        super(sop_loss_nov, self).__init__()
        self.num_classes = num_classes

        self.u = nn.Parameter(torch.empty(num_examp, 1, dtype=torch.float32))

        self.init_param(mean=0.0, std=1e-8)

    def init_param(self, mean=0., std=1e-8):
        torch.nn.init.normal_(self.u, mean=mean, std=std)

    def forward(self, index, output, label):
        eps = 1e-4

        U_square = self.u[index] ** 2 * label

        U_square = torch.clamp(U_square, 0, 1)

        original_prediction = F.softmax(output, dim=1)
        prediction = torch.clamp(
            original_prediction + U_square, min=eps)

        prediction = F.normalize(prediction, p=1, eps=eps)
        prediction = torch.clamp(prediction, min=eps, max=1.0)

        label_one_hot = self.soft_to_hard(output.detach())

        MSE_loss = F.mse_loss((label_one_hot + U_square), label,  reduction='sum') / len(label)
        loss = torch.mean(-torch.sum(label * torch.log(prediction), dim=-1))

        loss += MSE_loss

        avg_prediction = torch.mean(prediction, dim=0)
        prior_distr = 1.0 / self.num_classes * torch.ones_like(avg_prediction)

        avg_prediction = torch.clamp(avg_prediction, min=eps, max=1.0)

        balance_kl = torch.mean(-(prior_distr * torch.log(avg_prediction)).sum(dim=0))

        loss += 0.1 * balance_kl

        return loss

    def soft_to_hard(self, x):
        with torch.no_grad():
            return (torch.zeros(len(x), self.num_classes)).cuda().scatter_(1, (x.argmax(dim=1)).view(-1, 1), 1)


class sop_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, beta=1, lambda_=10, epoch_all=150):
        super(sop_loss, self).__init__()
        self.num_classes = num_classes

        self.u = nn.Parameter(torch.empty(num_examp, 1, dtype=torch.float32))
        self.v = nn.Parameter(torch.empty(num_examp, num_classes, dtype=torch.float32))

        self.init_param(mean=0.0, std=1e-8)

    def init_param(self, mean=0., std=1e-8):
        torch.nn.init.normal_(self.u, mean=mean, std=std)
        torch.nn.init.normal_(self.v, mean=mean, std=std)

    def forward(self, index, output, label):
        eps = 1e-4

        U_square = self.u[index] ** 2 * label
        V_square = self.v[index] ** 2 * (1 - label)

        U_square = torch.clamp(U_square, 0, 1)
        V_square = torch.clamp(V_square, 0, 1)

        original_prediction = F.softmax(output, dim=1)
        prediction = torch.clamp(
            original_prediction + U_square - V_square.detach(), min=eps)

        prediction = F.normalize(prediction, p=1, eps=eps)
        prediction = torch.clamp(prediction, min=eps, max=1.0)

        label_one_hot = self.soft_to_hard(output.detach())

        MSE_loss = F.mse_loss((label_one_hot + U_square - V_square), label,  reduction='sum') / len(label)
        loss = torch.mean(-torch.sum(label * torch.log(prediction), dim=-1))

        loss += MSE_loss

        avg_prediction = torch.mean(prediction, dim=0)
        prior_distr = 1.0 / self.num_classes * torch.ones_like(avg_prediction)

        avg_prediction = torch.clamp(avg_prediction, min=eps, max=1.0)

        balance_kl = torch.mean(-(prior_distr * torch.log(avg_prediction)).sum(dim=0))

        loss += 0.1 * balance_kl

        return loss

    def soft_to_hard(self, x):
        with torch.no_grad():
            return (torch.zeros(len(x), self.num_classes)).cuda().scatter_(1, (x.argmax(dim=1)).view(-1, 1), 1)


class sop_trans_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, beta=1, lambda_=10, epoch_all=150):
        super(sop_trans_loss, self).__init__()
        self.num_classes = num_classes
        self.num_examp = num_examp
        self.trans = nn.Parameter(torch.eye(num_classes, num_classes, dtype=torch.float32))

        self.u = nn.Parameter(torch.empty(num_examp, 1, dtype=torch.float32))
        self.v = nn.Parameter(torch.empty(num_examp, num_classes, dtype=torch.float32))

        self.init_param(mean=0.0, std=1e-8)

    def init_param(self, mean=0., std=1e-8):
        torch.nn.init.normal_(self.u, mean=mean, std=std)
        torch.nn.init.normal_(self.v, mean=mean, std=std)

    def forward(self, index, output, label):
        eps = 1e-4
        T = self.trans
        T = (T - T.min()) / (T.max() - T.min())  # 归一化

        U_square = self.u[index] ** 2 * label
        V_square = self.v[index] ** 2 * (1 - label)

        U_square = torch.clamp(U_square, 0, 1)
        V_square = torch.clamp(V_square, 0, 1)
        T = torch.clamp(T, 0, 1)

        original_prediction = F.softmax(output, dim=1)
        prediction = torch.clamp(
            original_prediction @ T.detach()  + U_square - V_square.detach(), min=eps)

        prediction = F.normalize(prediction, p=1, eps=eps)
        prediction = torch.clamp(prediction, min=eps, max=1.0)

        label_one_hot = self.soft_to_hard(output.detach())

        MSE_loss = F.mse_loss((label_one_hot @ T + U_square - V_square), label,  reduction='sum') / len(label)
        loss = torch.mean(-torch.sum(label * torch.log(prediction), dim=-1))

        loss += MSE_loss

        avg_prediction = torch.mean(prediction, dim=0)
        prior_distr = 1.0 / self.num_classes * torch.ones_like(avg_prediction)

        avg_prediction = torch.clamp(avg_prediction, min=eps, max=1.0)

        balance_kl = torch.mean(-(prior_distr * torch.log(avg_prediction)).sum(dim=0))

        loss += 0.1 * balance_kl

        return loss

    def soft_to_hard(self, x):
        with torch.no_grad():
            return (torch.zeros(len(x), self.num_classes)).cuda().scatter_(1, (x.argmax(dim=1)).view(-1, 1), 1)


class sop_balance_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, beta=1, lambda_=10, epoch_all=150):
        super(sop_balance_loss, self).__init__()
        self.num_classes = num_classes
        self.num_examp = num_examp
        self.target = torch.zeros(num_examp, self.num_classes).cuda()
        self.beta = beta
        self.lambda_ = lambda_
        self.epoch_all = epoch_all
        self.trans = nn.Parameter(torch.eye(num_classes, num_classes, dtype=torch.float32))

        self.u = nn.Parameter(torch.empty(num_examp, 1, dtype=torch.float32))
        self.v = nn.Parameter(torch.empty(num_examp, num_classes, dtype=torch.float32))

        self.init_param(mean=0.0, std=1e-8)

    def init_param(self, mean=0., std=1e-8):
        torch.nn.init.normal_(self.u, mean=mean, std=std)
        torch.nn.init.normal_(self.v, mean=mean, std=std)

    def forward(self, index, output, output_ema, label, epoch, confi_weight):
        eps = 1e-4
        T = self.trans
        T = (T - T.min()) / (T.max() - T.min())  # 归一化

        U_square = self.u[index] ** 2 * label
        V_square = self.v[index] ** 2 * (1 - label)

        U_square = torch.clamp(U_square, 0, 1)
        V_square = torch.clamp(V_square, 0, 1)

        y_pred = F.softmax(output_ema, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.data.detach()
        beta = (0.1 - self.beta) * epoch / self.epoch_all + 1
        self.target[index] = beta * self.target[index] + (1 - beta) * (y_pred_ / y_pred_.sum(dim=1, keepdim=True))

        original_prediction = F.softmax(output, dim=1)
        prediction = torch.clamp(
            original_prediction @ T.detach()  + U_square - V_square.detach(), min=eps)

        prediction = F.normalize(prediction, p=1, eps=eps)
        prediction = torch.clamp(prediction, min=eps, max=1.0)

        label_one_hot = self.soft_to_hard(output.detach())

        MSE_loss = torch.sum(
            torch.sum(F.mse_loss((label_one_hot @ T + U_square - V_square), label, reduction='none'), dim=1) *
            confi_weight[index].cuda()) / len(label)
        loss = torch.mean(-torch.sum(label * torch.log(prediction), dim=-1) * (1 - confi_weight[index]).cuda())

        loss += MSE_loss

        return loss

    def soft_to_hard(self, x):
        with torch.no_grad():
            return (torch.zeros(len(x), self.num_classes)).cuda().scatter_(1, (x.argmax(dim=1)).view(-1, 1), 1)


class sop_noconfi_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, beta=1, lambda_=10, epoch_all=150):
        super(sop_noconfi_loss, self).__init__()
        self.num_classes = num_classes
        self.num_examp = num_examp
        self.target = torch.zeros(num_examp, self.num_classes).cuda()
        self.beta = beta
        self.lambda_ = lambda_
        self.epoch_all = epoch_all
        self.trans = nn.Parameter(torch.eye(num_classes, num_classes, dtype=torch.float32))

        self.u = nn.Parameter(torch.empty(num_examp, 1, dtype=torch.float32))
        self.v = nn.Parameter(torch.empty(num_examp, num_classes, dtype=torch.float32))

        self.init_param(mean=0.0, std=1e-8)

    def init_param(self, mean=0., std=1e-8):
        torch.nn.init.normal_(self.u, mean=mean, std=std)
        torch.nn.init.normal_(self.v, mean=mean, std=std)

    def forward(self, index, output, output_ema, label, epoch, confi_weight):
        eps = 1e-4
        T = self.trans @ self.trans.t()
        T = (T - T.min()) / (T.max() - T.min())  # 归一化

        U_square = self.u[index] ** 2 * label
        V_square = self.v[index] ** 2 * (1 - label)

        U_square = torch.clamp(U_square, 0, 1)
        V_square = torch.clamp(V_square, 0, 1)
        T = torch.clamp(T, 0, 1)

        y_pred = F.softmax(output_ema, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.data.detach()
        beta = (0.1 - self.beta) * epoch / self.epoch_all + 1
        self.target[index] = beta * self.target[index] + (1 - beta) * (y_pred_ / y_pred_.sum(dim=1, keepdim=True))

        original_prediction = F.softmax(output, dim=1)
        prediction = torch.clamp(
            original_prediction @ T.detach()  + U_square - V_square.detach(), min=eps)

        prediction = F.normalize(prediction, p=1, eps=eps)
        prediction = torch.clamp(prediction, min=eps, max=1.0)

        label_one_hot = self.soft_to_hard(output.detach())

        MSE_loss = F.mse_loss((label_one_hot @ T + U_square - V_square), label, reduction='sum') / len(label)
        loss = torch.mean(-torch.sum(label * torch.log(prediction), dim=-1))

        loss += MSE_loss

        return loss

    def soft_to_hard(self, x):
        with torch.no_grad():
            return (torch.zeros(len(x), self.num_classes)).cuda().scatter_(1, (x.argmax(dim=1)).view(-1, 1), 1)
