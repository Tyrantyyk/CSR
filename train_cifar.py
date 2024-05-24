from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from net import *
from sklearn.mixture import GaussianMixture
from utils import *
from fmix import *
from losses import *
import wandb
import time

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=256, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.05, type=float, help='initial learning rate')
parser.add_argument('--noise_type', type=str, help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100',
                    default='clean')
parser.add_argument('--noise_path', type=str, help='path of CIFAR-10_human.pt', default=None)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--noise_mode', default='sym')
parser.add_argument('--num_epochs', default=150, type=int)
parser.add_argument('--seed', default=123, type=int)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=100, type=int)
parser.add_argument('--data_path', default=None, type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--rho_range', default='0.5,0.5', type=str,
                    help='ratio of clean labels (rho)')
parser.add_argument('--tau', default=0.99, type=float,
                    help='high-confidence selection threshold')
parser.add_argument('--pretrain_ep', default=10, type=int)
parser.add_argument('--warmup_ep', default=50, type=int)
parser.add_argument('--threshold', default=0.95, type=float)
parser.add_argument('--fmix', action='store_true', default=False)
parser.add_argument('--start_expand', default=100, type=int)
parser.add_argument('--examp', default=50000, type=int)
parser.add_argument('--save_note', type=str, default='')
parser.add_argument('--use_wandb', type=bool, default=True)
parser.add_argument('--is_human', action='store_true', default=False)
parser.add_argument('--lr_u', default=1, type=int)
parser.add_argument('--lr_v', default=100, type=int)
parser.add_argument('--lr_trans', default=0.00001, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)


args = parser.parse_args()
wandb.init(project="Promix_based", entity="tyrantyyk")
wandb.config.update(args)

[args.rho_start, args.rho_end] = [float(item) for item in args.rho_range.split(',')]
print(args)

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if args.noise_mode == "IDN":
    import dataloader_IDN as dataloader
else:
    import dataloader_cifar as dataloader


# Hyper Parameters
noise_type_map = {'clean': 'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1',
                  'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label',
                  'noisy100': 'noisy_label'}
args.noise_type = noise_type_map[args.noise_type]
# load dataset
if args.data_path is None:
    if args.dataset == 'cifar10':
        args.data_path = './data/cifar-10'
    elif args.dataset == 'cifar100':
        args.data_path = './data/cifar-100'
    else:
        pass

if args.noise_path is None:
    if args.dataset == 'cifar10':
        args.noise_path = './data/CIFAR-10_human.pt'
    elif args.dataset == 'cifar100':
        args.noise_path = './data/CIFAR-100_human.pt'
    else:
        pass

def momentum_update_ema(net, ema_model, eman=False, momentum=0.999):
    if eman:
        state_dict_main = net.state_dict()
        state_dict_ema = ema_model.state_dict()
        for (k_main, v_main), (k_ema, v_ema) in zip(state_dict_main.items(), state_dict_ema.items()):
            assert k_main == k_ema, "state_dict names are different!"
            assert v_main.shape == v_ema.shape, "state_dict shapes are different!"
            if 'num_batches_tracked' in k_ema:
                v_ema.copy_(v_main)
            else:
                v_ema.copy_(v_ema * momentum + (1. - momentum) * v_main)
    else:
        for param_q, param_k in zip(net.parameters(), ema_model.parameters()):
            param_k.data = param_k.data * 0.999 + param_q.data * (1. - 0.999)

# Training
def train(epoch, net, ema_net, net2, ema_net2, labeled_trainloader, prob1, prob2, confi_weight1, confi_weight2):
    net.train()
    net2.train()
    w = linear_rampup2(epoch, args.warmup_ep)
    w_x1 = torch.tensor(prob1)
    w_x22 = torch.tensor(prob2)
    for batch_idx, (inputs_x, inputs_x2, labels_x, index, true_labels) in enumerate(labeled_trainloader):
        batch_size = inputs_x.size(0)
        w_x = w_x1[index]
        w_x2 = w_x22[index]

        # Transform label to one-hot
        loss_hard, loss_cr, loss_hard2, loss_cr2, sop_loss, sop_loss2, loss_ce, loss_ce2 = 0, 0, 0, 0, 0, 0, 0, 0
        d1_label = labels_x.clone().cuda()
        d1_label2 = labels_x.clone().cuda()
        labels_x2 = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)

        w_x = w_x.view(-1, 1).type(torch.FloatTensor)
        w_x2 = w_x2.view(-1, 1).type(torch.FloatTensor)

        index = index.cuda()
        inputs_x, inputs_x2, labels_x, labels_x2, w_x, w_x2 = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), labels_x2.cuda(), w_x.cuda(), w_x2.cuda()
        outputs_x = net(inputs_x)
        outputs_x2 = net(inputs_x2)
        outputs_a = ema_net(inputs_x)
        outputs_x21 = net2(inputs_x)
        outputs_x22 = net2(inputs_x2)
        outputs_a2 = ema_net2(inputs_x)

        # loss 1 for net 1
        with torch.no_grad():
            # label refinement of labeled samples
            px = train_loss1.target[index]
            px2 = train_loss2.target[index]
            plabel = torch.argmax(px, dim=1)
            plabel2 = torch.argmax(px2, dim=1)
            pred_net = F.one_hot(px.max(dim=1)[1], args.num_class).float()
            pred_net2 = F.one_hot(px2.max(dim=1)[1], args.num_class).float()

            if epoch >= args.num_epochs - args.start_expand:
                # correct
                high_conf_correct = px.max(dim=1)[0] > args.tau
                high_conf_correct2 = px2.max(dim=1)[0] > args.tau
                d1_label[high_conf_correct] = plabel[high_conf_correct]
                d1_label2[high_conf_correct2] = plabel2[high_conf_correct2]
                labels_x = torch.zeros(batch_size, args.num_class).cuda().scatter_(1, d1_label.view(-1, 1), 1)
                labels_x2 = torch.zeros(batch_size, args.num_class).cuda().scatter_(1, d1_label2.view(-1, 1), 1)
                w_x[high_conf_correct] = 1
                w_x2[high_conf_correct2] = 1

                # dismatch
                dismatch = px.max(dim=1)[1] != px2.max(dim=1)[1]
                w_x[dismatch] = 0
                w_x2[dismatch] = 0

            pseudo_label_l = labels_x * w_x + pred_net * (1 - w_x)
            pseudo_label_l2 = labels_x2 * w_x2 + pred_net2 * (1 - w_x2)
            idx_chosen = torch.where(w_x == 1)[0]
            idx_chosen2 = torch.where(w_x2 == 1)[0]

        l = np.random.beta(4, 4)
        l = max(l, 1 - l)
        X_w_c = inputs_x[idx_chosen]
        pseudo_label_c = pseudo_label_l[idx_chosen]
        idx = torch.randperm(X_w_c.size(0))
        X_w_c_rand = X_w_c[idx]
        pseudo_label_c_rand = pseudo_label_c[idx]
        X_w_c_mix = l * X_w_c + (1 - l) * X_w_c_rand
        pseudo_label_c_mix = l * pseudo_label_c + (1 - l) * pseudo_label_c_rand
        logits_mix = net(X_w_c_mix)
        loss_mix = CEsoft(logits_mix, targets=pseudo_label_c_mix).mean()
        # mixup loss
        x_fmix = fmix(X_w_c)
        logits_fmix = net(x_fmix)
        loss_fmix = fmix.loss(logits_fmix, (pseudo_label_c.detach()).long())

        if epoch < args.num_epochs - args.start_expand:
            sop_loss = train_loss1(index, outputs_x, outputs_x, labels_x, epoch, confi_weight1)
        else:
            loss_cr = CEsoft(outputs_x2[idx_chosen], targets=pseudo_label_l[idx_chosen]).mean()
            # loss_ce = CEsoft(outputs_x[idx_chosen], targets=pseudo_label_l[idx_chosen]).mean()
            update_pseudolabel = train_loss1(index, outputs_x, outputs_a, labels_x, epoch, confi_weight1)

        loss_net1 = sop_loss + w * (loss_cr + loss_mix + loss_fmix) + loss_ce


        l = np.random.beta(4, 4)
        l = max(l, 1 - l)
        X_w_c = inputs_x[idx_chosen2]
        pseudo_label_c = pseudo_label_l2[idx_chosen2]
        idx = torch.randperm(X_w_c.size(0))
        X_w_c_rand = X_w_c[idx]
        pseudo_label_c_rand = pseudo_label_c[idx]
        X_w_c_mix2 = l * X_w_c + (1 - l) * X_w_c_rand
        pseudo_label_c_mix2 = l * pseudo_label_c + (1 - l) * pseudo_label_c_rand
        logits_mix2 = net2(X_w_c_mix2)
        loss_mix2 = CEsoft(logits_mix2, targets=pseudo_label_c_mix2).mean()
        # mixup loss
        x_fmix2 = fmix(X_w_c)
        logits_fmix2 = net2(x_fmix2)
        loss_fmix2 = fmix.loss(logits_fmix2, (pseudo_label_c.detach()).long())

        if epoch < args.num_epochs - args.start_expand:
            sop_loss2 = train_loss2(index, outputs_x21, outputs_x21, labels_x2, epoch, confi_weight2)
        else:
            loss_cr2 = CEsoft(outputs_x22[idx_chosen2], targets=pseudo_label_l2[idx_chosen2]).mean()
            # loss_ce2 = CEsoft(outputs_x21[idx_chosen2], targets=pseudo_label_l2[idx_chosen2]).mean()
            update_pseudolabel = train_loss2(index, outputs_x21, outputs_a2, labels_x2, epoch, confi_weight2)

        loss_net2 = sop_loss2 + w * (loss_cr2 + loss_mix2 + loss_fmix2) + loss_ce2

        loss = loss_net1 + loss_net2

        # compute gradient and do SGD step
        optimizer1.zero_grad()
        if epoch < args.num_epochs - args.start_expand:
            optimizer_trans.zero_grad()
            optimizer_trans2.zero_grad()
            optimizer_overparametrization.zero_grad()
            optimizer_overparametrization2.zero_grad()
        loss.backward()
        optimizer1.step()
        if epoch < args.num_epochs - args.start_expand:
            optimizer_overparametrization.step()
            optimizer_overparametrization2.step()
            optimizer_trans.step()
            optimizer_trans2.step()
        momentum_update_ema(net, ema_net, eman=True)
        momentum_update_ema(net2, ema_net2, eman=True)

        wandb.log({"loss_net1": loss})



def warmup(epoch, net, ema1, net2, ema2, dataloader):
    net.train()
    net2.train()
    true_label = torch.zeros(50000)
    targets_list = torch.zeros(50000)
    for batch_idx, (inputs_w, inputs_s, labels, index, true) in enumerate(dataloader):
        batch_size = inputs_w.size(0)
        inputs_w, inputs_s, labels = inputs_w.cuda(), inputs_s.cuda(), labels.cuda()
        outputs = net(inputs_w)
        outputs21 = net2(inputs_w)
        true_label[index] = true.float().cpu()
        targets_list[index] = labels.float().cpu()

        # network 1
        if args.noise_mode == 'asym':
            penalty = conf_penalty(outputs)
        else:
            penalty = 0
        ce_loss = CEloss(outputs, labels)
        loss1 = ce_loss + penalty

        y_pred = F.softmax(outputs, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.data.detach()
        beta = (0.1 - train_loss1.beta) * epoch / train_loss1.epoch_all + 1
        train_loss1.target[index] = beta * train_loss1.target[index] + (1 - beta) * (
                y_pred_ / y_pred_.sum(dim=1, keepdim=True))

        # network 2
        if args.noise_mode == 'asym':
            penalty2 = conf_penalty(outputs21)
        else:
            penalty2 = 0
        ce_loss2 = CEloss(outputs21, labels)
        loss2 = ce_loss2 + penalty2

        y_pred = F.softmax(outputs21, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.data.detach()
        beta = (0.1 - train_loss2.beta) * epoch / train_loss2.epoch_all + 1
        train_loss2.target[index] = beta * train_loss2.target[index] + (1 - beta) * (
                y_pred_ / y_pred_.sum(dim=1, keepdim=True))

        loss = loss1 + loss2

        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()
        momentum_update_ema(net, ema1, eman=True)
        momentum_update_ema(net2, ema2, eman=True)
        wandb.log({"warmup loss": loss})

def test_train(epoch, net1, net2, dataloader):
    net1.eval()
    net2.eval()
    correctmean = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, _, targets, index, true) in enumerate(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)

            outputs_mean = (outputs1 + outputs2) / 2
            _, predicted_mean = torch.max(outputs_mean, 1)

            total += targets.size(0)
            correctmean += predicted_mean.eq(targets).cpu().sum().item()
    accmean = correctmean / total

    wandb.log({"train acc": accmean})

def test(epoch, net1, net2):
    net1.eval()
    net2.eval()
    correct = 0
    correct2 = 0
    correctmean = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)

            score1, predicted = torch.max(outputs1, 1)
            score2, predicted_2 = torch.max(outputs2, 1)
            outputs_mean = (outputs1 + outputs2) / 2
            _, predicted_mean = torch.max(outputs_mean, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
            correct2 += predicted_2.eq(targets).cpu().sum().item()
            correctmean += predicted_mean.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    acc2 = 100. * correct2 / total
    accmean = 100. * correctmean / total
    lr = optimizer1.state_dict()['param_groups'][0]['lr']

    print("| Test Epoch #%d\t Acc Net1: %.2f%%, Acc Net2: %.2f%% Acc Mean: %.2f%%\n" % (epoch, acc, acc2, accmean))
    test_log.write('Epoch:%d   Accuracy:%.2f\n' % (epoch, acc))
    test_log.flush()
    wandb.log({"test acc1": acc,
               "test acc2": acc2,
               "test acc_mean": accmean,
               "epoch": epoch,
               "lr": lr})

def test_ema(epoch, net1, net2):
    net1.eval()
    net2.eval()
    correct = 0
    correct2 = 0
    correctmean = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)

            score1, predicted = torch.max(outputs1, 1)
            score2, predicted_2 = torch.max(outputs2, 1)
            outputs_mean = (outputs1 + outputs2) / 2
            _, predicted_mean = torch.max(outputs_mean, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
            correct2 += predicted_2.eq(targets).cpu().sum().item()
            correctmean += predicted_mean.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    acc2 = 100. * correct2 / total
    accmean = 100. * correctmean / total
    lr = optimizer1.state_dict()['param_groups'][0]['lr']

    print("| Test Epoch #%d\t Acc Super teacher: %.2f%%, Acc Net2: %.2f%% Acc Mean: %.2f%%\n" % (epoch, acc, acc2, accmean))
    test_log.write('Epoch:%d   Accuracy:%.2f\n' % (epoch, acc))
    test_log.flush()

def cleanset_eval(teacher, student, true_label, targets_list):

    # student net
    idx_chosen_student = student == 1
    precision_s = (targets_list[idx_chosen_student] == true_label[idx_chosen_student]).sum() * 100 / idx_chosen_student.sum()
    recall_s = (targets_list[idx_chosen_student] == true_label[idx_chosen_student]).sum() * 100 / (targets_list == true_label).sum()

    # teacher net
    idx_chosen_teacher = teacher == 1
    precision_t = (targets_list[idx_chosen_teacher] == true_label[idx_chosen_teacher]).sum() * 100 / idx_chosen_teacher.sum()
    recall_t = (targets_list[idx_chosen_teacher] == true_label[idx_chosen_teacher]).sum() * 100 / (targets_list == true_label).sum()

    wandb.log({"precision_s": precision_s,
               "recall_s": recall_s,
               "precision_t": precision_t,
               "recall_t": recall_t
               })

def eval_train(model, rho, id):
    model.eval()
    losses = torch.zeros(50000)
    true_label = torch.zeros(50000)
    targets_list = torch.zeros(50000)
    confi_weight = torch.zeros(50000)
    num_class = 0
    with torch.no_grad():
        for batch_idx, (inputs, _, targets, index, label) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            num_class = outputs.shape[1]
            if id == 0:
                confi_w = CE(train_loss1.target[index], targets)
            else:
                confi_w = CE(train_loss2.target[index], targets)
            targets_cpu = targets.cpu()
            true_label[index] = label.float()
            for b in range(inputs.size(0)):
                losses[index[b]] = confi_w[b]
                confi_weight[index[b]] = confi_w[b]
                targets_list[index[b]] = targets_cpu[b]

    losses = (losses - losses.min()) / (losses.max() - losses.min())
    confi_weight = (confi_weight - confi_weight.min()) / (confi_weight.max() - confi_weight.min())

    prob = np.zeros(targets_list.shape[0])
    idx_chosen_sm = []
    min_len = 1e10
    for j in range(num_class):
        indices = np.where(targets_list.cpu().numpy() == j)[0]
        if len(indices) == 0:
            continue
        bs_j = targets_list.shape[0] * (1. / num_class)
        pseudo_loss_vec_j = losses[indices]
        sorted_idx_j = pseudo_loss_vec_j.sort()[1].cpu().numpy()
        partition_j = max(min(int(math.ceil(bs_j * rho)), len(indices)), 1)
        # at least one example
        idx_chosen_sm.append(indices[sorted_idx_j[:partition_j]])
        min_len = min(min_len, partition_j)

    idx_chosen_sm = np.concatenate(idx_chosen_sm)
    prob[idx_chosen_sm] = 1

    return prob, targets_list, true_label, confi_weight


def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)


class NegEntropy(object):
    def __call__(self, outputs):
        outputs = outputs.clamp(min=1e-12)
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


def create_model():
    model = DualNet(args.num_class)
    # model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model


stats_log = open('./checkpoint/%s_%s_%s' % (args.dataset, args.noise_type, args.num_epochs) + '_stats.txt', 'w')
test_log = open('./checkpoint/%s_%s_%s' % (args.dataset, args.noise_type, args.num_epochs) + '_acc.txt', 'w')

warm_up = args.pretrain_ep

loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size,
                                     num_workers=6, \
                                     root_dir=args.data_path, log=stats_log,
                                     noise_file='%s/%.1f_%s.json' % (args.data_path, args.r, args.noise_mode))

print('| Building net')
dualnet = create_model()
ema_net = create_model()

for param_main, param_ema in zip(dualnet.net1.parameters(), ema_net.net1.parameters()):
    param_ema.data.copy_(param_main.data)  # initialize
    param_ema.requires_grad = False  # not update by gradient
for param_main, param_ema in zip(dualnet.net2.parameters(), ema_net.net2.parameters()):
    param_ema.data.copy_(param_main.data)  # initialize
    param_ema.requires_grad = False  # not update by gradient

train_loss1 = sop_balance_loss(50000, args.num_class).cuda()
train_loss2 = sop_balance_loss(50000, args.num_class).cuda()
# build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
reparam_params = [{'params': train_loss1.u, 'lr': args.lr_u, 'weight_decay': 0},
                  {'params': train_loss1.v, 'lr': args.lr_v, 'weight_decay': 0}]
reparam_params2 = [{'params': train_loss2.u, 'lr': args.lr_u, 'weight_decay': 0},
                   {'params': train_loss2.v, 'lr': args.lr_v, 'weight_decay': 0}]
trans_params = [{'params': train_loss1.trans, 'lr': args.lr_trans, 'weight_decay': 0}]
trans_params2 = [{'params': train_loss2.trans, 'lr': args.lr_trans, 'weight_decay': 0}]

conf_penalty = NegEntropy()
optimizer_overparametrization = optim.SGD(reparam_params)
optimizer_overparametrization2 = optim.SGD(reparam_params2)
optimizer1 = optim.SGD([{'params': dualnet.net1.parameters()},
                        {'params': dualnet.net2.parameters()}], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
optimizer_trans = optim.SGD(trans_params)
optimizer_trans2 = optim.SGD(trans_params2)
scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, args.num_epochs)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_trans, args.num_epochs - args.start_expand)
scheduler3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_trans2, args.num_epochs - args.start_expand)

fmix = FMix()
fmix2 = FMix()
CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
CEsoft = CE_Soft_Label()

eval_loader = loader.run('warmup')
test_loader = loader.run('test')

start = time.time()
for epoch in range(args.num_epochs):
    # adjust_learning_rate(args, optimizer1, epoch)
    if epoch < warm_up:
        warmup_trainloader = loader.run('warmup')

        print('Warmup Net1')
        warmup(epoch, dualnet.net1, ema_net.net1, dualnet.net2, ema_net.net2, warmup_trainloader)
        scheduler1.step()

    else:
        rho = args.rho_start + (args.rho_end - args.rho_start) * linear_rampup2(epoch, args.warmup_ep)
        prob1, targets_list, true_label, confi_weight1 = eval_train(ema_net.net1, rho, 1)
        prob2, _, _, confi_weight2 = eval_train(ema_net.net2, rho, 0)
        pred1 = (prob1  == 1)
        cleanset_eval(prob1, prob2, targets_list, true_label)
        # print('Train Net1')
        total_trainloader = loader.run('warmup')
        train(epoch, dualnet.net1, ema_net.net1, dualnet.net2, ema_net.net2, total_trainloader, prob1, prob2, confi_weight1,
              confi_weight2)
        scheduler1.step()
        if epoch < args.num_epochs - args.start_expand:
            scheduler2.step()
            scheduler3.step()
    test_train(epoch, dualnet.net1, dualnet.net2, warmup_trainloader)
    test(epoch, dualnet.net1, dualnet.net2)
    test_ema(epoch, ema_net.net1, ema_net.net2)
    wandb.log({"time": time.time() - start})
    # regard the last ckpt as the best













