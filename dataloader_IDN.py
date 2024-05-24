from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from numpy.testing import assert_array_almost_equal
from torchnet.meter import AUCMeter
from randaugment import RandAugmentMC
import torch.nn.functional as F

import pandas as pd

def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class cifar_dataset(Dataset):
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_file='', pred=[], probability=[],
                 log='', q_pre=[], q_label=[]):

        self.r = r  # noise ratio
        self.transform = transform
        self.mode = mode
        idx_each_class_noisy = [[] for i in range(10)]

        if self.mode == 'test':
            if dataset == 'cifar10':
                test_dic = unpickle('%s/test_batch' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['labels']
            elif dataset == 'cifar100':
                test_dic = unpickle('%s/test' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['fine_labels']
        else:
            train_data = []
            train_label = []
            if dataset == 'cifar10':
                for n in range(1, 6):
                    dpath = '%s/data_batch_%d' % (root_dir, n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label + data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset == 'cifar100':
                train_dic = unpickle('%s/train' % root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))


            self.train_data = train_data
            self.clean_label = np.array(train_label)
            self.noise_label = list(pd.read_csv('/workspace/data/label_noisy/dependent'+str(self.r)+'.csv')['label_noisy'].values.astype(int))



    def __getitem__(self, index):
        if self.mode == 'labeled':
            img, target, clean = self.train_data[index], self.noise_label[index], self.clean_label[index]
            img = Image.fromarray(img)
            img1 = self.transform(img)
            return img1, target, clean, index
        elif self.mode == 'unlabeled':
            img, clean = self.train_data[index], self.clean_label[index]
            img = Image.fromarray(img)
            weak = self.transform[0](img)
            strong = self.transform[1](img)
            return self.transform[2](weak), self.transform[2](strong), index, clean
        elif self.mode == 'all':
            img, target, clean = self.train_data[index], self.noise_label[index], self.clean_label[index]
            img = Image.fromarray(img)
            weak = self.transform[0](img)
            strong = self.transform[1](img)
            return self.transform[2](weak), self.transform[2](strong), target, index, clean
        elif self.mode == 'test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)

    def multiclass_noisify(self, y, P, random_state=0):
        """ Flip classes according to transition probability matrix T.
        It expects a number between 0 and the number of classes - 1.
        """

        assert P.shape[0] == P.shape[1]
        assert np.max(y) < P.shape[0]

        # row stochastic matrix
        assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
        assert (P >= 0.0).all()

        m = y.shape[0]
        new_y = y.copy()
        flipper = np.random.RandomState(random_state)

        for idx in np.arange(m):
            i = y[idx]
            # draw a vector with only an 1
            flipped = flipper.multinomial(1, P[i, :], 1)[0]
            new_y[idx] = np.where(flipped == 1)[0]

        return new_y

    def build_for_cifar100(self, size, noise):
        """ random flip between two random classes.
        """
        assert (noise >= 0.) and (noise <= 1.)

        P = np.eye(size)
        cls1, cls2 = np.random.choice(range(size), size=2, replace=False)
        P[cls1, cls2] = noise
        P[cls2, cls1] = noise
        P[cls1, cls1] = 1.0 - noise
        P[cls2, cls2] = 1.0 - noise

        assert_array_almost_equal(P.sum(axis=1), 1, 1)
        return P

    def asymmetric_noise(self, train_label, num_classes=100):
        P = np.eye(num_classes)
        n = self.r
        nb_superclasses = 20
        nb_subclasses = 5

        if n > 0.0:
            for i in np.arange(nb_superclasses):
                init, end = i * nb_subclasses, (i + 1) * nb_subclasses
                P[init:end, init:end] = self.build_for_cifar100(nb_subclasses, n)

            y_train_noisy = self.multiclass_noisify(train_label, P=P,
                                                    random_state=0)
            actual_noise = (y_train_noisy != train_label).mean()
            assert actual_noise > 0.0
            return y_train_noisy

    def instance_noise(self):

        noise_label = torch.load('/workspace/data/noise_label/IDN_{:.1f}_C100.pt'.format(self.r))

        noisylabel = noise_label['noise_label_train']

        self.noise_label = np.array(noisylabel)

    def noisify_instance(self, train_data, train_labels, noise_rate):
        if max(train_labels) > 10:
            num_class = 100
        else:
            num_class = 10
        np.random.seed(0)

        q_ = np.random.normal(loc=noise_rate, scale=0.1, size=1000000)
        q = []
        for pro in q_:
            if 0 < pro < 1:
                q.append(pro)
            if len(q) == 50000:
                break

        w = np.random.normal(loc=0, scale=1, size=(32 * 32 * 3, num_class))

        noisy_labels = []
        for i, sample in enumerate(train_data):
            sample = sample.flatten()
            p_all = np.matmul(sample, w)
            p_all[train_labels[i]] = -1000000
            p_all = q[i] * F.softmax(torch.tensor(p_all), dim=0).numpy()
            p_all[train_labels[i]] = 1 - q[i]
            noisy_labels.append(np.random.choice(np.arange(num_class), p=p_all / sum(p_all)).tolist())
        over_all_noise_rate = 1 - float(torch.tensor(train_labels).eq(torch.tensor(noisy_labels)).sum()) / 50000
        return noisy_labels, over_all_noise_rate


class cifar_dataloader():
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file
        self.cifar10_mean = (0.4914, 0.4822, 0.4465)
        self.cifar10_std = (0.2471, 0.2435, 0.2616)
        self.cifar100_mean = (0.507, 0.487, 0.441)
        self.cifar100_std = (0.267, 0.256, 0.276)
        if self.dataset == 'cifar10':
            self.transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32,
                                      padding=int(32 * 0.125),
                                      padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            ])
            self.weak = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32,
                                      padding=int(32 * 0.125),
                                      padding_mode='reflect')])
            self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32,
                                      padding=int(32 * 0.125),
                                      padding_mode='reflect'),
                RandAugmentMC(n=2, m=10)])
            self.normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.cifar10_mean, std=self.cifar10_std)])

        elif self.dataset == 'cifar100':
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])
            self.weak = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32,
                                      padding=int(32 * 0.125),
                                      padding_mode='reflect')])
            self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32,
                                      padding=int(32 * 0.125),
                                      padding_mode='reflect'),
                RandAugmentMC(n=2, m=10)])
            self.normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.cifar100_mean, std=self.cifar100_std)])

    def run(self, mode, pred=[], prob=[], q_pre=[], q_label=[]):
        if mode == 'warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                        root_dir=self.root_dir, transform=[self.weak, self.strong, self.normalize],
                                        mode="all", noise_file=self.noise_file)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            return trainloader

        elif mode == 'train':
            labeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                            root_dir=self.root_dir, \
                                            transform=self.transform_train, mode="labeled", \
                                            noise_file=self.noise_file, pred=pred, probability=prob, log=self.log,
                                            q_pre=q_pre, q_label=q_label)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True)

            unlabeled_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                              root_dir=self.root_dir, \
                                              transform=[self.weak, self.strong, self.normalize], mode="unlabeled", \
                                              noise_file=self.noise_file, pred=pred)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size * 7,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True)
            return labeled_trainloader, unlabeled_trainloader

        elif mode == 'test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                         root_dir=self.root_dir, transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return test_loader

        elif mode == 'eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r,
                                         root_dir=self.root_dir, transform=self.transform_test, mode='all',
                                         noise_file=self.noise_file)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            return eval_loader
