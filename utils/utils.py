from torch.utils.data import Dataset
from PIL import Image
import os
import json
class SceneData(Dataset):
    def __init__(self, txt_file, image_dir, mode, transform=None, transform_label=None):
        with open(txt_file, 'r') as f:
            datasets = f.readlines()
        self.datasets = datasets
        self.image_dir = image_dir
        self.transform = transform
        self.transform_label = transform_label
        self.mode = mode
        samples = []
        for data in datasets:
            data = data.rstrip().split(" ")
            path = os.path.join(self.image_dir, self.mode, data[0])
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                continue
            sample = (data[0], int(data[1]))
            samples.append(sample)
        self.samples = samples

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.image_dir, self.mode, self.samples[index][0]))
        image = image.convert('RGB')
        label = self.samples[index][1]
        if self.transform is not None:
            image = self.transform(image)
        if self.transform_label is not None:
            label = self.transform_label(label)
        return image, label
    
    def __len__(self):
        return len(self.samples)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


import numpy as np 

def mixup(X, y, mixup_alpha=0.1):
    '''
        功能：图像增强，mixup
        参数：
            X：batch imgs
            y: batch labels
        超参：
            beta: beta分布的alpha和beta参数，这个可以自己设置，并观察结果
        引用：
            mixup: Beyond Empirical Risk Minimization(https://arxiv.org/abs/1710.09412)
    '''
    seed = np.random.beta(mixup_alpha, mixup_alpha)
    index = np.arange(X.shape[0])
    np.random.shuffle(index)
    images_a, images_b = X, X[index]
    labels_a, labels_b = y, y[index]
    mixed_images = seed * images_a + (1 - seed) * images_b
    return mixed_images, labels_a, labels_b, seed

