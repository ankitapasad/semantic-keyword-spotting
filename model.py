import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

LOG_MIN = 1.0e-7


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout())

    def forward(self, x):
        fmap = self.features(x)
        y = fmap.view(fmap.size(0), -1)
        y = self.classifier(y)
        return fmap, y


class VISDNN(nn.Module):
    def __init__(self, d_out, dropout):
        super(DNN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(4096, 3072),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(3072, 3072),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(3072, 3072),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(3072, 3072),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(3072, d_out))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.classifier(x)
        prob = self.sigmoid(x)
        return prob


class GLMDNN(nn.Module):
    def __init__(self, d_out, dropout, fsize):
        # fsize: size of feature
        super(GLMDNN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(fsize, 3072),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(3072, 3072),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(3072, 3072),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(3072, 3072),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(3072, d_out),
            nn.Sigmoid())

    def forward(self, x):
        # x: (B, L, feature_size)
        # max_idx: (B, K)
        B, L = x.size(0), x.size(1)
        x = x.view(B*L, -1)
        prob = self.classifier(x).view(B, L, -1)
        max_prob, max_idx = torch.max(prob, dim=1)
        attn_weights = prob
        return max_prob, attn_weights

    def init_weight(self, *args):
        for w in args:
            hin, hout = w.size(0), w.size(1)
            w.data.uniform_(-math.sqrt(6.0/(hin+hout)), math.sqrt(6.0/(hin+hout)))


class SpeechCNN(nn.Module):
    def __init__(self, d_out, dropout):
        # "filter_shapes": [
        #     [39, 9, 1, 64],
        #     [1, 10, 64, 256],
        #     [1, 11, 256, 1024]
        # ],
        # "pool_shapes": [
        #     [1, 3],
        #     [1, 3],
        #     [1, 75]
        # ],
        '''
        input: 800 frames
        each frame: 39-dimension features
        input -> 16 (mini_batch) x 1 x 39 x 800 
        Conv2d -> inchannels, outchannels, kernel_size
        '''
        super(SpeechCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(39, 9)), # 16 x 64 x 1 x 792
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(1, 3)), # 16 x 64 x 1 x 264
            nn.Conv2d(64, 256, kernel_size=(1, 10)), # 16 x 256 x 1 x 255
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(1, 3)), # 16 x 256 x 1 x 85
            nn.Conv2d(256, 1024, kernel_size=(1, 11)), # 16 x 1024 x 1 x 75
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(1, 75)) # 16 x 1024 x 1 x 1
        )
        self.classifierBoW = nn.Sequential(
            nn.Linear(1024, 4096), # 16 x 4096 x 1 x 1
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Linear(4096, vocab_size) # 16 x  (size of the vocbulary)
        )
        self.classifierKW = nn.Sequential(
            nn.Linear(1024, 4096), # 16 x 4096 x 1 x 1
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Linear(4096, d_out) # 16 x 67 (# keywords)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, vocab_size), # 16 x 4096 x 1 x 1
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Linear(vocab_size, 4096), # 16 x 4096 x 1 x 1
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Linear(4096, d_out) # 16 x 67 (# keywords)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 1024)
        y = self.classifier(x)
        prob = self.sigmoid(y)
        return prob


class SpeechAttnCNN(nn.Module):
    def __init__(self, d_out, dropout):
        super(SpeechAttnCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(39, 9)),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Conv2d(64, 256, kernel_size=(1, 10)),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Conv2d(256, 1024, kernel_size=(1, 11)),
            nn.ReLU(True))
        self.Wa = nn.Parameter(torch.FloatTensor(1024, 1))
        self.classifier = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Linear(4096, d_out)
        )
        self.sigmoid = nn.Sigmoid()
        self.init_weight(self.Wa)

    def forward(self, x):
        f = self.features(x)
        f = f.squeeze(dim=2).transpose(1, 2)  # (N, 1024, W)
        s = torch.matmul(f, self.Wa)
        alpha = F.softmax(s, dim=1)
        f_avg = torch.sum(alpha * f, dim=1)
        prob = self.sigmoid(self.classifier(f_avg))
        attn_weights = alpha.transpose(1, 2).unsqueeze(dim=1)  # (B, 1, 1, L)
        return prob, attn_weights

    def init_weight(self, *args):
        for w in args:
            hin, hout = w.size(0), w.size(1)
            w.data.uniform_(-math.sqrt(6.0/(hin+hout)), math.sqrt(6.0/(hin+hout)))


def loss(pred, grt, lambda_=1.0):
    # pred, grt: [B, L], [B, L]
    pred = torch.clamp(pred, min=LOG_MIN, max=1-LOG_MIN)
    l = -torch.mean(lambda_*grt*torch.log(pred)+(1-grt)*torch.log(1-pred))
    return l