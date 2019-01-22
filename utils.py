import os
import pickle
import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_curve, average_precision_score
from scipy.stats import spearmanr


def load_seq(label_dict, fns, n_bow):
    labels = []
    sil_start, sil_end = n_bow, n_bow+1
    for i in xrange(len(fns)):
        label_key = fns[i]
        label = []
        for word in label_dict[label_key]:
            if word < n_bow:
                label.append(word)
            else:
                label.append(n_bow-1)
        label = [sil_start]+label+[sil_end]
        labels.append(label)
    return labels


def make_mask(sents, pad, maxlen=None):
    # return padded sentence and a mask
    if maxlen is None:
        maxlen = max(map(len, sents))
    num_sents = len(sents)
    masked_sents = pad * np.ones((num_sents, maxlen), dtype=int)
    mask = np.zeros((num_sents, maxlen), dtype=int)
    for i in range(num_sents):
        masked_sents[i][0: len(sents[i])] = sents[i]
        mask[i][0: len(sents[i])] = 1
    return masked_sents, mask


def get_out(sents, pad):
    # return output seqs
    out_sents = pad * np.ones_like(sents, int)
    out_sents[:, :-1] = sents[:, 1:]
    return out_sents


def get_onehot(sents, n_vocab):
    # get onehot vec from sents
    # np.argmax(onehot, axis=-1) ==> sents
    num_sents, len_sent = sents.shape
    onehot = np.zeros((num_sents * len_sent, n_vocab), int)
    onehot[range(num_sents * len_sent), sents.reshape(-1)] = 1
    onehot = onehot.reshape((num_sents, len_sent, n_vocab))
    return onehot


def load_map(in_dir, map_names):
    # load map
    map_fullnames = map(lambda x: os.path.join(in_dir, x), map_names)
    x = []
    for i in xrange(len(map_fullnames)):
        smat = sio.loadmat(map_fullnames[i])["map"]
        map_size = smat.shape[0:2]
        smat = smat.reshape((map_size[0]*map_size[1], -1))
        x.append(smat)
    x = np.stack(x, axis=0)
    return x


def load_fc(fc_dir, fc_names):
    fc_fullnames = map(lambda x: os.path.join(fc_dir, x), fc_names)
    x = []
    for i in xrange(len(fc_fullnames)):
        smat = sio.loadmat(fc_fullnames[i])["fc"].reshape(-1)
        x.append(smat)
    x = np.stack(x, axis=0)
    return x


def load_label_dict(word_ids_fn):
    with open(word_ids_fn, "rb") as fo:
        label_dict = pickle.load(fo)
    return label_dict


def get_fscore(pred, grt):
    # pred, grt: (batch_size, n_bow)
    pred, grt = pred.astype(np.float32), grt.astype(np.float32)
    n_tp = np.sum(pred * grt)
    n_pred = np.sum(pred)
    n_true = np.sum(grt)
    precision = n_tp/n_pred
    recall = n_tp/n_true
    fscore = 2.*precision*recall/(precision + recall)
    return precision, recall, fscore

def getEER(pred,grt):
    # EER: false acceptance rate = false rejection rate
    fpr, tpr, thresholds = roc_curve(grt,pred,pos_label=1)
    fnr = 1 - tpr
    # eer_threshold = thresholds[np.nanargmin(np.absolute(fnr-fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr-fpr))]

    return eer

def getPrecx(pred,grt,x):
    # where should the pred be thresholded to select just the top x values
    thresh = np.partition(pred,len(pred)-x)[len(pred)-x]

    pred = (pred>=thresh).astype(int)
    tp = np.sum(pred*grt)
    ntot = np.sum(pred)

    return float(tp)/float(ntot)

def get_metrics(pred,grt,grtCounts): # pass 1000 (nSamples) x 67 (nClasses) arrays
    
    eer, prec10, precN, spearman = 0.0, 0.0, 0.0, 0.0
    nSamples = 0
    for i in range(pred.shape[0]):
        if(np.array_equal(grt[i],np.zeros(grt.shape[1]))): pass
        else:
            nSamples += 1
            eer += getEER(pred[i],grt[i])
            prec10 += getPrecx(pred[i],grt[i],10)
            precN += getPrecx(pred[i],grt[i],len(grt[i]))
            spearman += spearmanr(pred[i],grtCounts[i])[0]

    eer, prec10, precN, spearman = eer/nSamples, prec10/nSamples, precN/nSamples, spearman/nSamples

    # precision@10: retain the top 10 values from pred

    # precision@N: retain the top len(grt) values from pred

    # average precision: area under the ROC curve
    ap = average_precision_score(grt,pred)

    return eer, ap, spearman, prec10, precN

