import os
import json
import pickle
import argparse
import time
import copy
import pdb

import numpy as np
import scipy.io as sio

import model
import data_io
import data_dirs as ddir
import utils

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="speech module", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--train", action="store_true", help="train the model")
parser.add_argument("--test", action="store_true", help="test the model")
parser.add_argument("--model", type=str, help="model path")
parser.add_argument("--attn_dir", type=str, help="attn dir")
parser.add_argument("--threshold", type=float, default=0.4, help="threshold for precision-recall")
parser.add_argument("--dropout", type=float, default=0.25, help="dropout rate")
parser.add_argument("--learning_rate", type=float, default=1.0e-4, help="learning rate")
parser.add_argument("--alpha", type=float, default=0.8, help="multi-task loss weighing factor")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--test_batch_size", type=int, default=16, help="test batch size")
parser.add_argument("--epoch", type=int, default=60, help="epochs")
parser.add_argument("--n_bow", type=str, default=1000, help="number of words")
parser.add_argument("--n_pad", type=str, default=800, help="frame length to pad to")
parser.add_argument("--device", type=str, default="cuda", help="device")
parser.add_argument("--print_interval", type=int, default=200, help="interval (on iter) for printing info")
parser.add_argument("--attn", type=bool, default=False, help="model with/without attention")
parser.add_argument("--mt", type=int, default=1, help="model with/without multi-task learning")
parser.add_argument("--mtType", type=str, default="paracnn", help="type of multitask framework")
args = parser.parse_args()

if(args.mt):
    saveModel = ddir.sp_model_fn.split('.')[0]+'_alpha-'+str(args.alpha)+'_thresh-'+str(args.threshold)+'_'+args.mtType+'.pth'
    saveLog = ddir.sp_log_fn + '_alpha-'+str(args.alpha)+'_thresh-'+str(args.threshold)+'_'+args.mtType
else:
    saveModel = ddir.sp_model_fn.split('.')[0]+'_thresh-'+str(args.threshold)+'.pth'
    saveLog = ddir.sp_log_fn +'_thresh-'+str(args.threshold)
print("log: %s, model dir: %s" % (saveLog, saveModel))
print("print info per %d iter" % (args.print_interval))
print("visual input: %s" % (ddir.visionsig_fn))
print("weighing factor: %f" % (args.alpha))

caption_sentence_dict = pickle.load(open(ddir.flickr8k_captions_fn, "rb"))

# data
caption_bow_vec, n_vocab = data_io.load_captions(ddir.flickr8k_captions_fn, ddir.word_ids_fn, n_bow=-1)
vision_bow_vec = data_io.load_visionsig(ddir.visionsig_fn, sigmoid_thr=None)

# network
if args.test:
    print("Load model: %s" % (saveModel))
    network = torch.load(saveModel)
elif(args.attn): network = model.SpeechAttnCNN(args.n_bow, dropout=args.dropout).to(args.device)
elif(args.mt): network = model.SpeechCNNMT(args.n_bow, n_vocab, args.dropout, modelType=args.mtType).to(args.device)
else: network = model.SpeechCNN(args.n_bow, args.dropout).to(args.device)

optimizer = optim.Adam(network.parameters(), lr=args.learning_rate)


def run_net(Xs, Ys, YsBoW=None):
    Xs, Ys = torch.from_numpy(Xs).to(args.device), torch.from_numpy(Ys).to(args.device)
    if(args.mt): YsBoW = torch.from_numpy(YsBoW).to(args.device)
    N, H, W = Xs.size()[0], Xs.size()[1], Xs.size()[2]
    Xs = Xs.unsqueeze(dim=1) # .view(N, 1, H, W)
    if(args.attn): Ys_pred, attn_weights = network(Xs)
    elif(args.mt): Ys_predBoW, Ys_pred = network(Xs)
    else: Ys_pred = network(Xs)
    loss = model.loss(Ys_pred, Ys)
    # pdb.set_trace()
    if(args.mt): lossBoW = model.loss(Ys_predBoW, YsBoW)
    if(args.attn): return loss, Ys_pred.cpu().data.numpy(), attn_weights
    elif(args.mt): return loss, lossBoW, Ys_pred.cpu().data.numpy(), Ys_predBoW.cpu().data.numpy()
    else: return loss, Ys_pred.cpu().data.numpy()

def getKWprob(Xs):
    Xs = torch.from_numpy(Xs).to(args.device)
    N, H, W = Xs.size()[0], Xs.size()[1], Xs.size()[2]
    Xs = Xs.unsqueeze(dim=1) # .view(N, 1, H, W)
    if(args.attn): Ys_pred, _ = network(Xs)
    elif(args.mt): _, Ys_pred = network(Xs)
    else: Ys_pred = network(Xs)
    
    return Ys_pred.cpu().data.numpy()


def train(epoch):
    network.train()
    with open(ddir.sp_train_ids_fn, "r") as fo:
        train_ids = map(lambda x: x.strip(), fo.readlines())
    perm = np.random.permutation(len(train_ids)).tolist()
    perm = range(len(train_ids))
    lt, pred_multi, grt_multi, pred_multiBoW, ltKW, ltBoW = [], [], [], [], [], []
    for i in range(0, len(perm), args.batch_size):
        optimizer.zero_grad()
        idx = map(lambda x: train_ids[x], perm[i: i+args.batch_size])
        train_Xs, _ = data_io.load_mfcc(ddir.mfcc_dir, idx, args.n_pad)
        train_Xs = np.transpose(train_Xs, (0, 2, 1))
        train_Ys = np.stack(map(lambda x: vision_bow_vec[x[4:-2]], idx), axis=0)
        caption_Ys = np.stack(map(lambda x: caption_bow_vec[x], idx), axis=0)
        if(args.attn): l, pred, _ = run_net(train_Xs, train_Ys)
        elif(args.mt): l, lBoW, pred, predBoW = run_net(train_Xs, train_Ys, YsBoW=caption_Ys)
        else: l, pred = run_net(train_Xs, train_Ys)
        pred_multi.append(pred)
        grt_multi.append(caption_Ys)
        if(args.mt): 
            pred_multiBoW.append(predBoW)
            lossTot = args.alpha*l + (1-args.alpha)*lBoW
            ltKW.append(l.cpu().item())
            ltBoW.append(lBoW.cpu().item())
        else: lossTot = torch.tensor(l, requires_grad=True)
        lossTot.backward()
        torch.nn.utils.clip_grad_norm(network.parameters(), 5.0)
        optimizer.step()
        lt.append(lossTot.cpu().item())
        if len(lt) % args.print_interval == 0:
            # save model
            pred_multi, grt_multi = np.concatenate(pred_multi, axis=0), np.concatenate(grt_multi, axis=0)
            pred_multi = np.concatenate((pred_multi, np.zeros((pred_multi.shape[0], grt_multi.shape[1]-pred_multi.shape[1])).astype(np.float32)), axis=1)
            if(args.mt): 
                pred_multiBoW = np.concatenate(pred_multiBoW, axis=0)
                pred_multiBoW = np.concatenate((pred_multiBoW, np.zeros((pred_multiBoW.shape[0], grt_multi.shape[1]-pred_multiBoW.shape[1])).astype(np.float32)), axis=1)
            # pred_multi, grt_multi = np.concatenate(pred_multi, axis=0), np.concatenate(grt_multi, axis=0)
            # pred_multi = np.concatenate((pred_multi, np.zeros((pred_multi.shape[0], grt_multi.shape[1]-pred_multi.shape[1])).astype(np.float32)), axis=1)
            precision, recall, fscore = get_fscore(pred_multi >= args.threshold, grt_multi)
            if(args.mt): 
                precisionBoW, recallBoW, fscoreBoW = utils.get_fscore(pred_multiBoW >= args.threshold, grt_multi)
                pcont1 = "epoch: %d, train loss: %.3f" % (epoch, sum(lt)/len(lt))
                pcont2 = "train loss KW: %.3f, precision KW: %.3f, recall KW: %.3f, fscore KW: %.3f" % (sum(ltKW)/len(ltKW), precision, recall, fscore)
                pcont3 = "train loss BoW: %.3f, precision BoW: %.3f, recall BoW: %.3f, fscore BoW: %.3f" % (sum(ltBoW)/len(ltBoW), precisionBoW, recallBoW, fscoreBoW)
                print(pcont1)
                print(pcont2)
                print(pcont3)
            else: 
                pcont = "epoch: %d, train loss: %.3f, precision: %.3f, recall: %.3f, fscore: %.3f" % (epoch, sum(lt)/len(lt), precision, recall, fscore)
                print(pcont)
            with open(saveLog, "a+") as fo:
                if(args.mt): fo.write("\n"+pcont1+"\n"+pcont2+"\n"+pcont3+"\n")
                else: fo.write(pcont+"\n")
            lt, pred_multi, grt_multi, pred_multiBoW, ltKW, ltBoW = [], [], [], [], [], []
    return 0


def test(epoch, subset):
    network.eval()
    ids_fn_sem = ddir.sp_testSem_ids_fn
    ids_fn = ddir.sp_dev_ids_fn if subset == "dev" else ddir.sp_test_ids_fn
    with open(ids_fn, "r") as fo:
        ids = map(lambda x: x.strip(), fo.readlines())
    with open(ids_fn_sem, "r") as fo:
        ids_sem = map(lambda x: x.strip(), fo.readlines())
    lt, pred_multi, grt_multi, pred_multiBoW, ltKW, ltBoW = [], [], [], [], [], []
    for i in range(0, len(ids), args.test_batch_size):
        idx = ids[i: i+args.test_batch_size]
        Xs, _ = data_io.load_mfcc(ddir.mfcc_dir, idx, args.n_pad)
        Xs = np.transpose(Xs, (0, 2, 1))
        Ys = np.stack(map(lambda x: vision_bow_vec[x[4:-2]], idx), axis=0)
        caption_Ys = np.stack(map(lambda x: caption_bow_vec[x], idx), axis=0)
        if(args.attn): l, pred, _ = run_net(Xs, Ys)
        elif(args.mt): l, lBoW, pred, predBoW = run_net(Xs, Ys, caption_Ys)
        else: l, pred = run_net(Xs, Ys)
        pred_multi.append(pred)
        grt_multi.append(caption_Ys)
        if(args.mt): 
            pred_multiBoW.append(predBoW)
            lossTot = args.alpha*l + (1-args.alpha)*lBoW
            ltKW.append(l.cpu().item())
            ltBoW.append(lBoW.cpu().item())
        else: lossTot = torch.tensor(l, requires_grad=True)
        lt.append(lossTot.cpu().item())
    pred_multi, grt_multi = np.concatenate(pred_multi, axis=0), np.concatenate(grt_multi, axis=0)
    pred_multi = np.concatenate((pred_multi, np.zeros((pred_multi.shape[0], grt_multi.shape[1]-pred_multi.shape[1])).astype(np.float32)), axis=1)
    if(args.mt):
        pred_multiBoW = np.concatenate(pred_multiBoW, axis=0)
        pred_multiBoW = np.concatenate((pred_multiBoW, np.zeros((pred_multiBoW.shape[0], grt_multi.shape[1]-pred_multiBoW.shape[1])).astype(np.float32)), axis=1)
    precision, recall, fscore = utils.get_fscore(pred_multi >= args.threshold, grt_multi)
    if(args.mt): 
        precisionBoW, recallBoW, fscoreBoW = utils.get_fscore(pred_multiBoW >= args.threshold, grt_multi)
        pcont1 = "epoch: %d, dev loss: %.3f" % (epoch, sum(lt)/len(lt))
        pcont2 = "dev loss KW: %.3f, precision KW: %.3f, recall KW: %.3f, fscore KW: %.3f" % (sum(ltKW)/len(ltKW), precision, recall, fscore)
        pcont3 = "dev loss BoW: %.3f, precision BoW: %.3f, recall BoW: %.3f, fscore BoW: %.3f" % (sum(ltBoW)/len(ltBoW), precisionBoW, recallBoW, fscoreBoW)
        print(pcont1)
        print(pcont2)
        print(pcont3)
    else: 
        pcont = "epoch: %d, dev loss: %.3f, precision: %.3f, recall: %.3f, fscore: %.3f" % (epoch, sum(lt)/len(lt), precision, recall, fscore)
        print(pcont)
    with open(saveLog, "a+") as fo:
        if(args.mt): fo.write("\n"+pcont1+"\n"+pcont2+"\n"+pcont3+"\n")
        else: fo.write(pcont+"\n")
    return fscore

def testSem():
    network.eval()
    ids_fn_sem = ddir.sp_testSem_ids_fn
    ids_fn = ddir.sp_test_ids_fn
    with open(ids_fn, "r") as fo:
        ids = map(lambda x: x.strip('\n'), fo.readlines())
    with open(ids_fn_sem, "r") as fo:
        ids_sem = map(lambda x: x.strip('\n'), fo.readlines())

    mapping = data_io.get_mapping(ddir.flickr8k_keywords, ddir.keywords_test)
    value_bow = data_io.get_semValues(ddir.labels_csv, ddir.keywords_test)
    count_bow = data_io.get_semCounts(ddir.counts_csv, ddir.keywords_test)
    pred_multi = np.zeros([len(ids_sem),len(mapping)])
    for i in range(len(ids)):
        idx = [ids[i]]
        z = idx[0].split("_")
        del z[0]
        idxnew = "_".join(z)
        if(idxnew in ids_sem):
            Xs, _ = data_io.load_mfcc(ddir.mfcc_dir, idx, args.n_pad)
            Xs = np.transpose(Xs, (0, 2, 1))
            Ys = np.stack(map(lambda x: vision_bow_vec[x[4:-2]], idx), axis=0)
            caption_Ys = np.stack(map(lambda x: caption_bow_vec[x], idx), axis=0)
            pred = getKWprob(Xs)
            predMapped = pred[0][mapping]
            pred_multi[ids_sem.index(idxnew)] = predMapped
    eer, ap, spearman, prec10, precN = utils.get_metrics(pred_multi,value_bow,count_bow)
    pcont = "Subjective ratings: EER: %f, Average precision: %f, Precision@10: %f, Precision@N: %f, Spearman's rho: %f" % (eer, ap, prec10, precN, spearman)
    print(pcont)
    with open(saveLog, "a+") as fo:
        fo.write(pcont+"\n")


def get_attn_weights(subset):
    network.eval()
    if subset == "dev":
        ids_fn = ddir.sp_dev_ids_fn
    else:
        ids_fn = ddir.sp_train_ids_fn
    # ids_fn = dev_ids_fn if subset == "dev" else test_ids_fn
    with open(ids_fn, "r") as fo:
        ids = map(lambda x: x.strip(), fo.readlines())
    for i in range(0, len(ids), 1):
        # fns = imnames[i: i+test_batch_size]
        idx = ids[i: i+1]
        Xs, lengths = data_io.load_mfcc(ddir.mfcc_dir, idx, args.n_pad)
        Xs = np.transpose(Xs, (0, 2, 1))
        Ys = np.stack(map(lambda x: vision_bow_vec[x[4:-2]], idx), axis=0)
        _, _, attn_weights = run_net(Xs, Ys)
        attn_weights = F.upsample(attn_weights, size=(1, Xs.shape[2]), mode="bilinear").cpu().data.numpy().reshape((1, -1))
        attn_name = os.path.join(args.attn_dir, idx[0])
        lb = int(args.n_pad/2-lengths[0]/2)
        rb = lb+int(lengths[0])
        unpadded_attn_weights = attn_weights[:, lb: rb]/(attn_weights[:, lb: rb].sum())
        sio.savemat(attn_name, {"weight": unpadded_attn_weights})
    return 0

def getScore():

    score = 0

    return score


def apply_train():
    best_fscore = -1
    for epoch in range(args.epoch):
        train(epoch)
        fscore = test(epoch, "dev")
        if fscore > best_fscore:
            torch.save(network, saveModel)
            best_fscore = fscore
    return 0


def apply_test():
    # test(0, "test")
    testSem()
    return 0


def main():
    start = time.time()
    if args.train:
        apply_train()
    if args.test:
        apply_test()

    print("Time elapsed: %f seconds" % (time.time()-start))


if __name__ == "__main__":
    main()
