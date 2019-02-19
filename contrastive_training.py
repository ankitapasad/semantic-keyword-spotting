import os
# import json
# import pickle
import argparse
import time
import copy
import pdb
from operator import itemgetter

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
parser.add_argument("--threshold", type=float, default=0.4, help="threshold for precision-recall")
parser.add_argument("--dropout", type=float, default=0.25, help="dropout rate")
parser.add_argument("--learning_rate", type=float, default=1.0e-4, help="learning rate")
parser.add_argument("--alpha", type=float, default=0.8, help="multi-task loss weighing factor")
parser.add_argument("--batch_size", type=int, default=8, help="batch size")
parser.add_argument("--test_batch_size", type=int, default=16, help="test batch size")
parser.add_argument("--epoch", type=int, default=25, help="epochs")
parser.add_argument("--n_bow", type=str, default=1000, help="number of words")
parser.add_argument("--n_pad", type=str, default=800, help="frame length to pad to")
parser.add_argument("--device", type=str, default="cuda", help="device")
parser.add_argument("--print_interval", type=int, default=200, help="interval (on iter) for printing info")
parser.add_argument("--mt", type=int, default=0, help="model with/without multi-task learning")
parser.add_argument("--mtType", type=str, default="paracnn", help="type of multitask framework")
parser.add_argument("--layer", type=int, default = 2, help="intermediate layer representation used from vision model")
parser.add_argument("--nNegEx", type=int, default=2, help="number of negative examples each pair \
	of a positive example")
parser.add_argument("--margin", type=float, default=1.0, help="margin for contrastive loss")
parser.add_argument("--distType", type=str, default="cos", help="type of distance in contrastive loss")

args = parser.parse_args()


# save files
if(args.mt):
    # saveModel = ddir.new_dir+'/contrastiveLoss/output/sp_layer-'+str(args.layer)+'_'+'/log_alpha-'+str(args.alpha)+'_'+args.mtType+'.pth'
    saveLog = ddir.new_dir+'/contrastiveLoss/output/sp_layer-'+str(args.layer)+'_'+'/log_alpha-'+str(args.alpha)+'_thresh-'+str(args.threshold)+'_'+args.mtType
else:
    # saveModel = ddir.sp_output+'/'+args.dataset+'/sp.pth'
    saveLog = ddir.new_dir+'/contrastiveLoss/output/sp_layer-'+str(args.layer)
print("log: %s" % (saveLog))
print("print info per %d iter" % (args.print_interval))
print("visual input: %s" % (ddir.visionsig_fn))
print("weighing factor: %f" % (args.alpha))


# setting model 
if(args.train):
	modelVs = model.visionFFMV(1024, args.dropout).to(args.device) # speech model intermediate output is 2014 dim
	modelSp = model.SpeechCNNMV(args.n_bow, args.dropout).to(args.device)
elif(args.test):
	print("Loading model")
	modelVs = torch.load(ddir.model_vision)
	modelSp = torch.load(ddir.model_speech)

optimizer = optim.Adam(list(modelSp.parameters()) + list(modelVs.parameters()), lr=args.learning_rate)
# optimizerVs = optim.Adam(modelVs.parameters(), lr=args.learning_rate)

criterion = model.TripletLoss(n_neg = args.nNegEx, margin = args.margin)


# reading GT
caption_bow_vec, n_vocab = data_io.load_captions(ddir.flickr8k_captions_fn, ddir.word_ids_fn_new, n_bow=-1, flag="taslp")
vision_bow_vec = data_io.load_visionsig(ddir.visionsig_fn_combined, sigmoid_thr=None)


# reading the feature vector (input to ff model 1)
repDict = sio.loadmat(os.path.join(ddir.new_dir,"interim_features/layer"+str(args.layer)+".mat"))
fnames = []
fvecs = []
for key, value in repDict.items():
	fnames.append(key)
	fvecs.append(value)


def getNegEx(train_ids_sp, idx):
	# get a filtered list of train_ids tp sample from
	train_ids_filtered = copy.deepcopy(train_ids_sp)
	for ids in idx:
		train_ids_filtered = [x for x in train_ids_filtered if '_'.join(ids.split('_')[1:-1]) not in x ]
	idxSp = list(np.random.choice(np.array(train_ids_filtered),args.batch_size*args.nNegEx))
	
	return idxSp


def getNegFeatures(train_ids_sp, idx):
	# idx: speech ids
	idxSp = getNegEx(train_ids_sp, idx) # B*N
	negSpInput, _ = data_io.load_mfcc(ddir.mfcc_dir, idxSp, args.n_pad)
	negSpInput = torch.from_numpy(np.transpose(negSpInput, (0, 2, 1))).to(args.device).unsqueeze(dim=1)
	
	idxVs = ['_'.join(ids.split('_')[1:-1]) for ids in idxSp]
	negVsInput = torch.tensor(itemgetter(*idxVs)(repDict)).squeeze(1).to(args.device) # B*N X 2048

	neg_fVectorSp = modelSp(negSpInput)[0].view(args.batch_size, args.nNegEx, -1) # B x N x 1024
	neg_fVectorVs = modelVs(negVsInput).view(args.batch_size, args.nNegEx, -1) # B x N x 1024
	return neg_fVectorSp, neg_fVectorVs


def getContrastiveLoss(train_ids_sp, idxSp, fVectorSp):
	# fvectorSp: B x 1024
	modelVs.train()
	idxVs = ['_'.join(ids.split('_')[1:-1]) for ids in idxSp] # get the corresponding vision ids
	inFVectosVs = torch.tensor(itemgetter(*idxVs)(repDict)).squeeze(1).to(args.device) # B x 2048: vision input fvectors
	fVectorVs = modelVs(inFVectosVs) # B x 1024: vision output fvectors
	neg_fVectorSp, neg_fVectorVs = getNegFeatures(train_ids_sp, idxSp) # each: B x N x 1024
	loss = criterion(fVectorSp, fVectorVs, neg_fVectorSp, neg_fVectorVs)

	return loss


def run_net(Xs, Ys, YsBoW=None):
    Xs, Ys = torch.from_numpy(Xs).to(args.device), torch.from_numpy(Ys).to(args.device)
    if(args.mt): YsBoW = torch.from_numpy(YsBoW).to(args.device)
    N, H, W = Xs.size()[0], Xs.size()[1], Xs.size()[2]
    Xs = Xs.unsqueeze(dim=1) # .view(N, 1, H, W)
    if(args.mt): fVectorSp, Ys_predBoW, Ys_pred = modelSp(Xs,modelType=args.mtType)
    else: fVectorSp, Ys_pred = modelSp(Xs)
    loss = model.loss(Ys_pred, Ys)
    if(args.mt): 
    	lossBoW = model.loss(Ys_predBoW, YsBoW)
    	return loss, lossBoW, Ys_pred.cpu().data.numpy(), Ys_predBoW.cpu().data.numpy(), fVectorSp
    else: return loss, Ys_pred.cpu().data.numpy(), fVectorSp


def getKWprob(Xs):
    Xs = torch.from_numpy(Xs).to(args.device)
    N, H, W = Xs.size()[0], Xs.size()[1], Xs.size()[2]
    Xs = Xs.unsqueeze(dim=1) # .view(N, 1, H, W)
    if(args.attn): Ys_pred, _ = modelSp(Xs)
    elif(args.mt): _, Ys_pred = modelSp(Xs,modelType=args.mtType)
    else: Ys_pred = modelSp(Xs)
    
    return Ys_pred.cpu().data.numpy()


def train(epoch):
    modelSp.train()    
    with open(ddir.sp_train_ids_fn, "r") as fo:
        train_ids = map(lambda x: x.strip(), fo.readlines())
    perm = np.random.permutation(len(train_ids)).tolist()
    # perm = range(len(train_ids)) # no permutation
    lt, pred_multi, grt_multi, pred_multiBoW, ltKW, ltBoW = [], [], [], [], [], []
    for i in range(0, len(perm), args.batch_size):
        optimizer.zero_grad()
        idx = map(lambda x: train_ids[x], perm[i: i+args.batch_size])
        train_Xs, _ = data_io.load_mfcc(ddir.mfcc_dir, idx, args.n_pad)
        train_Xs = np.transpose(train_Xs, (0, 2, 1))
        train_Ys = np.stack(map(lambda x: vision_bow_vec[x[4:-2]], idx), axis=0)
        caption_Ys = np.stack(map(lambda x: caption_bow_vec[x], idx), axis=0)
        if(args.mt): l, lBoW, pred, predBoW, fVectorSp = run_net(train_Xs, train_Ys, YsBoW=caption_Ys)
        else: l, pred, fVectorSp = run_net(train_Xs, train_Ys)
        pred_multi.append(pred)
        grt_multi.append(caption_Ys)
        if(args.mt): 
            pred_multiBoW.append(predBoW)
            lossTot = args.alpha*l + (1-args.alpha)*lBoW
            ltKW.append(l.cpu().item())
            ltBoW.append(lBoW.cpu().item())
        else: lossTot = torch.tensor(l, requires_grad=True)
        lossTot += getContrastiveLoss(train_ids, idx, fVectorSp)
        lossTot.backward()
        torch.nn.utils.clip_grad_norm(modelSp.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm(modelVs.parameters(), 5.0)
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
            precision, recall, fscore = utils.get_fscore(pred_multi >= args.threshold, grt_multi)
            # eer, ap, prec10, precN = utils.get_metrics(pred_multi, grt_multi)
            # pcont4 = "Overall ratings: EER: %f, Average precision: %f, Precision@10: %f, Precision@N: %f" % (eer, ap, prec10, precN)
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
            # print(pcont4)
            # with open(saveLog, "a+") as fo:
            #     if(args.mt): fo.write("\n"+pcont1+"\n"+pcont2+"\n"+pcont3+"\n"+pcont4+"\n")
            #     else: fo.write(pcont+"\n"+pcont4+"\n")
            with open(saveLog, "a+") as fo:
                if(args.mt): fo.write("\n"+pcont1+"\n"+pcont2+"\n"+pcont3+"\n")
                else: fo.write(pcont+"\n")
            lt, pred_multi, grt_multi, pred_multiBoW, ltKW, ltBoW = [], [], [], [], [], []
    return 0


def test(epoch, subset):
    network.eval()
    pcont4=" "
    ids_fn_sem = ddir.sp_testSem_ids_fn
    ids_fn = ddir.sp_dev_ids_fn if subset == "dev" else ddir.sp_test_ids_fn
    with open(ids_fn, "r") as fo:
        ids = map(lambda x: x.strip(), fo.readlines())
    with open(ids_fn_sem, "r") as fo:
        ids_sem = map(lambda x: x.strip(), fo.readlines())
    lt, pred_multi, grt_multi, pred_multiBoW, ltKW, ltBoW, pred_multi_kw, grt_multi_kw = [], [], [], [], [], [], [], []
    for i in range(0, len(ids), args.test_batch_size):
        idx = ids[i: i+args.test_batch_size]
        Xs, _ = data_io.load_mfcc(ddir.mfcc_dir, idx, args.n_pad)
        Xs = np.transpose(Xs, (0, 2, 1))
        Ys = np.stack(map(lambda x: vision_bow_vec[x[4:-2]], idx), axis=0)
        caption_Ys = np.stack(map(lambda x: caption_bow_vec[x], idx), axis=0)
        if(args.mt): l, lBoW, pred, predBoW, _ = run_net(Xs, Ys, caption_Ys)
        else: l, pred, _ = run_net(Xs, Ys)
        pred_multi.append(pred)
        grt_multi.append(caption_Ys)
        if(args.mt): 
            pred_multiBoW.append(predBoW)
            lossTot = args.alpha*l + (1-args.alpha)*lBoW
            ltKW.append(l.cpu().item())
            ltBoW.append(lBoW.cpu().item())
        else: lossTot = torch.tensor(l, requires_grad=True)
        lt.append(lossTot.cpu().item())
        # if('taslp' in args.dataset):
        predMapped = (pred.T[mapping]).T
        pred_multi_kw.append(predMapped)
        # pdb.set_trace()
        grtMapped = (caption_Ys.T[mapping1]).T
        grt_multi_kw.append(grtMapped)

    pred_multi, grt_multi = np.concatenate(pred_multi, axis=0), np.concatenate(grt_multi, axis=0)
    pred_multi = np.concatenate((pred_multi, np.zeros((pred_multi.shape[0], grt_multi.shape[1]-pred_multi.shape[1])).astype(np.float32)), axis=1)
    # if('taslp' in args.dataset):
    pred_multi_kw, grt_multi_kw = np.concatenate(pred_multi_kw, axis=0), np.concatenate(grt_multi_kw, axis=0)
    pred_multi_kw = np.concatenate((pred_multi_kw, np.zeros((pred_multi_kw.shape[0], grt_multi_kw.shape[1]-pred_multi_kw.shape[1])).astype(np.float32)), axis=1)
    precision1, recall1, fscore1 = utils.get_fscore(pred_multi_kw >= args.threshold, grt_multi_kw)
    if(epoch==args.epoch-1 or subset=="test"): 
        ap1 = utils.get_metrics(pred_multi, grt_multi, flag='onlyAP')
        eer, ap, prec10, precN = utils.get_metrics(pred_multi_kw, grt_multi_kw)
        pcont4 = "Overall ratings: EER: %f, Average precision: %f, Average precision on all keywords: %f, Precision@10: %f, Precision@N: %f" % (eer, ap, ap1, prec10, precN)
        with open("tempcsvap.csv","a+") as fo: 
            fo.write(args.mtType+','+str(args.alpha)+','+str(ap1*100)+','+str(ap*100)+'\n')
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
    print(pcont4)
    with open(saveLog, "a+") as fo:
        if(args.mt): fo.write("\n"+pcont1+"\n"+pcont2+"\n"+pcont3+"\n"+pcont4+"\n")
        else: fo.write(pcont+"\n"+pcont4+"\n")
    # with open(saveLog, "a+") as fo:
    #     if(args.mt): fo.write("\n"+pcont1+"\n"+pcont2+"\n"+pcont3+"\n")
    #     else: fo.write(pcont+"\n")

    # return fscore

    return fscore1


def testSem():
    network.eval()
    ids_fn_sem = ddir.sp_testSem_ids_fn
    ids_fn = ddir.sp_test_ids_fn
    with open(ids_fn, "r") as fo:
        ids = map(lambda x: x.strip('\n'), fo.readlines())
    with open(ids_fn_sem, "r") as fo:
        ids_sem = map(lambda x: x.strip('\n'), fo.readlines())

    predKwList = []
    value_bow, gtKwList = data_io.get_semValues(ddir.labels_csv, ddir.keywords_test)
    count_bow = data_io.get_semCounts(ddir.counts_csv, ddir.keywords_test)
    pred_multi = np.zeros([len(ids_sem),len(mapping)])
    for i in range(len(ids)):
        idx = [ids[i]]
        z = idx[0].split("_")
        del z[0]
        idxnew = "_".join(z)
        if(idxnew in ids_sem):
            with open(saveLog, "a+") as fo:
                fo.write(str(idxnew)+"\n")
            Xs, _ = data_io.load_mfcc(ddir.mfcc_dir, idx, args.n_pad)
            Xs = np.transpose(Xs, (0, 2, 1))
            Ys = np.stack(map(lambda x: vision_bow_vec[x[4:-2]], idx), axis=0)
            caption_Ys = np.stack(map(lambda x: caption_bow_vec[x], idx), axis=0)
            pred = getKWprob(Xs)
            predMapped = pred[0][mapping]
            pred_multi[ids_sem.index(idxnew)] = predMapped
            ## Printing results for analysis purpose
            indices = np.argwhere(predMapped>args.threshold)
            temp = ''
            for k in range(len(indices)):
                word = kwList[indices[k][0]]
                prob = predMapped[indices[k][0]]
                temp += (word + '(' + str(prob) + '), ')
            temp = temp[:-2]
            out_str2 = 'Pred (' + str(args.threshold)+ '): ' + temp +'\n'

            n = len(gtKwList[ids_sem.index(idxnew)])
            indices = np.argpartition(predMapped,-n)[-n:]
            temp = ''
            for k in range(len(indices)):
                prob = predMapped[indices[k]]
                word = kwList[indices[k]]
                temp += (word + '(' + str(prob) + '), ')
            temp = temp[:-2]
            out_str3 = 'Pred (top n): ' + temp + '\n'

            temp = ''
            for word in gtKwList[ids_sem.index(idxnew)]:
                prob = predMapped[kwList.index(word)]
                temp += (word + '(' + str(prob) + '), ')
            temp = temp[:-2]
            out_str4 = 'GT probabilities: ' + temp + '\n'

            out_str1 = 'GT: ' + ', '.join(gtKwList[ids_sem.index(idxnew)]) +'\n'
            with open(saveLog, "a+") as fo:
                fo.write(out_str1+out_str2+out_str3+out_str4+"\n")
            predKwList = []
            # pdb.set_trace()
    eer, ap, spearman, prec10, precN = utils.get_metrics(pred_multi,value_bow,count_bow)
    pcont = "Subjective ratings: EER: %f, Average precision: %f, Precision@10: %f, Precision@N: %f, Spearman's rho: %f" % (eer, ap, prec10, precN, spearman)
    print(pcont)
    with open("tempcsv.csv","a+") as fo:
        fo.write(args.mtType+','+str(args.alpha)+','+str(100*spearman)+','+str(100*prec10)+','+str(100*precN)+','+str(100*eer)+','+str(100*ap)+'\n')
    with open(saveLog, "a+") as fo:
        fo.write(pcont+"\n")


def apply_train():
    best_score = -1
    for epoch in range(args.epoch):
        train(epoch)
        score = test(epoch, "dev") # could be fscore or average precision
        if score > best_score:
            torch.save(modelSp, ddir.model_speech)
            torch.save(modelVs, ddir.model_vision)
            best_score = score
    return 0


def apply_test():
    with open(saveLog, "a+") as fo:
        fo.write('************************************ TEST STARTS HERE ******************************'+'\n')
    test(0, "test")
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