'''
For evaluation on auxiliary task, args.mt = 1
'''
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
parser.add_argument("--batch_size", type=int, default=8, help="batch size")
parser.add_argument("--test_batch_size", type=int, default=16, help="test batch size")
parser.add_argument("--epoch", type=int, default=25, help="epochs")
parser.add_argument("--n_bow1", type=int, default=1000, help="number of words")
parser.add_argument("--n_bow2", type=int, default=1000, help="number of words for the auxiliary task")
parser.add_argument("--n_pad", type=str, default=800, help="frame length to pad to")
parser.add_argument("--device", type=str, default="cuda", help="device")
parser.add_argument("--print_interval", type=int, default=200, help="interval (on iter) for printing info")
parser.add_argument("--attn", type=bool, default=False, help="model with/without attention")
parser.add_argument("--mt", type=int, default=1, help="model with/without multi-task learning")
parser.add_argument("--mtType", type=str, default="paracnn", help="type of multitask framework")
parser.add_argument("--dataset", type=str, default="taslp", help="dataset used: taslp (flickr30k+mscoco) or interspeech (flickr30k")
parser.add_argument("--nLayerSeries", type=int, default=0, help="number of layers for BoW in series architecture")
parser.add_argument("--verbose", type=int, default=0, help="save the keyword predictions")
parser.add_argument("--iter", type=int, default=1, help="(temporary) iter number to evaluate the std dev")
parser.add_argument("--earlystop", type=int, default=0, help="strict early stopping or not")
parser.add_argument("--lrs", type=int, default=0, help="learning rate scheduling")
args = parser.parse_args()

if(args.mt and args.nLayerSeries!=0):
    saveModel = ddir.sp_output+'/'+args.dataset+'/sp_alpha-'+str(args.alpha)+'_'+args.mtType+'_'+str(args.nLayerSeries)+'.pth'
    saveLog = ddir.sp_output +'/'+args.dataset+'/log_alpha-'+str(args.alpha)+'_thresh-'+str(args.threshold)+'_'+args.mtType+'_'+str(args.nLayerSeries)
elif(args.mt and args.nLayerSeries==0):
    saveModel = ddir.sp_output+'/'+args.dataset+'/sp_alpha-'+str(args.alpha)+'_'+args.mtType+'_nBoW-'+str(args.n_bow2)+'.pth'
    saveLog = ddir.sp_output +'/'+args.dataset+'/log_alpha-'+str(args.alpha)+'_thresh-'+str(args.threshold)+'_'+args.mtType+'_nBoW-'+str(args.n_bow2)+'_aux'
    saveLog1 = ddir.sp_output +'/'+args.dataset+'/kwords_alpha-'+str(args.alpha)+'_thresh-'+str(args.threshold)+'_'+args.mtType+'_nBoW-'+str(args.n_bow2)
    saveCsv = ddir.sp_output +'/'+args.dataset+'/kwords_alpha-'+str(args.alpha)+'_thresh-'+str(args.threshold)+'_'+args.mtType+'_nBoW-'+str(args.n_bow2)+'.csv'
else:
    saveModel = ddir.sp_output+'/'+args.dataset+'/sp_iter-'+str(args.iter)+'_earlyStop-'+str(args.earlystop)+'_lrs-'+str(args.lrs)+'.pth'
    saveLog = ddir.sp_output +'/'+args.dataset+'/log_thresh-'+str(args.threshold)+'_iter-'+str(args.iter)+'_earlyStop-'+str(args.earlystop)+'_lrs-'+str(args.lrs)
    saveLog1 = ddir.sp_output +'/'+args.dataset+'/kwords_thresh-'+str(args.threshold)+'_lrs-'+str(args.lrs)
    saveCsv = ddir.sp_output +'/'+args.dataset+'/kwords_thresh-'+str(args.threshold)+'_lrs-'+str(args.lrs)+'.csv'
print("log: %s, model dir: %s" % (saveLog, saveModel))
print("print info per %d iter" % (args.print_interval))
print("visual input: %s" % (ddir.visionsig_fn))
print("weighing factor: %f" % (args.alpha))

caption_sentence_dict = pickle.load(open(ddir.flickr8k_captions_fn, "rb"))

# data
if(args.dataset == 'interspeech_new'): # my implementation of vision_nn
    caption_bow_vec, n_vocab = data_io.load_captions(ddir.flickr8k_captions_fn, ddir.word_ids_fn, n_bow=-1, flag=args.dataset)
    vision_bow_vec = data_io.load_visionsig(ddir.temp, sigmoid_thr=None)
else: 
    caption_bow_vec, n_vocab = data_io.load_captions(ddir.flickr8k_captions_fn, ddir.word_ids_fn, n_bow=-1, flag=args.dataset)
    vision_bow_vec = data_io.load_visionsig(ddir.visionsig_fn, sigmoid_thr=None)

if('taslp' in args.dataset):
    caption_bow_vec1, caption_bow_vec2 = data_io.load_bow_gt(ddir.flickr8k_captions_fn, ddir.word_ids_fn_new, n_bow1=args.n_bow1, n_bow2=args.n_bow2)
    if(args.dataset == 'taslp'): 
        vision_bow_vec = data_io.load_visionsig(ddir.visionsig_fn_new, sigmoid_thr=None)
    elif(args.dataset == 'taslp_new'):  # my implementation of vision_nn
        vision_bow_vec = data_io.load_visionsig(ddir.visionsig_fn_combined, sigmoid_thr=None)


# keyword list
if('taslp' in args.dataset):
    mapping, kwList = data_io.get_mapping(ddir.flickr8k_keywords_new, ddir.keywords_test) # kwList: list of 67 keywords
    mapping1, _ = data_io.get_mapping(ddir.caption_words, ddir.keywords_test) # kwList: list of 67 keywords
else:
    mapping, kwList = data_io.get_mapping(ddir.flickr8k_keywords, ddir.keywords_test) # kwList: list of 67 keywords
    mapping1, _ = data_io.get_mapping(ddir.caption_words_flickr30k, ddir.keywords_test) # kwList: list of 67 keywords
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
    # elif(args.mt): _, Ys_pred = network(Xs, modelType=args.mtType)
    elif(args.mt): 
        Ys_predBoW, Ys_pred = network(Xs)
        return Ys_pred.cpu().data.numpy(), Ys_predBoW.cpu().data.numpy()
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
            precision, recall, fscore = utils.get_fscore(pred_multi >= args.threshold, grt_multi)
            # eer, ap, prec10, precN = utils.get_metrics(pred_multi, grt_multi)
            # pcont4 = "Overall ratings: EER: %f, Average precision: %f, Precision@10: %f, Precision@N: %f" % (eer, ap, prec10, precN)
            # if(args.mt): 
            #     precisionBoW, recallBoW, fscoreBoW = utils.get_fscore(pred_multiBoW >= args.threshold, grt_multi)
            #     pcont1 = "epoch: %d, train loss: %.3f" % (epoch, sum(lt)/len(lt))
            #     pcont2 = "train loss KW: %.3f, precision KW: %.3f, recall KW: %.3f, fscore KW: %.3f" % (sum(ltKW)/len(ltKW), precision, recall, fscore)
            #     pcont3 = "train loss BoW: %.3f, precision BoW: %.3f, recall BoW: %.3f, fscore BoW: %.3f" % (sum(ltBoW)/len(ltBoW), precisionBoW, recallBoW, fscoreBoW)
            #     print(pcont1)
            #     print(pcont2)
            #     print(pcont3)
            # else: 
            pcont = "epoch: %d, train loss: %.3f, precision: %.3f, recall: %.3f, fscore: %.3f" % (epoch, sum(lt)/len(lt), precision, recall, fscore)
            print(pcont)
            # print(pcont4)
            # with open(saveLog, "a+") as fo:
            #     if(args.mt): fo.write("\n"+pcont1+"\n"+pcont2+"\n"+pcont3+"\n"+pcont4+"\n")
            #     else: fo.write(pcont+"\n"+pcont4+"\n")
            with open(saveLog, "a+") as fo:
                # if(args.mt): fo.write("\n"+pcont1+"\n"+pcont2+"\n"+pcont3+"\n")
                # else: 
                fo.write(pcont+"\n")
            lt, pred_multi, grt_multi, pred_multiBoW, ltKW, ltBoW = [], [], [], [], [], []
    return 0


def test(epoch, subset):
    network.eval()
    pcont4=" "
    ids_fn = ddir.sp_dev_ids_fn if subset == "dev" else ddir.sp_test_ids_fn
    with open(ids_fn, "r") as fo:
        ids = map(lambda x: x.strip(), fo.readlines())
    pred_multi, grt_multi, pred_multiBoW, grt_multiBoW, vis_multi,  = [], [], [], [], [], [], []
    for i in range(0, len(ids), args.test_batch_size):
        idx = ids[i: i+args.test_batch_size]
        Xs, _ = data_io.load_mfcc(ddir.mfcc_dir, idx, args.n_pad)
        Xs = np.transpose(Xs, (0, 2, 1))
        vision_Ys = np.stack(map(lambda x: vision_bow_vec[x[4:-2]], idx), axis=0) # GT from vision model
        caption_Ys1 = np.stack(map(lambda x: caption_bow_vec1[x], idx), axis=0) # GT for evaluating exact match kw pred metrics
        caption_Ys2 = np.stack(map(lambda x: caption_bow_vec2[x], idx), axis=0) # GT for bow loss
        if(args.mt): l, lBoW, pred, predBoW = run_net(Xs, vision_Ys, caption_Ys2)
        else: l, pred = run_net(Xs, vision_Ys)
        pred_multi.append(pred)
        grt_multi.append(caption_Ys1)
        if(args.mt): 
            pred_multiBoW.append(predBoW)
            grt_multiBoW.append(caption_Ys2)

    if(args.mt):
        pred_multiBoW, grt_multiBoW = np.concatenate(pred_multiBoW, axis=0), np.concatenate(grt_multiBoW, axis=0)
        pred_multiBoW = np.concatenate((pred_multiBoW, np.zeros((pred_multiBoW.shape[0], grt_multiBoW.shape[1]-pred_multiBoW.shape[1])).astype(np.float32)), axis=1)

    if(subset=='test'): # On keyword spotting
        # precisionBoW, recallBoW, fscoreBoW = utils.get_fscore(pred_multiBoW >= args.threshold, grt_multiBoW)
        # pcont3 = "Threshold = %.1f: precision BoW: %.3f, recall BoW: %.3f, fscore BoW: %.3f" % (args.threshold, precisionBoW, recallBoW, fscoreBoW)
        eer, ap, prec10, precN = utils.get_metrics(pred_multiBoW.T, grt_multiBoW.T)
        pcont5 = "Overall ratings (on BoW): EER: %f, Average precision: %f, Precision@10: %f, Precision@N: %f" % (eer, ap, prec10, precN)
        with open("aux_exact.csv","a+") as fo: 
            fo.write(args.mtType+','+str(args.alpha)+','+str(args.n_bow2)+','+str(prec10*100)+','+str(precN*100)+','+str(eer*100)+','+str(ap*100)+'\n')
        # print(pcont3+"\n")
        print(pcont5+"\n")
        # with open(saveLog, "a+") as fo:
            # fo.write("\n"+pcont5+"\n")

    return 0
'''
    if('interspeech' in args.dataset): return fscore
    else: return fscore1
'''

def testSem():
    network.eval()
    ids_fn_sem = ddir.sp_testSem_ids_fn
    ids_fn = ddir.sp_test_ids_fn
    with open(ids_fn, "r") as fo:
        ids = map(lambda x: x.strip('\n'), fo.readlines())
    with open(ids_fn_sem, "r") as fo:
        ids_sem = map(lambda x: x.strip('\n'), fo.readlines())
    # with open(os.path.join(ddir.flickr8k_dir, "word_ids/captions_dict.pkl"),'rb') as f:
    #     captions_dict = pkl.load(f)
    predKwList = []
    value_bow, gtKwDict, captionsDict = data_io.get_semValues(ddir.labels_csv, ddir.keywords_test)
    count_bow = data_io.get_semCounts(ddir.counts_csv, ddir.keywords_test)
    pred_multi, pred_multiBoW = np.zeros([len(ids_sem),len(mapping)]), np.zeros([len(ids_sem),len(mapping)])
    vis_multi = np.zeros([len(ids_sem),len(mapping)])
    for i in range(len(ids)):
        # caption = ' '.join(captions_dict[ids[i]])
        idx = [ids[i]]
        z = idx[0].split("_")
        del z[0]
        idxnew = "_".join(z)
        if(idxnew in ids_sem):
            Xs, _ = data_io.load_mfcc(ddir.mfcc_dir, idx, args.n_pad)
            Xs = np.transpose(Xs, (0, 2, 1))
            Ys = np.stack(map(lambda x: vision_bow_vec[x[4:-2]], idx), axis=0)
            visMapped = Ys[0][mapping]
            if(args.mt): 
                pred, predBoW = getKWprob(Xs)
                predBoWMapped = predBoW[0][mapping]
            else: pred = getKWprob(Xs)
            predMapped = pred[0][mapping]
            pred_multi[ids_sem.index(idxnew)] = predMapped
            pred_multiBoW[ids_sem.index(idxnew)] = predBoWMapped
            vis_multi[ids_sem.index(idxnew)] = visMapped

    eer, ap, spearman, prec10, precN = utils.get_metrics(pred_multiBoW,value_bow,count_bow)
    pcont = "Subjective ratings: EER: %f, Average precision: %f, Precision@10: %f, Precision@N: %f, Spearman's rho: %f" % (eer, ap, prec10, precN, spearman)
    print(pcont)
    with open("aux_sem.csv","a+") as fo:
        fo.write(args.mtType+','+str(args.alpha)+','+str(args.n_bow2)+','+str(spearman)+','+str(prec10*100)+','+str(precN*100)+','+str(eer*100)+','+str(ap*100)+'\n')

def apply_test():
    with open(saveLog, "a+") as fo:
        fo.write('************************************ TEST STARTS HERE ******************************'+'\n')
    test(0, "test")
    testSem()
    return 0


def main():
    start = time.time()
    if args.train:
        best_epoch = apply_train()
        print1 = "Best epoch: " + str(best_epoch) + "\n"
        print2 = "Time elapsed: " +str(time.time()-start) + " seconds \n"
        with open(saveLog, "a+") as fo:
            fo.write(print1+print2)
        print(print1)
        print(print2)

    if args.test:
        apply_test()

        print("Time elapsed: %f seconds" % (time.time()-start))


if __name__ == "__main__":
    main()
