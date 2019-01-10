import os
import model
import torch
import argparse
import data_io
import json
import pickle
import data_dirs as ddir
import torch.optim as optim
import numpy as np
import scipy.io as sio
import time
from torch.autograd import Variable
import torch.nn.functional as F
from utils import load_map, load_fc, load_label_dict, get_fscore

parser = argparse.ArgumentParser(description="speech module", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--train", action="store_true", help="train the model")
parser.add_argument("--test", action="store_true", help="test the model")
parser.add_argument("--model", type=str, help="model path")
parser.add_argument("--attn_dir", type=str, help="attn dir")
parser.add_argument("--threshold", type=float, default=0.4, help="threshold for precision-recall")
parser.add_argument("--dropout", type=float, default=0.25, help="dropout rate")
parser.add_argument("--learning_rate", type=float, default=1.0e-4, help="learning rate")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--test_batch_size", type=int, default=16, help="test batch size")
parser.add_argument("--epoch", type=int, default=60, help="epochs")
parser.add_argument("--n_bow", type=str, default=1000, help="number of words")
parser.add_argument("--n_pad", type=str, default=800, help="frame length to pad to")
parser.add_argument("--device", type=str, default="cuda", help="device")
parser.add_argument("--print_interval", type=int, default=200, help="interval (on iter) for printing info")
args = parser.parse_args()

print("log: %s, model dir: %s" % (ddir.sp_log_fn, ddir.sp_model_fn))
print("print info per %d iter" % (args.print_interval))
print("visual input: %s" % (ddir.visionsig_fn))

caption_sentence_dict = pickle.load(open(ddir.flickr8k_captions_fn, "rb"))

# data
caption_bow_vec, vision_bow_vec = data_io.load_captions(ddir.flickr8k_captions_fn, ddir.word_ids_fn, n_bow=-1), data_io.load_visionsig(ddir.visionsig_fn, sigmoid_thr=None)

# network
if args.model is not None:
    print("Load model: %s" % (args.model))
    network = torch.load(args.model)
else:
    # network = model.SpeechCNN(args.n_bow, args.dropout).to(args.device)
    network = model.SpeechAttnCNN(args.n_bow, dropout=args.dropout).to(args.device)

optimizer = optim.Adam(network.parameters(), lr=args.learning_rate)


def run_net(Xs, Ys):
    Xs, Ys = torch.from_numpy(Xs).to(args.device), torch.from_numpy(Ys).to(args.device)
    N, H, W = Xs.size()[0], Xs.size()[1], Xs.size()[2]
    Xs = Xs.unsqueeze(dim=1) # .view(N, 1, H, W)
    Ys_pred, attn_weights = network(Xs)
    loss = model.loss(Ys_pred, Ys)
    return loss, Ys_pred.cpu().data.numpy(), attn_weights


def train(epoch):
    network.train()
    with open(ddir.sp_train_ids_fn, "r") as fo:
        train_ids = map(lambda x: x.strip(), fo.readlines())
    perm = np.random.permutation(len(train_ids)).tolist()
    perm = range(len(train_ids))
    lt, pred_multi, grt_multi = [], [], []
    for i in range(0, len(perm), args.batch_size):
        optimizer.zero_grad()
        idx = map(lambda x: train_ids[x], perm[i: i+args.batch_size])
        train_Xs, _ = data_io.load_mfcc(ddir.mfcc_dir, idx, args.n_pad)
        train_Xs = np.transpose(train_Xs, (0, 2, 1))
        train_Ys = np.stack(map(lambda x: vision_bow_vec[x[4:-2]], idx), axis=0)
        caption_Ys = np.stack(map(lambda x: caption_bow_vec[x], idx), axis=0)
        l, pred, _ = run_net(train_Xs, train_Ys)
        pred_multi.append(pred)
        grt_multi.append(caption_Ys)
        l.backward()
        torch.nn.utils.clip_grad_norm(network.parameters(), 5.0)
        optimizer.step()
        lt.append(l.cpu().item())
        if len(lt) % args.print_interval == 0:
            # save model
            pred_multi, grt_multi = np.concatenate(pred_multi, axis=0), np.concatenate(grt_multi, axis=0)
            pred_multi = np.concatenate((pred_multi, np.zeros((pred_multi.shape[0], grt_multi.shape[1]-pred_multi.shape[1])).astype(np.float32)), axis=1)
            precision, recall, fscore = get_fscore(pred_multi >= args.threshold, grt_multi)
            pcont = "epoch: %d, train loss: %.3f, precision: %.3f, recall: %.3f, fscore: %.3f" % (epoch, sum(lt)/len(lt), precision, recall, fscore)
            print(pcont)
            with open(ddir.sp_log_fn, "a+") as fo:
                fo.write(pcont+"\n")
            lt, pred_multi, grt_multi = [], [], []
    return 0


def test(epoch, subset):
    network.eval()
    ids_fn = ddir.sp_dev_ids_fn if subset == "dev" else ddir.sp_test_ids_fn
    with open(ids_fn, "r") as fo:
        ids = map(lambda x: x.strip(), fo.readlines())
    lt, pred_multi, grt_multi = [], [], []
    for i in range(0, len(ids), args.test_batch_size):
        idx = ids[i: i+args.test_batch_size]
        Xs, _ = data_io.load_mfcc(ddir.mfcc_dir, idx, args.n_pad)
        Xs = np.transpose(Xs, (0, 2, 1))
        Ys = np.stack(map(lambda x: vision_bow_vec[x[4:-2]], idx), axis=0)
        caption_Ys = np.stack(map(lambda x: caption_bow_vec[x], idx), axis=0)
        l, pred, _ = run_net(Xs, Ys)
        lt.append(l.cpu().item())
        pred_multi.append(pred)
        grt_multi.append(caption_Ys)
    pred_multi, grt_multi = np.concatenate(pred_multi, axis=0), np.concatenate(grt_multi, axis=0)
    pred_multi = np.concatenate((pred_multi, np.zeros((pred_multi.shape[0], grt_multi.shape[1]-pred_multi.shape[1])).astype(np.float32)), axis=1)
    precision, recall, fscore = get_fscore(pred_multi >= args.threshold, grt_multi)
    pcont = "epoch: %d, %s loss: %.3f, precision: %.3f, recall: %.3f, fscore: %.3f" % (epoch, subset, sum(lt)/len(lt), precision, recall, fscore)
    print(pcont)
    with open(ddir.sp_log_fn, "a+") as fo:
        fo.write(pcont+"\n")
    return fscore


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


def apply_train():
    best_fscore = -1
    for epoch in range(args.epoch):
        train(epoch)
        fscore = test(epoch, "dev")
        if fscore > best_fscore:
            torch.save(network, ddir.sp_model_fn)
            best_fscore = fscore
    return 0


def apply_test():
    test(0, "test")
    return 0


def main():
    if args.train:
        apply_train()
    if args.test:
        apply_test()


if __name__ == "__main__":
    main()
