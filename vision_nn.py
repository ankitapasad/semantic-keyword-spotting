import os
import model
import torch
import pickle
import argparse
import data_dirs as ddir
import torch.optim as optim
import numpy as np
import scipy.io as sio
import time
from torch.autograd import Variable
from utils import load_map, load_fc, load_label_dict, get_fscore
import pdb

parser = argparse.ArgumentParser(description="vision network", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--train", action="store_true", help="train the model")
parser.add_argument("--test", action="store_true", help="test the model")
parser.add_argument("--test_prob", type=str, default=ddir.visionsig_fn, help="dir to save prob")
parser.add_argument("--test_map_dir", type=str, default=ddir.sp_map_dir,help="dir to load feature map")
parser.add_argument("--test_attn_dir", type=str, help="dir to save attn map")
parser.add_argument("--model", type=str, default=ddir.vs_model_fn, help="model path")
parser.add_argument("--n_bow", type=str, default=1000, help="number of words")
parser.add_argument("--threshold", type=float, default=0.5, help="threshold for precision-recall")
parser.add_argument("--dropout", type=float, default=0.25, help="dropout rate")
parser.add_argument("--learning_rate", type=float, default=1.0e-4, help="learning rate")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--test_batch_size", type=int, default=16, help="test batch size")
parser.add_argument("--epoch", type=int, default=10, help="epochs")
parser.add_argument("--feature_size", type=int, default=4096, help="feature dimension in map")
parser.add_argument("--device", type=str, default="cuda", help="device")
parser.add_argument("--print_interval", type=int, default=200, help="interval (on iter) for printing info")
parser.add_argument("--attn", type=bool, default=False, help="model with/without attention")
parser.add_argument("--data", type=str, default="taslp", help="interspeech: flickr30k, taslp: flickr30k+coco")
# parser.add_argument("--save_interval", type=int, default=10, help="interval (on epoch) for saving model")
args = parser.parse_args()

fvec_dir = ddir.map_dir_fvec
args.test_map_dir = ddir.sp_map_dir_fvec

if(args.data == "taslp"):
    args.test_prob = ddir.visionsig_fn_combined
    args.model = ddir.vs_model_fn_combined
    vs_log_fn = ddir.vs_log_fn_combined
    vs_output = ddir.vs_output_combined
    caption_fn = ddir.combined_caption_id_fn
    train_dir_fn = ddir.combined_train_dir_fn
    dev_dir_fn = ddir.combined_dev_dir_fn
    hidden_size = 2048
else:
    args.test_prob = ddir.temp
    # args.test_map_dir = ddir.temp
    vs_log_fn = ddir.vs_log_fn
    vs_output = ddir.vs_output
    caption_fn = ddir.caption_fn
    train_dir_fn = ddir.train_dir_fn
    dev_dir_fn = ddir.dev_dir_fn
    hidden_size = 3072
print("log: %s, model dir: %s" % (vs_log_fn, vs_output))
print("print info per %d iter" % (args.print_interval))

if args.test:
    print("Load model: %s" % (args.model))
    mlp = torch.load(args.model)
elif(args.attn):
    mlp = model.GLMDNN(args.n_bow, args.dropout, args.feature_size)
else:
    mlp = model.VISDNN(args.n_bow, hidden_size, args.dropout) # hidden_size: 2048 (2019), 3072 (2017)


mlp.to(args.device)
optimizer = optim.Adam(mlp.parameters(), lr=args.learning_rate)


def run_net(label_dict, map_dir, fns):
    fcs = []
    bow_vectors = np.zeros((len(fns), args.n_bow), dtype=np.float32)
    for i_data in xrange(len(fns)):
        fn = fns[i_data]
        for i_caption in range(5):
            label_key = "{}_{}".format(fn, i_caption)
            # filter uncommon words
            filtered_words = [w for w in label_dict[label_key] if w < args.n_bow]
            for i_word in filtered_words:
                bow_vectors[i_data, i_word] = 1
    fc_fullnames = map(lambda s: s+".mat", fns)
    if(args.attn): fcs = load_map(map_dir, fc_fullnames)
    else: fcs = load_fc(map_dir, fc_fullnames)
    fcs, bow_vectors = torch.from_numpy(fcs).to(args.device), torch.from_numpy(bow_vectors).to(args.device)
    # pdb.set_trace()
    if(args.attn): logits, max_idx = mlp(fcs) # 16 x 49 x 4096
    else: logits = mlp(fcs) # 32 x 4096
    loss = model.loss(logits, bow_vectors)
    
    return loss, logits.cpu().data.numpy(), bow_vectors.cpu().data.numpy()


def train(epoch, dir_fn, label_dict, in_dir):
    mlp.train()
    with open(dir_fn, "r") as fo:
        imnames = map(lambda x: x.strip(), fo.readlines())
    perm = np.random.permutation(len(imnames)).tolist()
    lt, pred_multi, grt_multi = [], [], []
    for i in range(0, len(perm), args.batch_size):
        optimizer.zero_grad()
        fns = imnames[i: i+args.batch_size]
        l, pred, grt = run_net(label_dict, in_dir, fns)
        pred_multi.append(pred)
        grt_multi.append(grt)
        l.backward()
        torch.nn.utils.clip_grad_norm(mlp.parameters(), 5.0)
        optimizer.step()
        lt.append(l.cpu().item())
        if len(lt) % args.print_interval == 0:
            # save model
            pred_multi, grt_multi = np.concatenate(pred_multi, axis=0), np.concatenate(grt_multi, axis=0)
            precision, recall, fscore = get_fscore(pred_multi >= args.threshold, grt_multi)
            pcont = "epoch: %d, train loss: %.3f, precision: %.3f, recall: %.3f, fscore: %.3f" % (epoch, sum(lt)/len(lt), precision, recall, fscore)
            print(pcont)
            with open(vs_log_fn, "a+") as fo:
                fo.write(pcont+"\n")
            lt, pred_multi, grt_multi = [], [], []
    return 0


def test(epoch, dir_fn, label_dict, in_dir):
    mlp.eval()
    with open(dir_fn, "r") as fo:
        imnames = map(lambda x: x.strip(), fo.readlines())
    lt, pred_multi, grt_multi = [], [], []
    for i in range(0, len(imnames), args.test_batch_size):
        fns = imnames[i: i+args.test_batch_size]
        l, pred, grt = run_net(label_dict, in_dir, fns)
        lt.append(l.cpu().item())
        pred_multi.append(pred)
        grt_multi.append(grt)
    pred_multi, grt_multi = np.concatenate(pred_multi, axis=0), np.concatenate(grt_multi, axis=0)
    precision, recall, fscore = get_fscore(pred_multi >= args.threshold, grt_multi)
    pcont = "epoch: %d, dev loss: %.3f, precision: %.3f, recall: %.3f, fscore: %.3f" % (epoch, sum(lt)/len(lt), precision, recall, fscore)
    print(pcont)
    with open(vs_log_fn, "a+") as fo:
        fo.write(pcont+"\n")
    return fscore


def apply_train():
    label_dict = load_label_dict(caption_fn)
    best_fscore = -1
    for epoch in range(args.epoch):
        train(epoch, train_dir_fn, label_dict, fvec_dir)
        fscore = test(epoch, dev_dir_fn, label_dict, fvec_dir)
        if fscore > best_fscore:
            torch.save(mlp, args.model) 
            best_fscore = fscore
    return 0


def apply_test():
    mlp.eval()
    if(args.attn):
        fc_dir, write_fn, attn_dir = args.test_map_dir, args.test_prob, args.test_attn_dir
    else: fc_dir, write_fn = args.test_map_dir, args.test_prob
    fc_fns = os.listdir(fc_dir)
    prob_dict = {}
    for i in range(0, len(fc_fns), args.test_batch_size):
        fns_i = fc_fns[i: i+args.test_batch_size]
        ffns = map(lambda x: os.path.join(fc_dir, x), fns_i)
        if(args.attn): fcs = load_map(fc_dir, fns_i)
        else: fcs = load_fc(fc_dir, fns_i)
        fcs = torch.from_numpy(fcs).to(args.device)
        if(args.attn): 
            probs, max_idx = mlp(fcs)
            max_id = max_idx.cpu().data.numpy()
        else: probs = mlp(fcs)
        probs = probs.cpu().data.numpy()
        if args.attn and attn_dir is not None:
            # save attn map
            for j in xrange(len(fns_i)):
                attn_fullfn = os.path.join(attn_dir, fns_i[j])
                # maxid = max_idx[j]
                # sio.savemat(attn_fullfn, {"MAXID": maxid})
                attn_weights = np.transpose(max_idx[j], (1, 0))
                sio.savemat(attn_fullfn, {"weight": attn_weights})
        for j in xrange(len(fns_i)):
            key_name = fns_i[j][:-4]
            prob_dict[key_name] = probs[j]
    if write_fn is not None:
        with open(write_fn, "wb") as fo:
            pickle.dump(prob_dict, fo)
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
