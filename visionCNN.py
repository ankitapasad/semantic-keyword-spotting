'''
For evaluation on auxiliary task, args.mt = 1
'''
import json
import argparse
import time
import pdb

import numpy as np

import data_io
import data_dirs as ddir
import utils

parser = argparse.ArgumentParser(description="speech module", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", type=str, default="taslp", help="dataset used: taslp (flickr30k+mscoco) or interspeech (flickr30k")
args = parser.parse_args()

print("visual input: %s" % (ddir.visionsig_fn))

caption_bow_vec1, _ = data_io.load_bow_gt(ddir.flickr8k_captions_fn, ddir.word_ids_fn_new)
if(args.dataset == 'taslp'): 
    vision_bow_vec = data_io.load_visionsig(ddir.visionsig_fn_new, sigmoid_thr=None)
elif(args.dataset == 'taslp_new'):  # my implementation of vision_nn
    vision_bow_vec = data_io.load_visionsig(ddir.visionsig_fn_combined, sigmoid_thr=None)


# keyword list
mapping, kwList = data_io.get_mapping(ddir.flickr8k_keywords_new, ddir.keywords_test) # kwList: list of 67 keywords


def test(epoch, subset):
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
        vis_multi.append(Ys)
        if(args.mt): 
            pred_multiBoW.append(predBoW)
            grt_multiBoW.append(caption_Ys2)

    vis_multi = np.concatenate(vis_multi, axis=0)
    if(args.mt):
        pred_multiBoW, grt_multiBoW = np.concatenate(pred_multiBoW, axis=0), np.concatenate(grt_multiBoW, axis=0)
        pred_multiBoW = np.concatenate((pred_multiBoW, np.zeros((pred_multiBoW.shape[0], grt_multiBoW.shape[1]-pred_multiBoW.shape[1])).astype(np.float32)), axis=1)
    eer, ap, prec10, precN = utils.get_metrics(vis_multi.T, grt_multiBoW.T)
    pcont5 = "Overall ratings (on vision GT): EER: %f, Average precision: %f, Precision@10: %f, Precision@N: %f" % (eer, ap, prec10, precN)
    print(pcont5)
    with open("vis_exact.csv","a+") as fo: 
        fo.write(args.mtType+','+str(args.alpha)+','+str(args.n_bow2)+','+str(prec10*100)+','+str(precN*100)+','+str(eer*100)+','+str(ap*100)+'\n')

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

def test():
    # testing the output of the vision model
    ids_fn_sem = ddir.sp_testSem_ids_fn
    ids_fn = ddir.sp_test_ids_fn
    with open(ids_fn, "r") as fo:
        ids = map(lambda x: x.strip('\n'), fo.readlines())
    with open(ids_fn_sem, "r") as fo:
        ids_sem = map(lambda x: x.strip('\n'), fo.readlines())
    value_bow, gtKwDict, captionsDict = data_io.get_semValues(ddir.labels_csv, ddir.keywords_test)
    count_bow = data_io.get_semCounts(ddir.counts_csv, ddir.keywords_test)
    vis_multi_sem = np.zeros([len(ids_sem),len(mapping)])
    vis_multi_exact, gt_multi = [], []
    for i in range(len(ids)):
        # caption = ' '.join(captions_dict[ids[i]])
        idx = [ids[i]]
        vis_vec = np.stack(map(lambda x: vision_bow_vec[x[4:-2]], idx), axis=0)
        caption_vec = np.stack(map(lambda x: caption_bow_vec1[x], idx), axis=0) # GT for evaluating exact match kw pred metrics
        vis_vec_mapped = vis_vec[0][mapping]
        vis_multi_exact.append(vis_vec)
        gt_multi.append(caption_vec)
        z = idx[0].split("_")
        del z[0]
        idxnew = "_".join(z)
        if(idxnew in ids_sem):
            vis_multi_sem[ids_sem.index(idxnew)] = vis_vec_mapped

    vis_multi_exact, gt_multi = np.concatenate(vis_multi_exact, axis=0), np.concatenate(gt_multi, axis=0)

    eer, ap, prec10, precN = utils.get_metrics(vis_multi_exact.T, gt_multi.T)
    pcont = "Overall ratings: EER: %f, Average precision: %f, Precision@10: %f, Precision@N: %f" % (eer, ap, prec10, precN)
    print(pcont)
    with open("vis_exact.csv","a+") as fo: 
        fo.write(str(prec10*100)+','+str(precN*100)+','+str(eer*100)+','+str(ap*100)+'\n')

    eer, ap, spearman, prec10, precN = utils.get_metrics(vis_multi_sem,value_bow,count_bow)
    pcont = "Subjective ratings: EER: %f, Average precision: %f, Precision@10: %f, Precision@N: %f, Spearman's rho: %f" % (eer, ap, prec10, precN, spearman)
    print(pcont)
    with open("vision_sem.csv","a+") as fo:
        fo.write(str(spearman)+','+str(prec10*100)+','+str(precN*100)+','+str(eer*100)+','+str(ap*100)+'\n')

def main():
    start = time.time()
    test()

    print("Time elapsed: %f seconds" % (time.time()-start))


if __name__ == "__main__":
    main()