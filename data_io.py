"""
Functions for dealing with data input and output.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2017
"""

from os import path
import numpy as np
import scipy.io as sio
import struct
import pickle

NP_DTYPE = np.float32
NP_ITYPE = np.int32


def read_kaldi_ark_from_scp(scp_fn, ark_base_dir=""):
    """
    Read a binary Kaldi archive and return a dict of Numpy matrices, with the
    utterance IDs of the SCP as keys. Based on the code:
    https://github.com/yajiemiao/pdnn/blob/master/io_func/kaldi_feat.py

    Parameters
    ----------
    ark_base_dir : str
        The base directory for the archives to which the SCP points.
    """

    ark_dict = {}

    with open(scp_fn) as f:
        for line in f:
            if line == "":
                continue
            utt_id, path_pos = line.replace("\n", "").split(" ")
            ark_path, pos = path_pos.split(":")

            ark_path = path.join(ark_base_dir, ark_path)

            ark_read_buffer = open(ark_path, "rb")
            ark_read_buffer.seek(int(pos),0)
            header = struct.unpack("<xcccc", ark_read_buffer.read(5))
            assert header[0] == "B", "Input .ark file is not binary"

            rows = 0
            cols = 0
            m, rows = struct.unpack("<bi", ark_read_buffer.read(5))
            n, cols = struct.unpack("<bi", ark_read_buffer.read(5))

            tmp_mat = np.frombuffer(ark_read_buffer.read(rows*cols*4), dtype=np.float32)
            utt_mat = np.reshape(tmp_mat, (rows, cols))

            ark_read_buffer.close()

            ark_dict[utt_id] = utt_mat

    return ark_dict


def pad_sequences(x, n_padded, center_padded=True):
    """Return the padded sequences and their original lengths."""
    padded_x = np.zeros((len(x), n_padded, x[0].shape[1]), dtype=NP_DTYPE)
    lengths = []
    for i_data, cur_x in enumerate(x):
        length = cur_x.shape[0]
        if center_padded:
            padding = int(np.round((n_padded - length) / 2.))
            if length <= n_padded:
                padded_x[i_data, padding:padding + length, :] = cur_x
            else:
                # Cut out snippet from sequence exceeding n_padded
                padded_x[i_data, :, :] = cur_x[-padding:-padding + n_padded]
            lengths.append(min(length, n_padded))
        else:
            length = min(length, n_padded)
            padded_x[i_data, :length, :] = cur_x[:length, :]
            lengths.append(length)
    return padded_x, lengths


def load_mfcc(mfcc_dir, idx, n_padded, center_padded=True):
    mfccs = []
    for id_ in idx:
        mfcc_name = path.join(mfcc_dir, id_+".mat")
        mfcc = sio.loadmat(mfcc_name)["mfcc"]
        mfccs.append(mfcc)
    padded_mfcc, lengths = pad_sequences(mfccs, n_padded, center_padded)
    return padded_mfcc, np.array(lengths, NP_DTYPE)


def load_visionsig(visionsig_fn, sigmoid_thr):
    with open(visionsig_fn, "rb") as fo:
        s = pickle.load(fo)
    # s = np.load(visionsig_fn)
    for k in s.keys():
        if sigmoid_thr is None:
            s[k] = s[k].astype(NP_DTYPE)
        else:
            s[k] = (s[k] >= sigmoid_thr).astype(NP_DTYPE)
    return s


def load_captions(captions_fn, word_ids_fn, n_bow):
    with open(word_ids_fn, "rb") as fo:
        word_ids_dict = pickle.load(fo)
    with open(captions_fn, "rb") as fo:
        caption_dict = pickle.load(fo)
    bow_vector_dict = {}
    if n_bow == -1:
        # update word_ids_dict
        for name, caption in caption_dict.items():
            for w in caption:
                if w not in word_ids_dict:
                    word_ids_dict[w] = len(word_ids_dict)+1
        n_bow = len(word_ids_dict)
    for name, caption in caption_dict.items():
        vec = np.zeros(n_bow, dtype=NP_DTYPE)
        # word_ids = map(lambda x: word_ids_dict[x], caption)
        for w in caption:
            if w in word_ids_dict:
                wid = word_ids_dict[w]
                if wid < n_bow:
                    vec[wid] = 1.0
        bow_vector_dict[name] = vec
    return bow_vector_dict, n_bow


def load_flickr8k_padded_bow_labelled(data_dir, subset, n_padded, label_dict,
                                      word_to_id, center_padded=True):
    """
    Return the padded Flickr8k speech matrices and bag-of-word label vectors.

    Only words in `word_to_id` is labelled. The bag-of-word vectors contain 1
    for a word that occurs and 0 for a word that does not occur.
    """

    assert subset in ["train", "dev", "test"]

    # Load data and shuffle
    npz_fn = path.join(data_dir, subset + ".npz")
    print "Reading: " + npz_fn
    features_dict = np.load(npz_fn)
    utterances = sorted(features_dict.keys())
    np.random.shuffle(utterances)
    x = [features_dict[i] for i in utterances]

    # Get lengths and pad
    padded_x, lengths = pad_sequences(x, n_padded, center_padded)
    # Get bag-of-word vectors
    bow_vectors = np.zeros((len(x), len(word_to_id)), dtype=NP_DTYPE)
    for i_data, utt in enumerate(utterances):
        for word in label_dict[utt]:
            if word in word_to_id:
                bow_vectors[i_data, word_to_id[word]] = 1
    return padded_x, bow_vectors, np.array(lengths, dtype=NP_DTYPE)


def load_flickr8k_padded_visionsig(speech_data_dir, subset, n_padded,
                                   visionsig_dict, d_visionsig, sigmoid_threshold=None,
                                   center_padded=True, tmp=None):
    """
    Return the padded Flickr8k speech matrices and vision sigmoid vectors.
    d_visionsig: n_most_common
    """

    assert subset in ["train", "dev", "test"]

    # Load data and shuffle
    npz_fn = path.join(speech_data_dir, subset + ".npz")
    print "Reading: " + npz_fn
    features_dict = np.load(npz_fn)
    utterances = sorted(features_dict.keys())
    np.random.shuffle(utterances)
    x = [features_dict[i] for i in utterances]

    # Get lengths and pad
    padded_x, lengths = pad_sequences(x, n_padded, center_padded)
    # print padded_x.shape
    # Get vision sigmoids
    visionsig_vectors = np.zeros((len(x), d_visionsig), dtype=NP_DTYPE)
    for i_data, utt in enumerate(utterances):
        image_key = utt[4:-2]
        if sigmoid_threshold is None:
            visionsig_vectors[i_data, :] = visionsig_dict[image_key][:d_visionsig]
        else:
            visionsig_vectors[i_data, np.where(visionsig_dict[image_key][:d_visionsig] >= \
                sigmoid_threshold)[0]] = 1
    return padded_x, visionsig_vectors, np.array(lengths, dtype=NP_DTYPE)

def get_semValues(data_dir, keywords):

    with open(keywords, "r") as fo:
        kw = map(lambda x: x.strip('\n'), fo.readlines())
    count = -1
    with open(data_dir) as csv_file:
        lines = csv_file.readlines()
    value_bow = np.zeros([len(lines)-1, len(kw)])
    for line in lines:
        line = line.strip("\n")
        if(count==-1): count += 1
        else: 
            keywords = (line.split(',')[2]).split('|')
            if(len(keywords)==1): pass 
            else:
                for word in keywords:
                    word = word.strip('\"')
                    value_bow[count][kw.index(word)] = 1
                count += 1

    return value_bow


def get_semCounts(data_dir, keywords):

    with open(keywords, "r") as fo:
        kw = map(lambda x: x.strip('\n'), fo.readlines())
    count = -1
    with open(data_dir) as csv_file:
        lines = csv_file.readlines()
    count_bow = np.zeros([len(lines)-1, len(kw)])
    for line in lines:
        line = line.strip("\n")
        if(count==-1): count += 1
        else: 
            keywords = (line.split(',')[2]).split('|')
            if(len(keywords)==1): pass 
            else:
                for word in keywords:
                    word = word.strip('\"')
                    count_bow[count][kw.index(word.split("=")[0])] = int(word.split("=")[1])
                count += 1

    return count_bow

def get_mapping(keywordsAll, keywordsSem):
    # get a mapping of indices which will help conver the 1000-dim probability array to 67-dim array
    sem2all = []
    with open(keywordsAll, "r") as fo:
        listAll = map(lambda x: x.strip('\n'), fo.readlines())
    with open(keywordsSem, "r") as fo:
        listSem = map(lambda x: x.strip('\n'), fo.readlines())
    for word in listSem:
        sem2all.append(listAll.index(word))

    return sem2all
