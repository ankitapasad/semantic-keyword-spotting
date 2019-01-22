import os

flickr30k_dir = "/share/data/speech/Datasets/flickr30k"
flickr8k_dir = "/share/data/speech/Datasets/flickr8k"
word_ids_fn = os.path.join(flickr30k_dir, "word_ids/word_to_id_content.pkl") # also used in speech module

# files in visual module (for training visual model, FC layers)
train_dir_fn = os.path.join(flickr30k_dir, "partition/flickr30k_all_no8ktraintest.txt")
dev_dir_fn = os.path.join(flickr30k_dir, "partition/flickr30k_dev.txt")

map_dir = os.path.join(flickr30k_dir, "glm_torch")

# vision files
caption_fn = os.path.join(flickr30k_dir, "word_ids/captions_word_ids_content_dict.pkl")
vs_output = os.path.join(flickr30k_dir, "output")
vs_log_fn = os.path.join(vs_output, "log")
vs_model_fn = os.path.join(vs_output, "vs.pth")

# files in speech module
mfcc_dir = os.path.join(flickr8k_dir, "mfcc_cmvn_dd_vad")
visionsig_fn = os.path.join(flickr8k_dir, "flickr8k_vis_v3.pkl") # soft labels for images
flickr8k_captions_fn = os.path.join(flickr8k_dir, "word_ids/captions_content_dict.pkl")
flickr8k_keywords = os.path.join(flickr8k_dir, "word_ids/keywords.txt")

sp_output = os.path.join(flickr8k_dir, "output")
sp_log_fn = os.path.join(sp_output, "log")
sp_model_fn = os.path.join(sp_output, "sp.pth")
sp_map_dir = os.path.join(flickr8k_dir, "glm_torch")

human_judgement_dir = "/share/data/speech/ankitap/sss/semantic_flickraudio-master/data"
keywords_test = os.path.join(human_judgement_dir,"keywords.txt")
labels_csv = os.path.join(human_judgement_dir,"semantic_flickraudio_labels.csv")
counts_csv = os.path.join(human_judgement_dir,"semantic_flickraudio_counts.csv")

sp_train_ids_fn, sp_dev_ids_fn, sp_test_ids_fn, sp_testSem_ids_fn = os.path.join(flickr8k_dir, "word_ids/train.id"), os.path.join(flickr8k_dir, "word_ids/dev.id"), os.path.join(flickr8k_dir, "word_ids/test.id"), os.path.join(flickr8k_dir, "word_ids/testSem.id")

## flickr 8k dataset
# flickr8k/word_ids/captions_content_dict.pkl: id - keywords
# flickr8k/wprd_ids/captions_dict.pkl: id - caption
# keyword list: 
# vocabulary: 
## flickr 30k dataset
# vocabulary: flickr30k/word_ids/word_to_id.txt
# content-words for training the image tagger: flickr30k/word_ids/word_to_id_content.txt

