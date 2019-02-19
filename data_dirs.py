import os

flickr30k_dir = "/share/data/speech/Datasets/flickr30k"
flickr8k_dir = "/share/data/speech/Datasets/flickr8k"
coco_dir = "/share/data/speech/Datasets/MSCOCO"
new_dir1 = "/share/data/speech/ankitap/sss/herman/models/mscoco+flickr30k/train_bow_mlp/891a3a3533"
new_dir = "/share/data/speech/ankitap/sss/taslp-data"
word_ids_fn = os.path.join(flickr30k_dir, "word_ids/word_to_id_content.pkl") # also used in speech module
word_ids_fn_all = os.path.join(flickr30k_dir, "word_ids/word_to_id.pkl") # also used in speech module
word_ids_fn_new = os.path.join(new_dir, "word_to_id.pkl")
npz_files_dir = new_dir+"/npz_dir"

# files in visual module (for training visual model, FC layers)
train_dir_fn = os.path.join(flickr30k_dir, "partition/flickr30k_all_no8ktraintest.txt")
dev_dir_fn = os.path.join(flickr30k_dir, "partition/flickr30k_dev.txt")
train_dir_fn_coco = os.path.join(new_dir, "data/mscoco/train.txt")
dev_dir_fn_coco = os.path.join(new_dir, "data/mscoco/dev.txt")
map_dir = os.path.join(flickr30k_dir, "glm_torch")

## combined files
### common
combined_data_dir = os.path.join(new_dir, "data/coco+flickr30k")
combined_train_dir_fn = os.path.join(combined_data_dir, "train.txt")
combined_dev_dir_fn = os.path.join(combined_data_dir, "dev.txt")
map_dir_fvec = os.path.join(npz_files_dir, "coco+flickr30k")
### vision
combined_caption_id_fn = os.path.join(combined_data_dir, "captions_word_ids_content.pkl")
vs_model_fn_combined = os.path.join(new_dir, "vs.pth")
vs_output_combined = os.path.join(new_dir, "output")
vs_log_fn_combined = os.path.join(vs_output_combined, "log")
vs_model_fn_combined = os.path.join(vs_output_combined, "vs.pth")
### speech
sp_map_dir_fvec = os.path.join(npz_files_dir, "flickr8k")
visionsig_fn_combined = os.path.join(new_dir, "flickr8k_vis_v3.pkl") # soft labels for images, 8000 files
temp = os.path.join(new_dir, "test.pkl")

# vision files
caption_fn = os.path.join(flickr30k_dir, "word_ids/captions_word_ids_content_dict.pkl")
caption_fn_all = os.path.join(flickr30k_dir, "word_ids/captions_word_ids_dict.pkl")
caption_fn_words = os.path.join(flickr30k_dir, "word_ids/captions.pkl")
caption_fn_content_id = os.path.join(flickr30k_dir, "word_ids/captions_word_ids_content_taslp.pkl")

vs_output = os.path.join(flickr30k_dir, "output")
vs_log_fn = os.path.join(vs_output, "log")
vs_model_fn = os.path.join(vs_output, "vs.pth")

coco_annotation_dir = os.path.join(coco_dir, "annotation_files")
coco_data_dir = os.path.join(coco_dir, "images")
coco_processed_dir = os.path.join(new_dir,"data/mscoco")

npz_flickr30k = os.path.join(npz_files_dir, "flickr30k")
npz_flickr8k = os.path.join(npz_files_dir, "flickr8k")
npz_train_coco = os.path.join(npz_files_dir, "coco/train")
npz_dev_coco = os.path.join(npz_files_dir, "coco/dev")
npz_test_coco = os.path.join(npz_files_dir, "coco/test")

save_fvec_dir = os.path.join(new_dir, "interim_features")

# files in speech module
mfcc_dir = os.path.join(flickr8k_dir, "mfcc_cmvn_dd_vad")
visionsig_fn = os.path.join(flickr8k_dir, "flickr8k_vis_v3.pkl") # soft labels for images, 8091 files
visionsig_fn_new = os.path.join(new_dir, "safegaurding/flickr8k_vis_v3.pkl") # soft labels for images, 8000 files

flickr8k_captions_fn = os.path.join(flickr8k_dir, "word_ids/captions_content_dict.pkl")
flickr8k_captions_id_fn = os.path.join(flickr8k_dir, "word_ids/captions_content_id_dict.pkl")
flickr8k_keywords = os.path.join(flickr8k_dir, "word_ids/keywords.txt")
flickr8k_keywords_new = os.path.join(new_dir,"keywords.txt")
caption_words = os.path.join(new_dir,"extended_vocab.txt") # flickr30k+mscoco
caption_words_flickr30k = os.path.join(flickr30k_dir,"word_ids/extended_vocab.txt") # only flickr30k

sp_output = os.path.join(flickr8k_dir, "output")
sp_log_fn = os.path.join(sp_output, "log")
sp_model_fn = os.path.join(sp_output, "sp.pth")
sp_map_dir = os.path.join(flickr8k_dir, "glm_torch")

human_judgement_dir = "/share/data/speech/ankitap/sss/semantic_flickraudio-master/data"
keywords_test = os.path.join(human_judgement_dir,"keywords.txt")
labels_csv = os.path.join(human_judgement_dir,"semantic_flickraudio_labels.csv")
counts_csv = os.path.join(human_judgement_dir,"semantic_flickraudio_counts.csv")

sp_train_ids_fn, sp_dev_ids_fn, sp_test_ids_fn, sp_testSem_ids_fn = os.path.join(flickr8k_dir, "word_ids/train.id"), os.path.join(flickr8k_dir, "word_ids/dev.id"), os.path.join(flickr8k_dir, "word_ids/test.id"), os.path.join(flickr8k_dir, "word_ids/testSem.id")

'''
## flickr 8k dataset
# flickr8k/word_ids/captions_content_dict.pkl: id - keywords
# flickr8k/wprd_ids/captions_dict.pkl: id - caption
# keyword list: 
# vocabulary: 
## flickr 30k dataset
# vocabulary: flickr30k/word_ids/word_to_id.txt
# content-words for training the image tagger: flickr30k/word_ids/word_to_id_content.txt
'''

'''
new directories for contrastive loss training
'''
con_dir = os.path.join(new_dir, "contrastiveLoss")
model_speeech = os.path.join(con_dir, "output/sp.pth")
model_vision = os.path.join(con_dir, "output/vis.pth")