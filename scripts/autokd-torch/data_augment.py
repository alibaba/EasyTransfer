import os
import time

import nlpaug.augmenter.word as naw
import csv
import sys
import random
import torch

aug_p_sentence_pair = 0.3
aug_p = 0.4
action = "substitute"
glove_model_path = "/home/daoyuanchen.cdy/work/bert_with_NAS/WORDEMB/glove.6B.300d.txt"
fasttext_model_path = "/home/daoyuanchen.cdy/work/bert_with_NAS/WORDEMB/wiki-news-300d-1M-subword.vec"
bert_model_path = "/home/daoyuanchen.cdy/work/bert_with_NAS/useful_codes/PyTransformer/examples/pre_trained_bert_models"

# QQP bug for *2 ~ *3
DATASET = "MNLI"
torch.cuda.set_device(1)
if DATASET == "SST2":
    data_dir = "/home/daoyuanchen.cdy/work/bert_with_NAS/GLUE/SST-2"
else:
    data_dir = "/home/daoyuanchen.cdy/work/bert_with_NAS/GLUE/" + DATASET
BEGIN_FROM_MID = True
# QQP: 363871,  MNLI: 392703
BEGIN_FROM_MID_POINT = 98175 * 2
END_AT_MID_POINT = 98175 * 3
augment_file = os.path.join(data_dir, "train.tsv")
out_file = os.path.join(data_dir, "train_aug_all.tsv") if not BEGIN_FROM_MID else \
    os.path.join(data_dir, "train_aug_" + str(BEGIN_FROM_MID_POINT) + "to" + str(END_AT_MID_POINT) + ".tsv")
print("Processing dataset: " + DATASET)
print("Out file path: " + out_file)

print("loading......, " + time.asctime(time.localtime(time.time())))
augs = [
    naw.WordEmbsAug(model_type='glove', model_path=glove_model_path, action=action),
    # naw.WordEmbsAug(model_type='fasttext', model_path=fasttext_model_path, action=action),
    naw.ContextualWordEmbsAug(model_path=bert_model_path, action=action, device='cuda')
]
print("loading finished. " + time.asctime(time.localtime(time.time())))


augment_iter = 20



# QQP format
# id	qid1	qid2	question1	question2	is_duplicate
def read_QQP(line):
    text_a = line[-3]
    text_b = line[-2]
    label = line[-1]
    other = ["0"] * 3  # the first eight columns will not be processed
    return text_a, text_b, other, label


def out_QQP(text_a, text_b, other, label, new_lines):
    augment_count = 0
    for _ in range(augment_iter):
        auger = random.choice(augs)
        auger.aug_p = aug_p_sentence_pair
        try:
            augmented_text_a = auger.augment(text_a)
            augmented_text_b = auger.augment(text_b)
        except ValueError:
            continue
        augment_count += 1
        out_list = other + [augmented_text_a, augmented_text_b, label]
        # f_out.write("\t".join(out_list) + "\n")
        new_lines.append("\t".join(out_list) + "\n")
    if augment_count != augment_iter:
        print("Several augmentations fail, success number is " + str(augment_count) + "\n")

# RTE format
# index	sentence1	sentence2	label

# QNLI format
# index	question	sentence	label
def read_qnli(line):
    text_a = line[-3]
    text_b = line[-2]
    label = line[-1]
    other = line[0]  # the first eight columns will not be processed
    return text_a, text_b, other, label

def out_qnli(text_a, text_b, other, label, new_lines):
    augment_count = 0
    for _ in range(augment_iter):
        auger = random.choice(augs)
        auger.aug_p = aug_p_sentence_pair
        try:
            augmented_text_a = auger.augment(text_a)
            augmented_text_b = auger.augment(text_b)
        except ValueError:
            continue
        augment_count += 1
        out_list = [other, augmented_text_a, augmented_text_b, label]
        new_lines.append("\t".join(out_list) + "\n")
    if augment_count != augment_iter:
        print("Several augmentations fail, success number is " + str(augment_count) + "\n")



# MNLI format
# index	promptID	pairID	genre	sentence1_binary_parse	sentence2_binary_parse	sentence1_parse	sentence2_parse	 sentence1	sentence2	label1	gold_label
def read_mnli(line):
    text_a = line[-4]
    text_b = line[-3]
    label = line[-1]
    other = ["x"] * 8  # the first eight columns will not be processed
    return text_a, text_b, other, label


def out_mnli(text_a, text_b, other, label, new_lines):
    augment_count = 0
    for _ in range(augment_iter):
        auger = random.choice(augs)
        auger.aug_p = aug_p_sentence_pair
        try:
            augmented_text_a = auger.augment(text_a)
            augmented_text_b = auger.augment(text_b)
        except ValueError:
            continue
        augment_count += 1
        out_list = other + [augmented_text_a, augmented_text_b, label, label]
        # f_out.write("\t".join(out_list) + "\n")
        new_lines.append("\t".join(out_list) + "\n")
    if augment_count != augment_iter:
        print("Several augmentations fail, success number is " + str(augment_count) + "\n")


def read_sst2(line):
    text_a = line[0]
    text_b = ''
    label = line[1]
    return text_a, text_b, label

def out_sst2(text_a, text_b, other, f_out, new_lines):
    augment_count = 0
    for _ in range(augment_iter):
        auger = random.choice(augs)
        auger.aug_p = aug_p
        try:
            augmented_text = auger.augment(text_a)
        except ValueError:
            continue
        augment_count += 1
        # f_out.write(augmented_text + "\t" + other + "\n")
        new_lines.append(augmented_text + "\t" + other + "\n")

    if augment_count != augment_iter:
        print("Several augmentations fail, success number is " + str(augment_count) + "\n")


def read_mrpc(line):
    text_a = line[-2]
    text_b = line[-1]
    other = line[0:-2]
    return text_a, text_b, other


def out_mrpc(text_a, text_b, other, f_out, new_lines):
    augment_count = 0
    for _ in range(augment_iter):
        auger = random.choice(augs)
        auger.aug_p = aug_p_sentence_pair
        try:
            augmented_text_a = auger.augment(text_a)
            augmented_text_b = auger.augment(text_b)
        except ValueError:
            continue
        augment_count += 1
        out_list = other + [augmented_text_a, augmented_text_b]
        new_lines.append("\t".join(out_list) + "\n")
    if augment_count != augment_iter:
        print("Several augmentations fail, success number is " + str(augment_count) + "\n")

save_freq = 500
with open(out_file, "a+", encoding="utf-8-sig") as f_out:
    new_lines = []
    line_number = 0
    with open(augment_file, "r", encoding="utf-8-sig") as f_in:
        reader = csv.reader(f_in, delimiter="\t", quotechar=None)
        print("begin augmentation. " + time.asctime(time.localtime(time.time())))
        if BEGIN_FROM_MID:
            for _ in range(BEGIN_FROM_MID_POINT):
                next(reader)
            line_number = BEGIN_FROM_MID_POINT
        else:
            first_line = next(reader)
            # f_out.write('\t'.join(first_line) + "\n")
            new_lines.append('\t'.join(first_line) + "\n")
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            if DATASET == 'SST2':
                text_a, text_b, other = read_sst2(line)
                # f_out.write('\t'.join(line) + "\n")
                new_lines.append('\t'.join(line) + "\n")
                out_sst2(text_a, text_b, other, f_out, new_lines)
            elif DATASET == "MRPC":
                text_a, text_b, other = read_mrpc(line)
                # f_out.write('\t'.join(line) + "\n")
                new_lines.append('\t'.join(line) + "\n")
                out_mrpc(text_a, text_b, other, f_out, new_lines)
            elif DATASET == "MNLI":
                text_a, text_b, other, label = read_mnli(line)
                new_lines.append('\t'.join(line) + "\n")
                out_mnli(" ".join(text_a.split()), " ".join(text_b.split()), other, label, new_lines)
            elif DATASET == "QQP":
                text_a, text_b, other, label = read_QQP(line)
                new_lines.append('\t'.join(line) + "\n")
                out_QQP(text_a, text_b, other, label, new_lines)
            elif DATASET == "QNLI":
                text_a, text_b, other, label = read_qnli(line)
                new_lines.append('\t'.join(line) + "\n")
                out_qnli(" ".join(text_a.split()), " ".join(text_b.split()), other, label, new_lines)
            elif DATASET == "RTE":
                # the format of RTE is the same as the QNLI
                text_a, text_b, other, label = read_qnli(line)
                new_lines.append('\t'.join(line) + "\n")
                out_qnli(" ".join(text_a.split()), " ".join(text_b.split()), other, label, new_lines)
            line_number += 1
            if line_number % 10 == 0:
                print(time.asctime(time.localtime(time.time())))
                print("task " + DATASET + ", processed line " + str(line_number) + "\n")
            if len(new_lines) > save_freq:
                print("Save augmented texts at line point: " + str(line_number) + "\n")
                f_out.writelines(new_lines)
                new_lines = []
            if len(new_lines) > END_AT_MID_POINT:
                print("Will exit, save augmented texts at line point: " + str(line_number) + "\n")
                f_out.writelines(new_lines)
                exit()

        print("Save augmented texts at line point: " + str(line_number) + "\n")
        f_out.writelines(new_lines)






