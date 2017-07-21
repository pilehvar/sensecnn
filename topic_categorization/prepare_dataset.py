#! /usr/bin/env python
# -*- coding: utf-8 -*-

import random
import cPickle as pkl
import gzip
import numpy
import os
import inspect
import logging
import argparse

logging.basicConfig()
logger = logging.getLogger('data preparation')
logger.setLevel(logging.INFO)

coarse_map_20k = {
    0:5, #class-1 alt.atheism_dis
    1:0, #class-2 comp.graphics_dis
    2:0, #class-3 comp.os.ms-windows.misc_dis
    3:0, #class-4 comp.sys.ibm.pc.hardware_dis
    4:0, #class-5 comp.sys.mac.hardware_dis
    5:0, #class-6 comp.windows.x_dis
    6:3, #class-7 misc.forsale_dis
    7:1, #class-8 rec.autos_dis
    8:1, #class-9 rec.motorcycles_dis
    9:1, #class-10        rec.sport.baseball_dis
    10:1,#class-11        rec.sport.hockey_dis
    11:2,#class-12        sci.crypt_dis
    12:2,#class-13        sci.electronics_dis
    13:2,#class-14        sci.med_dis
    14:2,#class-15        sci.space_dis
    15:5,#class-16        soc.religion.christian_dis
    16:4,#class-17        talk.politics.guns_dis
    17:4,#class-18        talk.politics.mideast_dis
    18:4,#class-19        talk.politics.misc_dis
    19:5,#class-20        talk.religion.misc_dis"
    }


def read_files(path, class_num, dataset_id):
    annotated_sentences = []

    with open(path+'/class-' + str(class_num) + '/' + dataset_id + '.txt') as ifile:
        for line in ifile:
            annotated_sentences.append(line.strip())

    return annotated_sentences


def build_dict(paths, dataset_id, classes, filter=[]):

    sentences = []

    if len(filter) == 0:
        for i in range(classes):
            filter.append(i+1)

    for path in paths:
        for i in filter:
            sentences.extend(read_files(path, i, dataset_id))

    logger.info('Read {} sentences.'.format(len(sentences)))

    logger.info('Building dictionary...')

    wordcount = dict()
    for ss in sentences:
        words = ss.strip().split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1

    counts = wordcount.values()
    keys = wordcount.keys()

    sorted_idx = numpy.argsort(counts)[::-1]

    worddict = dict()
    rev_dic = dict()

    for idx, ss in enumerate(sorted_idx):
        worddict[keys[ss]] = idx+2  # leave 0 and 1 (UNK)
        rev_dic[idx+2] = keys[ss]

    logger.info('{} total words with {} unique words'.format(numpy.sum(counts),len(keys)))

    return worddict, rev_dic


def grab_data(path, class_num, dataset_id, dictionary):

    sentences = read_files(path, class_num, dataset_id)
    logger.debug('class-{}: {} '.format(str(class_num),len(sentences)))

    seqs = [None] * len(sentences)
    for idx, ss in enumerate(sentences):
        words = ss.strip().split()
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]

    return seqs


def fold_data(all_data, folds, fold_number):
    train_x = all_data[0]
    train_y = all_data[1]

    rand_indices = []
    for i in range(len(train_x)):
        rand_indices.append(i)
    random.seed(1337)
    random.shuffle(rand_indices)

    data = []
    for i in range(len(train_x)):
        data.append((train_x[i], train_y[i]))

    if len(rand_indices) != len(data):
        logger.error('Rand index does not match! {} {}'.format(len(rand_indices), len(data)))

    div_size = int(len(data) / folds)

    train_x = []
    train_y = []
    test_x = []
    test_y = []

    for k, ind in enumerate(rand_indices):
        td = data[ind]

        if k > ((fold_number - 1) * div_size) and k <= (fold_number * div_size):
            test_x.append(td[0])
            test_y.append(td[1])
        else:
            train_x.append(td[0])
            train_y.append(td[1])

    return (train_x, train_y), ([],[]), (test_x, test_y)


def compile_dataset( dataset_id,
                     num_classes,
                     text_dataset_path,
                     out_dir,
                     cv_setting,
                     filter_classes=[],
                     coarse={}):

    paths = []
    if cv_setting:
        paths.append(text_dataset_path)
    else:
        paths.append(text_dataset_path + "train/")
        paths.append(text_dataset_path + "test/")


    dictionary, rev_dict = build_dict(paths,
                                              dataset_id,
                                              num_classes,
                                              filter=filter_classes)

    cwd = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    out_path = cwd + "/" + out_dir + "/"

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    filepath = out_path + dataset_id

    f = open(filepath+'.dict.pkl', 'wb')
    pkl.dump(dictionary, f, -1)
    pkl.dump(rev_dict, f, -1)
    f.close()

    # if the classes are to be limited to a subset
    classes_to_check = filter_classes
    if len(classes_to_check) == 0:
        for i in range(num_classes):
            classes_to_check.append(i+1)

    logger.info("Generating the dataset... collecting sentences.")

    train_x = []
    train_y = []

    k = 0
    for i in classes_to_check:
        h = k
        if len(coarse) > 0:
            #coarsen the classes, group them according to the map given
            h = coarse[k]

        # paths[0] contains the training portion (or whole dataset in case of cross-validation)
        x_class = grab_data(paths[0], i, dataset_id, dictionary)
        train_x.extend(x_class)
        train_y.extend([h] * len(x_class))
        k += 1

    f = open(filepath + '.pkl', 'wb')
    pkl.dump((train_x, train_y), f, -1)

    if cv_setting:
        # no test portion in this case
        pkl.dump(([], []), f, -1)

    else:
        logger.info("Compiling the test portion...")
        test_x = []
        test_y = []

        k = 0
        for i in classes_to_check:
            h = k
            if len(coarse) > 0:
                # coarsen the classes
                h = coarse[k]

            x_class = grab_data(paths[1], i, dataset_id, dictionary)
            test_x.extend(x_class)
            test_y.extend([h] * len(x_class))
            k += 1

        pkl.dump((test_x, test_y), f, -1)

    f.close()

    logger.info('Compiled the dataset and saved to {}'.format(out_path))



def load_data(path, n_words=100000, valid_portion=0.1, maxlen=None,
              sort_by_len=True):

    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    train_set = pkl.load(f)
    test_set = pkl.load(f)
    f.close()
    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y

    # split training set into validation set
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)

    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    train_set_x = remove_unk(train_set_x)
    valid_set_x = remove_unk(valid_set_x)
    test_set_x = remove_unk(test_set_x)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    test = (test_set_x, test_set_y)

    return train, valid, test


def get_word_index(path, revDic=True):
    rev_dic = dict()
    with open(path) as f:
        dic = pkl.load(f)
        if revDic:
            try:
                rev_dic = pkl.load(f)
            except EOFError:
                rev_dic = {v: k for k, v in dic.items()}
    if revDic:
        return dic, rev_dic
    else:
        return dic


if __name__ == '__main__':

    '''
    Reads the raw datasets from text_dataset_path and compiles them.


    The text_dataset_path should contain n sub-directories, for the n classes of the dataset, named "class-i" {i=1..n}
    class-i contains a single text file, named dataset_id.txt (for each configuration) with each line representing an item (eg, sentence, paragraph, ...)

    For instance, for a dataset (say "my_dataset") with three classes, we have:

    my_dataset/
        class-1/
            word.txt
            wn.txt
        class-2/
            word.txt
            wn.txt
        class-3/
            word.txt
            wn.txt


    * For datasets that come with training/test partition (eg, 20k), the structure remains the same but within two main directories: "train" and "test".
      Each folder contains n sub-directories for the n classes of the corresponding portion of the dataset.

      my_dataset/
            train/
                class-1/
                    word.txt
                    wn.txt
            ...

            test/
                class-1/
                    word.txt
                    wn.txt
            ...


    '''

    # dataset_name = 'bbc'
    # dataset_id = 'supersense_bn'
    # text_dataset_path = '/Users/taher/Work/Data/text-classification/document_categorization/package/bbc/'
    # number_of_classes = 5
    # filtered = []
    # cross_validation = True
    # coarsening_map = {}

    # dataset_name = '20k'
    # dataset_id = 'supersense_bn'
    # text_dataset_path = '/Users/taher/Work/Data/text-classification/document_categorization/package/20k/'
    # number_of_classes = 20
    # filtered = []
    # cross_validation = False
    # coarsening_map = coarse_map_20k


    # dataset_name = 'ohsumed'
    # dataset_id = 'supersense_bn'
    # text_dataset_path = '/Users/taher/Work/Data/text-classification/document_categorization/package/ohsumed/'
    # number_of_classes = 23
    # filtered = []
    # cross_validation = False
    # coarsening_map = {}


    out_dir = 'data/' + dataset_name + '/'
    compile_dataset(dataset_id,
                    number_of_classes,
                    text_dataset_path,
                    out_dir,
                    cross_validation,
                    filter_classes=filtered,
                    coarse=coarsening_map)

