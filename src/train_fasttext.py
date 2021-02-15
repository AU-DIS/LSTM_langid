import glob
import os
import tempfile

import fasttext

"""
Parameters for supervised training of fasttext
    input             # training file path (required)
    lr                # learning rate [0.1]
    dim               # size of word vectors [100]
    ws                # size of the context window [5]
    epoch             # number of epochs [5]
    minCount          # minimal number of word occurences [1]
    minCountLabel     # minimal number of label occurences [1]
    minn              # min length of char ngram [0]
    maxn              # max length of char ngram [0]
    neg               # number of negatives sampled [5]
    wordNgrams        # max length of word ngram [1]
    loss              # loss function {ns, hs, softmax, ova} [softmax]
    bucket            # number of buckets [2000000]
    thread            # number of threads [number of cpus]
    lrUpdateRate      # change the rate of updates for the learning rate [100]
    t                 # sampling threshold [0.0001]
    label             # label prefix ['__label__']
    verbose           # verbose [2]
    pretrainedVectors # pretrained word vectors (.vec file) for supervised learning []
"""


def train_fasttext(path, split, model_path):
    training_tmp, test_tmp = create_temp_file_split(split, path)
    model = fasttext.train_supervised(training_tmp, epoch=15, minn=1, maxn=6)
    model.save_model(model_path)
    print(model.test(test_tmp))
    os.remove(training_tmp)
    os.remove(test_tmp)


def create_temp_file_split(split_index, data_path):
    files = []
    for filename in glob.iglob(data_path + '**/*' + '.txt', recursive=True):
        files.append(filename)

    test_tmp = tempfile.NamedTemporaryFile(encoding='utf-8', mode='w', delete=False)
    training_tmp = tempfile.NamedTemporaryFile(encoding='utf-8', mode='w', delete=False)
    for i in range(len(files)):
        filename = files[i]
        file = open(filename, 'r', encoding='utf-8')
        if str(split_index) + ".txt" in filename:
            test_tmp.write(file.read())
        else:
            training_tmp.write((file.read()))
    training_tmp_name = training_tmp.name
    test_tmp_name = test_tmp.name
    training_tmp.close()
    test_tmp.close()
    return training_tmp_name, test_tmp_name
