# The following import should be removed once PEP 563 becomes the default
# https://www.python.org/dev/peps/pep-0563/
from __future__ import annotations

import random
from typing import Optional, Sequence

import jsonlines
import torch
from torch.utils.data import Sampler, Dataset
from random import shuffle


class LangIDDataSet(Dataset):
    """
    Basic class to model data sets for language identification.
    """

    def __init__(self, file_path, min_length=5):
        self.initial_data = []
        self.data = []
        self.char_to_idx = {'PAD': 0}
        self.lang_to_idx = {}
        self._load_from_file(file_path, min_length=min_length)
        self.weight_dict = self.make_weight_dict()

    def _load_from_file(self, file_path: Optional[str], min_length=5):
        if file_path is not None:
            char_set = set()
            lang_set = set()
            with jsonlines.open(file_path) as reader:
                for line in reader:
                    if len(line['text']) < min_length:
                        continue
                    self.initial_data.append(line)
                    for c in line['text']:
                        char_set.add(c)
                    lang_set.add(line['label'])
            for c in sorted(char_set):
                if c not in self.char_to_idx.keys():
                    self.char_to_idx[c] = len(self.char_to_idx)
            for lang in sorted(lang_set):
                if lang not in self.lang_to_idx.keys():
                    self.lang_to_idx[lang] = len(self.lang_to_idx)
        self.data = self.initial_data

    def make_weight_dict(self) -> dict:
        """
        Instantiates the weight dict for this dataset
        The formula used is weight = most_frequent/lang_freq.
        Such that the most frequent has a frequency of 1

        :return: A dict with a mapping from a language to weight
        """
        weight_dict = None
        if len(self.initial_data) > 0:
            frequency_dict = {}
            labels = [elem['label'] for elem in self.initial_data]
            for label in self.lang_to_idx.keys():
                frequency_dict[label] = labels.count(label)
            most_frequent = max(frequency_dict.values())
            weight_dict = {label:(most_frequent/frequency_dict[label]) for label in frequency_dict.keys()}
        return weight_dict

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def make_from_files(files: Sequence[str]) -> LangIDDataSet:
        """
        Creates an instance of LangIDDataSet that contains the content of the given files.
        :param files: list of files to read from
        :return: instance of LangIDDataSet that contains the content of all given files
        """
        if len(files) == 0:
            raise ValueError("No files provided")
        val_to_return = LangIDDataSet(file_path=None)
        for f in files:
            val_to_return._load_from_file(f)
        val_to_return.initial_data = val_to_return.data
        val_to_return.weight_dict = val_to_return.make_weight_dict()
        return val_to_return

    def get_tag_set(self) -> list:
        """returns ordered list of language labels in the dataset
        Returns:
            list -- Ordered list of language labels
        """
        langs = list(self.lang_to_idx.keys())
        langs.sort()
        return langs

    def get_lang_to_idx(self) -> dict:
        """get dict from lang to id, ordered alphabetically
        Returns:
            dict -- For converting language code to an id
        """
        lang_to_idx = {}
        for lang in self.get_tag_set():
            lang_to_idx[lang] = len(lang_to_idx)
        return lang_to_idx

    def randomize_data(self, upper_lim=20, lower_lim=5):
        """
        Takes the original data and creates random length examples with length between upper limit and lower limit
        :param upper_lim: The maximum character length of training example
        :param lower_lim: The minimum character length of training example
        """
        new_data = []
        for line in self.initial_data:
            sentence = line['text']
            label = line['label']
            remaining = sentence
            while lower_lim < len(remaining):
                lim = random.randint(lower_lim, upper_lim)
                m = min(len(remaining), lim)
                new_sentence = remaining[:m]
                new_data.append({'text': new_sentence, 'label': label})
                split = remaining[m:].split(" ", 1)
                if len(split) <= 1:
                    break
                remaining = split[1]
        random.shuffle(new_data)
        self.data = new_data

    def save_data_as_jsonl(self, output_file):
        with jsonlines.open(output_file, mode='w') as writer:
            for line in self.data:
                writer.write(line)


class PyTorchLIDDataSet(Dataset):
    """
    PyTorch-specific wrapper that converts items to PyTorch tensors.
    """

    def __init__(self, decoree: LangIDDataSet):
        self.data = []
        if decoree is not None:
            self.decoree = decoree
        self.char_to_idx = decoree.char_to_idx
        self.lang_to_idx = decoree.lang_to_idx
        self.tensorify_all()

    def __getitem__(self, idx):
        if not isinstance(idx, list):
            return self.data[idx]
        txt = []
        label = []
        for i in idx:
            item = self.data[i]
            txt.append(item[0])
            label.append(item[1])
        return torch.stack(txt), torch.stack(label)

    def make_weight_dict(self) -> dict:
        return self.decoree.make_weight_dict()

    def __len__(self):
        return len(self.data)

    def get_tag_set(self) -> list:
        return self.decoree.get_tag_set()

    def get_char_to_idx(self) -> dict:
        return self.decoree.char_to_idx

    def get_lang_to_idx(self) -> dict:
        """get dict from lang to id, ordered alphabetically
        Returns:
            dict -- For converting language code to an id
        """
        return self.decoree.get_lang_to_idx()

    def randomize_data(self, upper_lim=20, lower_lim=5):
        """
        Takes the original data and creates random length examples with length between upper limit and lower limit.
        :param upper_lim: The maximum character length of training example
        :param lower_lim: The minimum character length of training example
        """
        new_data = []
        for line in self.decoree.initial_data:
            sentence = line['text']
            label = line['label']
            remaining = sentence
            while lower_lim < len(remaining):
                lim = random.randint(lower_lim, upper_lim)
                m = min(len(remaining), lim)
                new_sentence = remaining[:m]
                new_data.append(self.tensorify({'text': new_sentence, 'label': label}))
                split = remaining[m:].split(" ", 1)
                if len(split) <= 1:
                    break
                remaining = split[1]
        random.shuffle(new_data)
        self.data = new_data

    def tensorify(self, data_point):
        sentence = data_point['text']
        lang = data_point['label']
        idxs = [self.char_to_idx.get(c, len(self.char_to_idx)) for c in sentence]
        return torch.tensor(idxs, dtype=torch.long), torch.tensor(self.lang_to_idx[lang], dtype=torch.long)

    def tensorify_all(self):
        new_data = []
        for elem in self.decoree:
            new_data.append(self.tensorify(elem))
        self.data = new_data

    def set_lang_to_idx(self, l_to_idx):
        self.lang_to_idx = l_to_idx
        self.decoree.lang_to_idx = l_to_idx

    def set_char_to_idx(self, c_to_idx):
        self.char_to_idx = c_to_idx
        self.decoree.char_to_idx = c_to_idx


# Based on https://github.com/chrisvdweth/ml-toolkit/blob/master/pytorch/utils/data/text/dataset.py
class BucketBatchSampler(Sampler):
    """
    This class creates batches containing equal length examples.
    """
    def __init__(self, batch_size, inputs):
        self.batch_size = batch_size
        self.input = inputs
        self.batch_list = self._generate_batch_map()
        self.num_batches = len(self.batch_list)

    def _generate_batch_map(self):
        batch_map = {}
        for idx, item in enumerate(self.input):
            length = len(item[0])
            if length not in batch_map:
                batch_map[length] = [idx]
            else:
                batch_map[length].append(idx)
        # Use batch_map to split indices into batches of equal size
        # e.g., for batch_size=3, batch_list = [[23,45,47], [49,50,62], [63,65,66], ...]
        batch_list = []
        for length, indices in batch_map.items():
            for group in [indices[i:(i + self.batch_size)] for i in range(0, len(indices), self.batch_size)]:
                batch_list.append(group)
        return batch_list

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        shuffle(self.batch_list)
        for i in self.batch_list:
            yield i
