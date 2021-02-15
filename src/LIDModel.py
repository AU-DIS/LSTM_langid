import math
import os
import tempfile
import time
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from language_datasets import BucketBatchSampler


def correct_predictions(scores, labels):
    pred = torch.argmax(scores, dim=1)
    return (pred == labels).float().sum()


class LIDModel(nn.Module):
    def __init__(self, char_to_idx, lang_to_idx):
        # Char_to_idx should be a map that converts a character to a number
        # Lang_to_idx should be a map that converts a language to a number
        # Char_to_idx should contain a padding symbol ("PAD", 0)
        self.char_to_idx = char_to_idx
        self.lang_to_idx = lang_to_idx
        self.idx_to_lang = dict([(value, key) for key, value in lang_to_idx.items()])
        self.vocab_size = len(char_to_idx) + 1  # The additional + 1 is for unknown characters
        self.lang_set_size = len(lang_to_idx)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(LIDModel, self).__init__()

    def pad_collate(self, batch):
        (sentences, labels) = batch[0]
        sentences = sentences.to(self.device)
        labels = labels.to(self.device)
        return sentences, labels

    def safe_char_to_idx(self, c):
        if c in self.char_to_idx:
            return self.char_to_idx[c]
        else:
            return len(self.char_to_idx)

    def prepare_single_sentence(self, sentence):
        idxs = [self.safe_char_to_idx(c) for c in sentence]
        return torch.tensor(idxs, dtype=torch.long, device=self.device).view(1, len(sentence))

    def forward(self, sentences):
        raise NotImplemented

    def predict(self, sentence):
        self.eval()
        prep_sentence = self.prepare_single_sentence(sentence)
        X = self(prep_sentence)
        max = torch.argmax(X)
        lang_guess = self.idx_to_lang[int(max.item())]
        self.train()
        return lang_guess

    def rank(self, sentence):
        self.eval()
        prep_sentence = self.prepare_single_sentence(sentence)
        logit = self(prep_sentence)
        smax = F.softmax(logit, dim=-1)
        arr = []
        for lang, index in self.lang_to_idx.items():
            arr.append((lang, smax[0][index].item()))
        self.train()
        return arr

    def fit(self, train_dataset, dev_dataset, tb_writers, _log, optimizer, epochs=3, batch_size=64,
            weight_dict=None,
            experiment=None):
        tb_train, tb_dev = tb_writers
        test_sampler = BucketBatchSampler(batch_size, dev_dataset)
        dataloader_dev = DataLoader(dev_dataset, shuffle=False, drop_last=False,
                                    collate_fn=self.pad_collate, sampler=test_sampler)
        weights = None
        if weight_dict is not None:
            weights = torch.zeros(len(weight_dict)).to(self.device)
            for lang in weight_dict:
                indx = self.lang_to_idx[lang]
                weights[indx] = weight_dict[lang]
        loss_train = nn.CrossEntropyLoss(weight=weights)
        loss_dev = nn.CrossEntropyLoss()

        _log.info(f"Running for {epochs} epochs")
        for epoch in range(epochs):
            self.train()
            avg_total_loss, num_correct_preds = 0, 0
            epoch_start_time = time.time()
            train_dataset.randomize_data()
            sampler = BucketBatchSampler(batch_size, train_dataset)
            dataloader_train = DataLoader(train_dataset, shuffle=False, drop_last=False,
                                          collate_fn=self.pad_collate, sampler=sampler)
            # Logit is the pre-softmax scores
            for idx, batch in enumerate(tqdm(dataloader_train, leave=False)):
                optimizer.zero_grad()
                tensor_sentences, labels = batch
                logit = self(tensor_sentences)
                loss_nll = loss_train(logit, labels)
                num_correct_preds += correct_predictions(logit, labels)
                loss = loss_nll
                avg_total_loss += loss.item()
                loss.backward()
                optimizer.step()
            avg_total_loss /= sampler.batch_count()
            accuracy = num_correct_preds / len(train_dataset)
            if math.isnan(avg_total_loss):
                _log.warning("Loss is nan. This can happen if gradients explode. Try lowering the learning rate "
                             "/ increasing weight decay")
            _log.info(f"Average training error in epoch {epoch + 1}: {avg_total_loss:.5f} "
                      f"and training accuracy: {accuracy:.4f}")
            step_num = epoch
            tb_train.add_scalar("Accuracy", accuracy, step_num)
            tb_train.add_scalar("Loss", avg_total_loss, step_num)
            self.eval()
            # Test model
            avg_total_loss, num_correct_preds = 0, 0
            for _, batch in enumerate(tqdm(dataloader_dev, leave=False)):
                tensor_sentences, labels = batch
                logit = self(tensor_sentences)
                loss_nll = loss_dev(logit, labels)
                num_correct_preds += correct_predictions(logit, labels)
                avg_total_loss += loss_nll.item()
            avg_total_loss /= test_sampler.batch_count()
            accuracy = num_correct_preds / len(dev_dataset)
            _log.info(f"Average total loss dev: {avg_total_loss:.5f}, accuracy: {accuracy:.4f}, ")
            tb_dev.add_scalar("Accuracy", accuracy, step_num)
            tb_dev.add_scalar("Loss", avg_total_loss, step_num)
            if experiment is not None:
                self.save_model(experiment, "E" + str(epoch))
            _log.info("Time spent in epoch {0}: {1:.2f} ".format(epoch + 1, time.time() - epoch_start_time))

    def save_model(self, exp, fileending=""):
        """Saves a pytorch model fully and adds it as artifact

        Arguments:
            pred_prob  -- list or numpy array to save as .npy file
        """
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        required_model_information = {'char_to_idx': self.char_to_idx, 'lang_to_idx': self.lang_to_idx,
                                 'model_state_dict': self.state_dict()}

        torch.save(required_model_information, tmpf.name)
        fname = "trained_model_dict" + fileending + ".pth"
        exp.add_artifact(tmpf.name, fname)
        tmpf.close()
        os.unlink(tmpf.name)
