import os
import pickle
import tempfile
from typing import Optional, List
import numpy as np
from sacred import Ingredient, Experiment

from language_datasets import LangIDDataSet

data_loading_ingredient = Ingredient('data_loader')


@data_loading_ingredient.config
def config():
    train_path = "../datasets/processed/UD20langs"
    test_path = "../datasets/processed/UD20_10Char"
    train_folds = [1, 2, 3, 4]
    test_folds = [0]


@data_loading_ingredient.capture
def load_training_test_set(train_path: str, test_path: str, train_folds: List[int], test_folds: List[int]) -> (
Optional[LangIDDataSet], Optional[LangIDDataSet]):
    """
    Loads the provided folds into two datasets.
    :param training_path: where to find training folds (i.e the x.jsonl files)
    :param test_path: where to find test folds (i.e the x.jsonl files), equal to training path if unspecified
    :param train_folds: Folds to include from training_path
    :param test_folds: Folds to include from test_path
    :return: (train_dataset, test_dataset)
    """
    train_dataset = None
    if train_folds is not None and len(train_folds) > 0:
        files_to_load_from = [f"{train_path}/{id}.jsonl" for id in train_folds]
        train_dataset = LangIDDataSet.make_from_files(files_to_load_from)
    test_dataset = None
    if test_folds is not None and len(test_folds) > 0:
        files_to_load_from = [f"{test_path}/{id}.jsonl" for id in test_folds]
        test_dataset = LangIDDataSet.make_from_files(files_to_load_from)
    return train_dataset, test_dataset


@data_loading_ingredient.capture
def load_training_folds(train_folds: List[int], train_path: str) -> Optional[LangIDDataSet]:
    """
    Loads the provided folds into a single dataset.
    :param folds: set of folds
    :param data_path: where to find the folds (i.e. the x.jsonl files)
    :return: a dataset consisting of all provided folds
    """
    return load_folds(train_folds, train_path)


@data_loading_ingredient.capture
def load_test_folds(test_folds: List[int], test_path: str) -> Optional[LangIDDataSet]:
    """
    Loads the provided folds into a single dataset.
    :param folds: set of folds
    :param data_path: where to find the folds (i.e. the x.jsonl files)
    :return: a dataset consisting of all provided folds
    """
    return load_folds(test_folds, test_path)


def load_folds(folds: List[int], path: str) -> Optional[LangIDDataSet]:
    dataset = None
    if folds is not None and len(folds) > 0:
        files_to_load_from = [f"{path}/{id}.jsonl" for id in folds]
        dataset = LangIDDataSet.make_from_files(files_to_load_from)
    return dataset


@data_loading_ingredient.capture
def save_probs(pred_prob, ex: Experiment, file_ending=""):
    """Saves probabilities as a .npy file and adds it as artifact

    Arguments:
        pred_prob  -- list or numpy array to save as .npy file
    """
    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
    np.save(tmpf.name, pred_prob)
    fname = "prediction_probabilities" + file_ending + ".npy"
    ex.add_artifact(tmpf.name, fname)
    tmpf.close()
    os.unlink(tmpf.name)


@data_loading_ingredient.capture
def save_lang_to_idx(lang_to_idx: dict, ex: Experiment):
    """Saves the lang_to_idx dict as an artifact

    Arguments:
        lang_to_idx {dict} -- The dict to save in a file
    """
    tmpf = tempfile.NamedTemporaryFile(dir="", delete=False, suffix=".pkl")
    pickle.dump(lang_to_idx, tmpf)
    tmpf.flush()
    ex.add_artifact(tmpf.name, "lang_to_idx.pkl")
    tmpf.close()
    os.unlink(tmpf.name)
