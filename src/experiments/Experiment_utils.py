import logging
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from LIDModel import LIDModel
from experiments.data_loading import load_test_folds, load_training_folds, save_probs, save_lang_to_idx
from language_datasets import LangIDDataSet, PyTorchLIDDataSet
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Tuple


def add_tensor_board_writers(log_to_tensor_board: bool, _log, ex, _run) -> Tuple[
    Optional[SummaryWriter], Optional[SummaryWriter]]:
    """
    Creates SummaryWriter's for train and dev for the given experiment.
    Args:
        log_to_tensor_board: Whether to set up writers to Tensor Board
        _log: log object
        ex: experiment object
        _run: run ID

    Returns:
        A tuple of SummaryWriters (train, dev) or a tuple of None, None
    """
    if log_to_tensor_board:
        _log.info("Tensor Board logging enabled, configuring SummaryWriter's")
        # find the FileStorageObserver
        path = None
        for o in ex.observers + ex.current_run.observers:
            from sacred.observers import FileStorageObserver
            if isinstance(o, FileStorageObserver):
                path = o.basedir
                break
        if path is not None:
            run_id = _run._id
            from pathlib import Path
            log_dir = Path(path)
            log_dir = log_dir / run_id / "tb_logs"
            _log.info(f"Tensor Board logs will be written to {log_dir}")
            writer_train = SummaryWriter(log_dir=str(log_dir / "train"))
            writer_dev = SummaryWriter(log_dir=str(log_dir / "test"))
            return writer_train, writer_dev
        else:
            _log.warning(
                "No FileStorageObserver configured. Will not log to Tensor Board")
            return None, None
    else:
        return None, None

def create_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s: "%(message)s"',
                                      datefmt='%m/%d/%Y %H:%M:%S'))
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)


def test_model(data_set, model):
    model.eval()
    lang_to_idx = model.lang_to_idx
    data_loader = DataLoader(data_set, batch_size=1)
    pred_prob = np.zeros((len(data_set), len(lang_to_idx)+1))

    for i, item in enumerate(tqdm(data_loader, leave=False)):
        probs = model.rank(item['text'][0])
        for lang, prob in probs:
            pred_prob[i, lang_to_idx[lang]] = prob
        pred_prob[i, len(lang_to_idx)] = lang_to_idx[item['label'][0]]
    return pred_prob


def idx_maps(path) -> (dict, dict):
    full_dataset = LangIDDataSet(path)
    lang_to_idx = full_dataset.lang_to_idx
    char_to_idx = full_dataset.char_to_idx
    weight_dict = full_dataset.weight_dict
    return lang_to_idx, char_to_idx, weight_dict


def train_model(exp, data_set, test_dataset, _log, LIDModel: 'LIDModel', training_params,
                tb_writers, weight_dict: Optional[dict] = None):
    optimizer, weight_decay, lr, batch_size, epochs = training_params
    if optimizer.strip().lower() == "sgd":
        opti = optim.SGD(LIDModel.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opti = optim.AdamW(params=LIDModel.parameters())
    LIDModel.fit(data_set, test_dataset, tb_writers, _log, opti, epochs=epochs, weight_dict=weight_dict,
                 experiment=exp, batch_size=batch_size)


def run_training(exp, model, maps, _log, tb_writers, training_params, to_train=True):
    # Load train data set
    _log.info("Loading test data")
    test_dataset = load_test_folds()
    lang_to_idx, char_to_idx, weight_dict = maps
    test_dataset.char_to_idx = char_to_idx
    test_dataset.lang_to_idx = lang_to_idx
    test_dataset_converted = PyTorchLIDDataSet(test_dataset)
    if to_train:
        _log.info("Loading train data")
        train_dataset_normal = load_training_folds()
        train_dataset_normal.char_to_idx = char_to_idx
        train_dataset_normal.lang_to_idx = lang_to_idx
        train_dataset = PyTorchLIDDataSet(train_dataset_normal)
        _log.info("Training model")
        train_model(exp, train_dataset, test_dataset_converted, _log, model,
                    weight_dict=weight_dict, tb_writers=tb_writers, training_params=training_params)
    _log.info("Testing model")
    eval_data = test_model(data_set=test_dataset, model=model)
    _log.info("Saving model")
    model.save_model(exp)
    _log.info("Saving predictions and lang_to_idx")
    save_probs(eval_data, exp)
    save_lang_to_idx(test_dataset.lang_to_idx, exp)
    if tb_writers[0] is not None:
        tb_writers[0].close()
    if tb_writers[1] is not None:
        tb_writers[1].close()
