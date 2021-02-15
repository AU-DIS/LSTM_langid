from typing import Optional

import torch
from sacred import Experiment

from experiments.Experiment_utils import idx_maps, add_tensor_board_writers, run_training, create_logger
from experiments.data_loading import data_loading_ingredient
from src.LSTMLID import LSTMLIDModel

LSTM_exp = Experiment('LSTM_experiment', ingredients=[data_loading_ingredient])

# Attach the logger to the experiment
LSTM_exp.logger = create_logger()

@LSTM_exp.config
def config():
    pretrained_model = None
    epochs = 1
    log_to_tensorboard: bool = True
    seed = 42
    hidden_dim = 100
    embedding_dim = 75
    num_lstm_layers = 2
    optimizer = 'SGD'
    lr = 0.1
    weight_decay = 0.00001
    batch_size = 64


@LSTM_exp.capture
def load_LSTM_model(pretrained_model_path: Optional[str], char_to_idx: dict, lang_to_idx: dict,
                    hidden_dim, embedding_dim, num_lstm_layers, _log):
    if pretrained_model_path is not None:
        model_dict = torch.load(pretrained_model_path)
        LSTM_model = LSTMLIDModel(model_dict['char_to_idx'], model_dict['lang_to_idx'], model_dict['embedding_dim'], model_dict['hidden_dim'], model_dict['layers'])
        LSTM_model.load_state_dict(model_dict['model_state_dict'])

    else:
        LSTM_model = LSTMLIDModel(char_to_idx=char_to_idx, lang_to_idx=lang_to_idx,
                                  hidden_dim=hidden_dim, embedding_dim=embedding_dim,
                                  layers=num_lstm_layers)
    return LSTM_model


@LSTM_exp.automain
def main(pretrained_model, _log, log_to_tensorboard, _run, data_loader,
         epochs, weight_decay, batch_size, lr, optimizer):
    training_params = optimizer, weight_decay, lr, batch_size, epochs
    # Load train data set
    tb_writers = add_tensor_board_writers(log_to_tensorboard, _log, LSTM_exp, _run)
    _log.info("Loading idx maps")
    maps = idx_maps(f"{data_loader['train_path']}/all.jsonl")
    lang_to_idx, char_to_idx, weight_dict = maps
    _log.info("Loading LSTM model")
    LSTM_model = load_LSTM_model(pretrained_model_path=pretrained_model, char_to_idx=char_to_idx,
                                 lang_to_idx=lang_to_idx, _log=_log)
    to_train = pretrained_model is None
    run_training(LSTM_exp, LSTM_model, maps, _log, tb_writers, training_params, to_train)
