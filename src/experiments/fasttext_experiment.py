import fasttext
from time import sleep
from tqdm import tqdm
import logging
from sacred import Experiment
from torch.utils.data import DataLoader

from experiments.data_loading import data_loading_ingredient, load_training_folds, save_probs, save_lang_to_idx, \
    load_test_folds
from train_fasttext import train_fasttext

fasttext_exp = Experiment('Fasttext_experiment', ingredients=[data_loading_ingredient])

# set up a custom logger
logger = logging.getLogger()
logger.handlers = []
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s: "%(message)s"',
                                  datefmt='%m/%d/%Y %H:%M:%S'))
logger.addHandler(ch)
logger.setLevel(logging.INFO)

# Attach the logger to the experiment
fasttext_exp.logger = logger

@fasttext_exp.config
def config():
    model_path = "../models/fasttext_lid_small.ftz"
    num_lang_preds = 300
    to_train = False
    to_quant = True


@fasttext_exp.capture
def test_model(data_set, model, num_lang_preds, lang_to_idx):
    langs = lang_to_idx.keys()
    pred_prob = []
    pred_prob10 = []
    dataloader = DataLoader(data_set)
    for i, elem in enumerate(tqdm(dataloader)):
        data_item = data_set.__getitem__(i)
        result = model.predict(data_item['text'], num_lang_preds, threshold=-1)
        probabilities = []
        for lang in langs:
            for j in range(len(result[0])):
                lang_guess = result[0][j][-3:]
                if "_"+lang == lang_guess:
                    probabilities.append(result[1][j])
        probabilities.append(lang_to_idx[data_item['label']])
        pred_prob.append(probabilities)
    return pred_prob


@fasttext_exp.capture
def load_fasttext_model(model_path):
    return fasttext.load_model(model_path)

@fasttext_exp.automain
def main(model_path : str, num_lang_preds, data_loader, to_train, to_quant, _log):
    # Load train data set
    train_path = data_loader['train_path']
    test_path = data_loader['test_path']
    test_fold = data_loader['test_folds'][0]
    if to_train:
        _log.info("Loading train data")
        train_fasttext(train_path, test_fold, model_path)

    _log.info("Loading test data")
    test_dataset = load_test_folds()
    _log.info("Loading fasttext model")
    fasttext_predictor = load_fasttext_model(model_path=model_path)
    _log.info("Testing model")
    lang_to_idx = test_dataset.lang_to_idx
    pred_prob = test_model(data_set=test_dataset, model=fasttext_predictor, lang_to_idx=lang_to_idx, num_lang_preds=num_lang_preds)
    _log.info("Saving predictions and lang_to_idx, be aware that the probabilities doesn't add up to 1")
    save_probs(pred_prob, fasttext_exp)
    save_lang_to_idx(lang_to_idx, fasttext_exp)


