import logging
from typing import List, Optional, Sequence
from langid.langid import LanguageIdentifier
import numpy as np
from sacred import Experiment
from tqdm import tqdm

from torch.utils.data import DataLoader

from experiments.Experiment_utils import create_logger
from experiments.data_loading import data_loading_ingredient, load_test_folds, save_probs, save_lang_to_idx


ex = Experiment('LangID_experiment', ingredients=[data_loading_ingredient])
# Attach the logger to the experiment
ex.logger = create_logger()

@ex.capture
def test_model(data_set=None, langider=None, lang_to_idx=None, ) -> np.ndarray:
    """
    Tests a given langid.py model on the given data set.
    :param data_set: data set to test on
    :param langider: model to test
    :param lang_to_idx: mapping of languages to ids
    """
    import numpy as np
    langs = data_set.get_tag_set()
    pred_prob = np.zeros((len(data_set), len(langs) + 1))
    dataloader = DataLoader(data_set)
    for i, elem in enumerate(tqdm(dataloader)):
        text = elem['text'][0]
        label = elem['label'][0]
        ranking = langider.rank(text)
        for lang, prob in ranking:
            pred_prob[i, lang_to_idx[lang]] = prob
        pred_prob[i, len(langs)] = lang_to_idx[label]
    return pred_prob


@ex.config
def config():
    model_path = None  # Which model to load, if none, the built-in langid.py model is used


@ex.capture
def load_langid_model(model_path: Optional[str], lang_set: Sequence[str]) -> LanguageIdentifier:
    """
    Loads the provided langid.py model. If none provided, then it loads the default model.
    :param model_path: path to model to load
    :param lang_set: language set to which the model should be restricted. Provide empty list for
        no restrictions.
    :return: language identifier
    """
    if model_path is None:
        from langid import langid
        langider = LanguageIdentifier.from_modelstring(langid.model, norm_probs=True)
    else:
        langider = LanguageIdentifier.from_modelpath(model_path, norm_probs=True)
    if len(lang_set) > 0:
        langider.set_languages(langs=lang_set)
    return langider


@ex.automain
def main(_log):
    _log.info("Loading test data")
    test_dataset = load_test_folds()
    tag_set = test_dataset.get_tag_set()
    _log.info("Loading langid.py model")
    langider = load_langid_model(lang_set=tag_set)
    _log.info("Testing model")
    lang_to_idx = test_dataset.lang_to_idx
    eval_data = test_model(data_set=test_dataset, langider=langider, lang_to_idx=lang_to_idx)
    _log.info("Saving predictions and lang_to_idx")
    save_probs(eval_data, ex)
    save_lang_to_idx(test_dataset.lang_to_idx, ex)

