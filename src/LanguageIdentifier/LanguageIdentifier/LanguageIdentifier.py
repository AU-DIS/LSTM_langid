from LanguageIdentifier.LSTMLID import LSTMLIDModel
import torch
from pathlib import Path
import pkgutil
import pickle
import os
import sys
class LanguageIdentifier:
    def __init__(self, directory_path: Path):
        model_information_dict = torch.load(directory_path, map_location=torch.device('cpu'))
        self.model = LSTMLIDModel(model_information_dict['char_to_idx'], model_information_dict['lang_to_idx'], model_information_dict['embedding_dim'], model_information_dict['hidden_dim'], model_information_dict['layers'])
        self.model.load_state_dict(model_information_dict['model_state_dict'], strict=False)
    def predict(self, text:str):
        return self.model.predict(text)
    def rank(self, text:str):
        return self.model.rank(text)

lid = LanguageIdentifier(Path(__file__).parent/'LID_mixed_model.pkl')
def predict(text:str):
    return lid.predict(text)
def rank(text:str):
    return lid.rank(text)