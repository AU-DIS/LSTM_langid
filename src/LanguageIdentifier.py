from LIDModel import LIDModel
import numpy as np
from LSTMLID import LSTMLIDModel
import torch
from pathlib import Path
import os
class LanguageIdentifier:
    def __init__(self, directory_path: Path):
        model_information_dict = torch.load(directory_path)
        self.model = LSTMLIDModel(model_information_dict['char_to_idx'], model_information_dict['lang_to_idx'], model_information_dict['embedding_dim'], model_information_dict['hidden_dim'], model_information_dict['layers'])
        self.model.load_state_dict(model_information_dict['model_state_dict'], strict=False)
    def predict(self, text:str):
        return self.model.predict(text)
    def rank(self, text:str):
        return self.model.rank(text)

lid = LanguageIdentifier(Path("../models/LID_mixed_model.pkl"))
print(lid.predict("Hello world"))
print(lid.rank("Hello world"))