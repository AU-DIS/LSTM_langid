import torch
import torch.nn as nn
from LIDModel import LIDModel
import tempfile
import os

class LSTMLIDModel(LIDModel):
    def __init__(self, char_to_idx, lang_to_idx, embedding_dim, hidden_dim, layers):
        super(LSTMLIDModel, self).__init__(char_to_idx, lang_to_idx)
        self.hidden_dim = hidden_dim
        self.num_layers = layers
        self.embedding_dim = embedding_dim
        self.char_embeddings = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=char_to_idx["PAD"])
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=layers, bidirectional=True, batch_first=True)
        self.hidden2lang = nn.Linear(hidden_dim * 2, self.lang_set_size)
        self.to(self.device)

    def forward(self, sentences):
        X = self.char_embeddings(sentences)
        X, _ = self.lstm(X)
        X = self.hidden2lang(X)
        logit = torch.sum(X, dim=1)
        return logit
    def save_model(self, exp, fileending=""):
        """Saves a dict containing statedict and other required model parameters and adds it as artifact

        Arguments:

        """
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        required_model_information = {'char_to_idx': self.char_to_idx, 'lang_to_idx': self.lang_to_idx,
            'model_state_dict': self.state_dict(), 'embedding_dim': self.embedding_dim, 'hidden_dim': self.hidden_dim,
            'layers': self.num_layers}

        torch.save(required_model_information, tmpf.name)
        fname = "trained_LID_model" + fileending + ".pth"
        exp.add_artifact(tmpf.name, fname)
        tmpf.close()
        os.unlink(tmpf.name)

