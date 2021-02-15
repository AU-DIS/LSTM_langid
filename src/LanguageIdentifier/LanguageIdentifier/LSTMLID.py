import torch
import torch.nn as nn
from LanguageIdentifier.LIDModel import LIDModel


class LSTMLIDModel(LIDModel):
    def __init__(self, char_to_idx, lang_to_idx, embedding_dim, hidden_dim, layers):
        super(LSTMLIDModel, self).__init__(char_to_idx, lang_to_idx)
        self.hidden_dim = hidden_dim
        self.num_layers = layers
        self.char_embeddings = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=char_to_idx["PAD"])
        # The LSTM takes character embeddings as inputs and outputs a state for each character with size hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=layers, bidirectional=True, batch_first=True)
        # The linear layer that maps from a state to a language
        self.hidden2lang = nn.Linear(hidden_dim * 2, self.lang_set_size)
        self.to(self.device)

    def forward(self, sentences):
        X = self.char_embeddings(sentences)
        X, _ = self.lstm(X)
        # We deviate slightly from the blog post here
        # Instead of majority vote, we sum over the probabilities and then choose the most likely
        # Should be a nicer gradient because of that
        X = self.hidden2lang(X)
        logit = torch.sum(X, dim=1)
        return logit

