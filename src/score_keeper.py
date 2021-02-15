from typing import List

import matplotlib
from pathlib import Path
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


class score_keeper:
    def add_prediction(self, predicted, label):
        self.pred_labels.append((predicted, label))

    def print_score(self, number):
        print(number)

    def __init__(self):
        self.pred_labels = []
        self.accuracy1Counter = 0
        self.accuracy3Counter = 0
        self.accuracy5Counter = 0

    def add_pred_prob(self, lang_to_idx, pred_probs):
        idx_to_lang = {v: k for k, v in lang_to_idx.items()}
        for elem in pred_probs:
            pred_prob_tuples = []
            for idx in idx_to_lang.keys():
                pred_prob_tuples.append((idx_to_lang[idx], elem[idx]))
            sorted_probs = sorted(pred_prob_tuples, key=lambda x: x[1], reverse=True)
            self.add_prediction(sorted_probs[0][0], idx_to_lang[elem[-1]])

            if sorted_probs[0][0] == idx_to_lang[elem[-1]]:
                self.accuracy1Counter += 1

            top3langs = []
            for i in range(3):
                top3langs.append(sorted_probs[i][0])
            if (idx_to_lang[elem[-1]] in top3langs):
                self.accuracy3Counter += 1
            top5langs = []
            for i in range(5):
                top5langs.append(sorted_probs[i][0])
            if (idx_to_lang[elem[-1]] in top5langs):
                self.accuracy5Counter += 1

    def get_accuracy1(self):
        return self.accuracy1Counter / len(self.pred_labels)

    def get_accuracy3(self):
        return self.accuracy3Counter / len(self.pred_labels)

    def get_accuracy5(self):
        return self.accuracy5Counter / len(self.pred_labels)

    def print_confusion_matrix(self):
        predictions, labels = zip(*self.pred_labels)
        df = pd.DataFrame({'preds': predictions, 'labels': labels})
        category_types_labels = df.labels.unique()
        category_types_preds = df.labels.unique()
        cat_dtype = pd.api.types.CategoricalDtype(
            categories=category_types_labels, ordered=True)
        df_pred = df.preds.astype(cat_dtype).cat.codes
        df_labels = df.labels.astype(cat_dtype).cat.codes
        result = pd.concat([df_pred, df_labels], axis=1)
        if (-1 in df_pred.values):
            category_types_preds = np.insert(category_types_preds, 0, "Other")
        confusion_matrix = pd.crosstab(df_labels, df_pred, rownames=['Actual'], colnames=['Predicted'],
                                       normalize='index')
        ax = sns.heatmap(confusion_matrix, annot=True, xticklabels=category_types_preds,
                         yticklabels=category_types_labels)
        # Inverting axis to put "other" to the right
        ax.invert_xaxis()
        ax.invert_yaxis()
        plt.show()

    def print_confusion_matrix_all_langs(self, langs: List, base_path = None):
        #This looks very ugly if many different languages are present
        predictions, labels = zip(*self.pred_labels)
        pred_labels = sorted(list(set(predictions)))
        actual_labels = sorted(list(set(labels)))
        df_preds = pd.Categorical(predictions, pred_labels)
        df_labels = pd.Categorical(labels, actual_labels)
        confusion_matrix = pd.crosstab(df_labels, df_preds, rownames=['Actual'], colnames=['Predicted'], normalize='index').round(4).multiply(100)

        confusion_matrix = confusion_matrix[langs]
        confusion_matrix = confusion_matrix.loc[langs]

        mask = confusion_matrix.applymap(lambda x : lambdafunc(x))

        cmap = matplotlib.cm.get_cmap('jet')
        rgba = cmap(6)

        plt.figure(figsize = (80,50))
        sns.set(font_scale=6)
        ax = sns.heatmap(confusion_matrix, annot=mask, xticklabels=langs, yticklabels=langs, linewidths=0.5, annot_kws={'size':64}, cmap="jet", fmt='', cbar = False, linecolor=rgba)
        plt.savefig(base_path/"confusionmatrix_all.pdf", bbox_inches='tight')
        plt.show()
    def print_confusion_matrix_select(self, langs: List, base_path = None):

        predictions, labels = zip(*self.pred_labels)
        pred_labels = sorted(list(set(predictions)))
        actual_labels = sorted(list(set(labels)))
        df_preds = pd.Categorical(predictions, pred_labels)
        df_labels = pd.Categorical(labels, actual_labels)
        confusion_matrix = pd.crosstab(df_labels, df_preds, rownames=['Actual'], colnames=['Predicted'],
                                       normalize='index').round(4)
        confusion_matrix = confusion_matrix.multiply(100)

        confusion_matrix = confusion_matrix[langs]
        confusion_matrix = confusion_matrix.loc[langs]
        mask = confusion_matrix.applymap(lambda x: lambdafunc(x))

        cmap = matplotlib.cm.get_cmap('jet')
        rgba = cmap(6)

        plt.figure(figsize=(100, 50))
        sns.set(font_scale=12)
        ax = sns.heatmap(confusion_matrix, annot=mask, xticklabels=langs, yticklabels=langs,
                         annot_kws={'size': 100}, linewidths=0.5, square=True, cmap="jet", fmt='', linecolor=rgba, cbar=False)

        ax.set_facecolor("blue")
        plt.savefig(base_path/"confusionmatrix_select.pdf", bbox_inches='tight')
        plt.show()

    def get_f1(self, type):
        predictions, labels = zip(*self.pred_labels)
        return f1_score(labels, predictions, average=type)

    def simple_percentage_print(self):
        language_to_scores_dict = {}
        for prediction, label in self.pred_labels:
            if (label in language_to_scores_dict.keys()):
                if (prediction == label):
                    language_to_scores_dict[label]['Correct'] = language_to_scores_dict[label]['Correct'] + 1
                else:
                    language_to_scores_dict[label]['Wrong'] = language_to_scores_dict[label]['Wrong'] + 1
            else:
                newDict = {}
                newDict['Wrong'] = 0
                newDict['Correct'] = 0
                language_to_scores_dict[label] = newDict
                if (prediction == label):
                    language_to_scores_dict[label]['Correct'] = language_to_scores_dict[label]['Correct'] + 1
                else:
                    language_to_scores_dict[label]['Wrong'] = language_to_scores_dict[label]['Wrong'] + 1

        for language in language_to_scores_dict:
            print(language)
            print(language_to_scores_dict[language])
            num_correct = language_to_scores_dict[language]['Correct']
            num_wrong = language_to_scores_dict[language]['Wrong']
            print("percentage correct: ", num_correct / (num_correct + num_wrong))
            print("----------------------------------------------------------")


def lambdafunc(x):
    if x < 1:
        return ""
    return "%.2f" % x


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
