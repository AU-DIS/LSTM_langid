from time import sleep

from sacred.observers import FileStorageObserver
from experiments.fasttext_experiment import fasttext_exp
from experiments.data_loading import data_loading_ingredient

folds = 5
fasttext_exp.observers.append(FileStorageObserver('fastText_pretrained_experimentUD20'))
for i in range(folds):
    sleep(2)
    @data_loading_ingredient.config
    def update_cfg():
        train_path = "../../datasets/processed/UD20"
        test_path = "../../datasets/processed/UD20/sub_dataset"
        train_folds = [(i + j + 1) % folds for j in range(folds - 1)]
        test_folds = [i]
    @fasttext_exp.config
    def config():
        model_path = "../../models/lid.176.bin"
        num_lang_preds = 300
        to_train = False
    print(f"Running experiment {i}")
    fasttext_exp.run()
