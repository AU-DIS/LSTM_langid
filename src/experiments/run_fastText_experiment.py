from time import sleep
from sacred.observers import FileStorageObserver

from experiments.fasttext_experiment import fasttext_exp
from experiments.data_loading import data_loading_ingredient

@data_loading_ingredient.config
def update_cfg():
    train_path = "../../datasets/processed/UD20S"
    test_path = train_path

fasttext_exp.observers.append(FileStorageObserver('../experiments/FastText_experiments'))
num_lang_preds = 20
maximum_chars = 50
folds = 5
to_train = True



for i in range(folds):
    trainfolds = [(i + j + 1) % folds for j in range(folds - 1)]
    testfolds = [i]
    sleep(1)
    @data_loading_ingredient.config
    def update_cfg():
        train_path = "../../datasets/processed/UD20"
        test_path = "../../datasets/processed/UD20/sub_dataset"
        train_folds = trainfolds
        test_folds = testfolds
    config_updates = {
        'model_path': "../models/our_fasttext" + str(testfolds[0]) + ".bin",
        'num_lang_preds': num_lang_preds,
        'to_train': to_train,
    }
    fasttext_exp.run(config_updates=config_updates)


print("All experiments are completed")
