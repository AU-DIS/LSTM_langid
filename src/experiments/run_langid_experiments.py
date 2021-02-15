from sacred.observers import FileStorageObserver
from experiments.LangID_experiment import ex
from experiments.data_loading import data_loading_ingredient



ex.observers.append(FileStorageObserver('../experiments/langid_experimentsmixed'))
folds = 5

for i in range(folds):
    testfolds = [i]
    @data_loading_ingredient.config
    def update_cfg():
        train_path = "../../datasets/processed/mixed"
        test_path = "../../datasets/processed/mixed/sub_dataset"
        train_folds = []
        test_folds = testfolds
    ex.run()


print("All experiments are completed")
