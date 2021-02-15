from time import sleep
from sacred.observers import FileStorageObserver
from experiments.data_loading import data_loading_ingredient
from src.experiments.LSTM_experiment import LSTM_exp

LSTM_exp.observers.append(FileStorageObserver('../experiments/LSTM_experimentmixed'))
epochs = 25
embedding_dims = [150]
hidden_dims = [150]
weight_decay = 0.00001
seed = 1
lr = 0.1
batch_size = 64
folds = 5
# sgd and adam are possible optimizers
optimizer = "adam"
count = 1
total = len(embedding_dims) * len(hidden_dims) * folds

for i in range(folds):
    sleep(2)
    @data_loading_ingredient.config
    def update_cfg():
        train_path = "../../datasets/processed/mixed"
        test_path = "../../datasets/processed/mixed/sub_dataset"
        train_folds = [(i + j + 1) % folds for j in range(folds - 1)]
        test_folds = [i]
    for emb_dim in embedding_dims:
        for hid_dim in hidden_dims:
            config_updates = {
                'epochs': epochs,
                'embedding_dim': emb_dim,
                'hidden_dim': hid_dim,
                'weight_decay': weight_decay,
                'lr': lr,
                'optimizer': optimizer,
                'seed': seed,
                'batch_size': batch_size,
            }
            print("Running experiment {0} of {1} with config {2}".format(count, total, config_updates))
            count += 1
            LSTM_exp.run(config_updates=config_updates)
print("All experiments are completed")
