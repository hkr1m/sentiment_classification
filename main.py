import models, train, loader
import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse

class Config(object):
    def set(self):
        if self.device == "auto":
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        print(f"Using {self.device} device")
        if self.emb:
            self.embedding = loader.get_embedding(embedding_path=embedding_path,
                                                  word2vec_path=word2vec_path,
                                                  word2id=word2id).to(self.device)
            self.vocab_size = self.embedding.size(0)
            self.embedding_dim = self.embedding.size(1)
        else:
            self.vocab_size = len(word2id)
            self.embedding_dim = 50
        self.num_class = 2

class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description="Sentiment Classification")
        self.add_argument("-m", "--model", choices=["TextCNN", "BiRNN", "LSTM", "GRU"], default="TextCNN")
        self.add_argument("-l", "--model-load-path", dest="model_load_path", default="")
        self.add_argument("-s", "--model-save-path", dest="model_save_path", default="")
        self.add_argument("--device", choices=["cuda", "mps", "cpu", "auto"], default="auto")
        self.add_argument("-b", "--batch-size", dest="batch_size", type=int, default=50)
        self.add_argument("--lr", dest="learning_rate", type=float, default=1e-3)
        self.add_argument("-e", "--epochs", type=int, default=10)
        self.add_argument("-d", "--dropout-rate", dest="dropout_rate", type=float, default=0.3)
        self.add_argument("--unemb", dest="emb", action="store_false", default=True)
        self.add_argument("-len", "--max-sent-len", dest="max_sent_len", type=int, default=120)
        self.add_argument("-f", "--feature-size", dest="feature_size", type=int, default=20)
        self.add_argument("-w", "--window-sizes", dest="window_sizes", type=int, nargs="*", default=[3, 5, 7])
        self.add_argument("-n", "--num-layers", dest="num_layers", type=int, default=2)
        self.add_argument("-hd", "--hidden-dim", dest="hidden_dim", type=int, default=100)

if __name__ == "__main__":
    train_path = "data/train.txt"
    validation_path = "data/validation.txt"
    test_path = "data/test.txt"
    word2id_path = "src/word2id.pkl"
    embedding_path = "src/embedding.npy"
    word2vec_path = "src/wiki_word2vec_50.bin"
    
    config = Config()
    Parser().parse_args(namespace=config)
    word2id = loader.get_word2id(word2id_path=word2id_path,
                          data_paths=[train_path, validation_path, test_path])
    config.set()

    training_data = loader.SemtimentDataset(train_path, word2id, config.max_sent_len)
    validation_data = loader.SemtimentDataset(validation_path, word2id, config.max_sent_len)
    testing_data = loader.SemtimentDataset(test_path, word2id, config.max_sent_len)
    train_dataloader = DataLoader(training_data, batch_size=config.batch_size, shuffle=True)
    valid_dataloder = DataLoader(validation_data, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(testing_data, batch_size=config.batch_size, shuffle=True)

    if config.model == "TextCNN":
        model = models.TextCNN(config).to(config.device)
    elif config.model == "BiRNN":
        model = models.BiRNN(config).to(config.device)
    elif config.model == "LSTM":
        model = models.LSTM(config).to(config.device)
    elif config.model == "GRU":
        model = models.GRU(config).to(config.device)

    if len(config.model_load_path):
        print(f"Loading weights from {config.model_load_path}")
        model.load_state_dict(torch.load(config.model_load_path))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

    print(f"Training with model {config.model}")

    for t in range(config.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train.train_loop(train_dataloader, model, loss_fn, optimizer, scheduler)
        print("Validating...")
        train.test_loop(valid_dataloder, model, loss_fn)
        print("Testing...")
        train.test_loop(valid_dataloder, model, loss_fn)

    print("Done!")
    if len(config.model_save_path):
        torch.save(model.state_dict(), config.model_save_path)
        print(f"Weights saved in {config.model_save_path}")