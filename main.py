import model, train, loader
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
        self.add_argument("-m", "--model", choices=["TextCNN", "BiRNN", "LSTM", "GRU", "MLP"], default="TextCNN", help="指定模型，默认使用 TextCNN")
        self.add_argument("-l", "--model-load-path", dest="model_load_path", default="", help="指定模型参数加载路径")
        self.add_argument("-s", "--model-save-path", dest="model_save_path", default="", help="指定模型参数保存路径")
        self.add_argument("--device", choices=["cuda", "mps", "cpu", "auto"], default="auto", help="指定 PyTorch 使用的设备")
        self.add_argument("-b", "--batch-size", dest="batch_size", type=int, default=50, help="指定 batch 大小")
        self.add_argument("--lr", dest="learning_rate", type=float, default=1e-3, help="指定 learning rate")
        self.add_argument("-e", "--epochs", type=int, default=10, help="指定要运行的 epoch 数")
        self.add_argument("-d", "--dropout-rate", dest="dropout_rate", type=float, default=0.5, help="指定训练期间使用的 dropout rate")
        self.add_argument("--unemb", dest="emb", action="store_false", default=True, help="不采用预训练的 embedding")
        self.add_argument("-len", "--max-sent-len", dest="max_sent_len", type=int, default=120, help="指定最大截取或补全的句子长度")
        self.add_argument("-f", "--feature-size", dest="feature_size", type=int, default=50, help="指定每个卷积核的特征数，用于 TextCNN")
        self.add_argument("-w", "--window-sizes", dest="window_sizes", type=int, nargs="*", default=[3, 4, 5], help="指定卷积核大小，用于 TextCNN")
        self.add_argument("-n", "--num-layers", dest="num_layers", type=int, default=2, help="指定隐含层数，用于 BiRNN")
        self.add_argument("-hd", "--hidden-dim", dest="hidden_dim", type=int, default=100, help="指定每个隐含层大小，用于 BiRNN")
        self.add_argument("-hs", "--hidden-sizes", dest="hidden_sizes", type=int, nargs="*", default=[512, 512], help="指定隐含层大小，用于 MLP")

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
    valid_dataloader = DataLoader(validation_data, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(testing_data, batch_size=config.batch_size, shuffle=True)

    if config.model == "TextCNN":
        model = model.TextCNN(config).to(config.device)
    elif config.model == "BiRNN":
        model = model.BiRNN(config).to(config.device)
    elif config.model == "LSTM":
        model = model.LSTM(config).to(config.device)
    elif config.model == "GRU":
        model = model.GRU(config).to(config.device)
    elif config.model == "MLP":
        model = model.MLP(config).to(config.device)

    if len(config.model_load_path):
        print(f"Loading weights from {config.model_load_path}")
        model.load_state_dict(torch.load(config.model_load_path))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

    print(f"Training with model {config.model}")

    writer = SummaryWriter("log")

    for epoch in range(config.epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        loss = train.train_loop(train_dataloader, model, loss_fn, optimizer, scheduler)
        writer.add_scalar("train loss", loss, epoch + 1)
        print("Validating...")
        loss, precision, recall, accuracy, F_measure = train.test_loop(valid_dataloader, model, loss_fn)
        print(f"Valid loss: {loss:>6f}\n\
  Precision: {precision:>6f}, Recall: {recall:>6f} \n\
  Accuracy: {accuracy:>6f}, F_measure: {F_measure:>6f}")
        writer.add_scalar("valid loss", loss, epoch + 1)
        writer.add_scalar("valid precision", precision, epoch + 1)
        writer.add_scalar("valid recall", recall, epoch + 1)
        writer.add_scalar("valid accuracy", accuracy, epoch + 1)
        writer.add_scalar("valid F1 score", F_measure, epoch + 1)
        print("Testing...")
        loss, precision, recall, accuracy, F_measure = train.test_loop(test_dataloader, model, loss_fn)
        print(f"Test loss: {loss:>6f}\n\
  Precision: {precision:>6f}, Recall: {recall:>6f} \n\
  Accuracy: {accuracy:>6f}, F_measure: {F_measure:>6f}")
        writer.add_scalar("test loss", loss, epoch + 1)
        writer.add_scalar("test precision", precision, epoch + 1)
        writer.add_scalar("test recall", recall, epoch + 1)
        writer.add_scalar("test accuracy", accuracy, epoch + 1)
        writer.add_scalar("test F1 score", F_measure, epoch + 1)

    writer.close()

    print("Done!")
    if len(config.model_save_path):
        torch.save(model.state_dict(), config.model_save_path)
        print(f"Weights saved in {config.model_save_path}")