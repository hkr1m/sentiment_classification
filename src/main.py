import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from gensim.models import KeyedVectors
import os
import pickle
from sklearn.metrics import f1_score

class Config(object):
    def __init__(self, word2vec):
        self.load_model = False
        self.learning_rate = 1e-3
        self.batch_size = 50
        self.epochs = 100
        self.dropout_rate = 0.3
        self.num_class = 2
        self.pretrained_embed = True
        self.embedding = word2vec
        self.vocab_size = embedding.shape[0]
        self.embedding_dim = embedding.shape[1]
        self.feature_size = 20
        self.window_sizes = [3, 5, 7]
        self.max_sent_len = 120
        self.num_layers = 2
        self.hidden_dim = 100

class SemtimentDataset(Dataset):
    def __init__(self, data_path, word2id, max_sent_len):
        self.sents = []
        self.lables = []
        with open(data_path, "r", encoding="utf-8") as file:
            for line in file:
                words = line.strip().split()
                sent = torch.zeros(max_sent_len, dtype=torch.long)
                idx = 0
                for word in words[1:]:
                    if (idx == max_sent_len):
                        break
                    try:
                        sent[idx] = word2id[word]
                        idx = idx + 1
                    except KeyError:
                        pass
                self.sents.append(sent)
                self.lables.append(int(words[0]))

    def __len__(self):
        return len(self.lables)

    def __getitem__(self, idx):
        return self.sents[idx], self.lables[idx]

class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size,
                                      embedding_dim=config.embedding_dim)
        if config.pretrained_embed:
            self.embedding.weight.data.copy_(config.embedding)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=config.embedding_dim,
                                     out_channels=config.feature_size,
                                     kernel_size=ker),
                           nn.ReLU(),
                           nn.MaxPool1d(kernel_size=config.max_sent_len-ker+1))
            for ker in config.window_sizes
        ])
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc = nn.Linear(in_features=config.feature_size*len(config.window_sizes),
                            out_features=config.num_class)
    
    def forward(self, x):
        embed_x = self.embedding(x) # (batch_size, max_len_sent, embedding_dim)
        embed_x = embed_x.permute(0, 2, 1) # (batch_size, embedding_dim, max_sent_len)
        y = [conv(embed_x).squeeze(2) for conv in self.convs] # [(batch_size, feature_size)]
        y = torch.cat(y, dim=1) # (batch_size, feature_size_sum)
        return self.fc(self.dropout(y))

# class RNN(nn.Module):
#     def __init__(self, config):
#         super(RNN, self).__init__()
#         self.config = config
#         self.embedding = nn.Embedding(num_embeddings=config.vocab_size,
#                                       embedding_dim=config.embedding_dim)
#         if config.pretrained_embed:
#             self.embedding.weight.data.copy_(config.embedding)
#         self.i2h = nn.Linear(in_features=config.embedding_dim,
#                              out_features=config.hidden_dim)
#         self.h2h = nn.Linear(in_features=config.hidden_dim,
#                              out_features=config.hidden_dim)
#         self.h2o = nn.Linear(in_features=config.hidden_dim,
#                              out_features=64)
#         self.fc = nn.Linear(64, 2)

#     def step(self, input, hidden):
#         hidden = torch.tanh(self.i2h(input) + self.h2h(hidden))
#         return hidden
    
#     def forward(self, x):
#         embed_x = self.embedding(x).permute(1, 0, 2) # (len, batch, embed)
#         hidden = torch.zeros(x.size(0), self.config.hidden_dim).to(device)
#         for word in embed_x:
#             hidden = self.step(word, hidden)
#         return self.fc(self.h2o(hidden))
    
class RNN(nn.Module):
    def __init__(self, config, rnn_model):
        super(RNN, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size,
                                      embedding_dim=config.embedding_dim)
        self.rnn = rnn_model
        if config.pretrained_embed:
            self.embedding.weight.data.copy_(config.embedding)
        self.decoder = nn.Linear(2 * config.hidden_dim, 64)
        self.fc1 = nn.Linear(64, config.num_class)

    def forward(self, x):
        embed_x = self.embedding(x) # (batch, len, embed)
        output, (hidden, cell) = self.rnn(embed_x) # (num_layers * directions, batch, embed)
        hidden = hidden.view(self.config.num_layers, -1, x.size(0), self.config.hidden_dim)
        hidden = torch.cat(hidden[-1].unbind(0), dim=-1)
        return self.fc1(self.decoder(hidden))

class RNN_LSTM(RNN):
    def __init__(self, config):
        super(RNN_LSTM, self).__init__(config, rnn_model=nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout_rate,
            bidirectional=True,
            batch_first=True
        ))

def get_word2id(wordid_path, data_paths = []):
    print("load word2id...")
    word2id = {"_NULL_": 0}
    if len(data_paths) > 0:
        for path in data_paths:
            with open(path, "r", encoding="utf-8") as file:
                for line in file:
                    words = line.strip().split()
                    for word in words[1:]:
                        if word not in word2id.keys():
                            word2id[word] = len(word2id)
        with open(wordid_path, "wb") as file:
            pickle.dump(word2id, file)
    else:
        with open(wordid_path, "rb") as file:
            word2id = pickle.load(file)
    return word2id

def get_embedding(embedding_path, word2vec_path = "", word2id = {}):
    print("load embedding...")
    if len(word2vec_path) > 0:
        vocab_size = len(word2id)
        model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        embedding = np.random.uniform(-1., 1., [vocab_size, model.vector_size])
        for word in word2id.keys():
            try:
                embedding[word2id[word]] = model[word]
            except KeyError:
                pass
        np.save(embedding_path, embedding)
    else:
        embedding = np.load(embedding_path)
    return torch.Tensor(embedding)

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    scheduler.step()

# def test_loop(dataloader, model, loss_fn):
#     model.eval()
#     # 验证过程
#     val_loss, val_acc = 0.0, 0.0
#     count, correct = 0, 0
#     full_true = []
#     full_pred = []
#     for _, (x, y) in enumerate(dataloader):
#         x, y = x.to(device), y.to(device)
#         output = model(x)
#         loss = loss_fn(output, y)
#         val_loss += loss.item()
#         correct += (output.argmax(1) == y).float().sum().item()
#         count += len(x)
#         full_true.extend(y.cpu().numpy().tolist())
#         full_pred.extend(output.argmax(1).cpu().numpy().tolist())
#     val_loss *= config.batch_size
#     val_loss /= len(dataloader.dataset)
#     val_acc = correct / count
#     f1 = f1_score(np.array(full_true), np.array(full_pred), average="binary")
#     print(f"val_loss: {val_loss:>4f}, val_acc: {val_acc:>4f}, f1: {f1:>4f}")
#     return val_loss, val_acc, f1

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0.
    FP, TP, FN, TN, P, N = 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            tf = pred.argmax(1) == y
            for i in range(tf.size(0)):
                if y[i] == 0:
                    P = P + 1
                    if tf[i]: TP = TP + 1
                    else: FN = FN + 1
                else:
                    N = N + 1
                    if tf[i]: TN = TN + 1
                    else: FP = FP + 1
    test_loss /= num_batches
    try:
        precision = TP / (TP+FP)
        recall = TP / P
        accuracy = (TP+TN) / (P+N)
        F_measure = 2 / (1/precision + 1/recall)
        print(f"Test Error: \n\
  Precision: {precision:>6f}, Recall: {recall:>6f} \n\
  Accuracy: {accuracy:>6f}, F_measure: {F_measure:>6f}")
    except ZeroDivisionError:
        print("Divide zero in F-mesure calculation")

if __name__ == "__main__":
    file_path = os.path.abspath(os.path.realpath(__file__))
    root_path = file_path[:file_path.find("src")]
    train_path =os.path.join(root_path, "data/train.txt")
    validation_path = os.path.join(root_path, "data/validation.txt")
    test_path = os.path.join(root_path, "data/test.txt")
    word2id_path = os.path.join(root_path, "src/word2id.pkl")
    embedding_path = os.path.join(root_path, "src/embedding.npy")
    word2vec_path = os.path.join(root_path, "src/wiki_word2vec_50.bin")
    model_path = os.path.join(root_path, "src/model.pth")
    word2id = get_word2id(wordid_path=word2id_path,
                          data_paths=[train_path, validation_path, test_path])
    embedding = get_embedding(embedding_path=embedding_path,
                              word2vec_path=word2vec_path,
                              word2id=word2id)
    config = Config(embedding)
    training_data = SemtimentDataset(train_path, word2id, config.max_sent_len)
    validation_data = SemtimentDataset(validation_path, word2id, config.max_sent_len)
    testing_data = SemtimentDataset(test_path, word2id, config.max_sent_len)
    train_dataloader = DataLoader(training_data, batch_size=config.batch_size, shuffle=True)
    valid_dataloder = DataLoader(validation_data, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(testing_data, batch_size=config.batch_size, shuffle=True)

    device = (
        "cuda"
        if torch.cuda.is_available()
        #else "mps"
        #if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = RNN_LSTM(config).to(device)
    if config.load_model:
        model.load_state_dict(torch.load(model_path))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

    for t in range(config.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(valid_dataloder, model, loss_fn)

    print("Done!")

    torch.save(model.state_dict(), model_path)
    print("Model saved.")