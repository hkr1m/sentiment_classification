import numpy as np
import torch
from torch.utils.data import Dataset
from gensim.models import KeyedVectors
import os, pickle

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

def get_word2id(word2id_path, data_paths, rebuild = False):
    print("Loading word2id...")
    word2id = {"_NULL_": 0}
    if rebuild or not os.path.exists(word2id_path) or os.path.getsize(word2id_path) == 0:
        for path in data_paths:
            with open(path, "r", encoding="utf-8") as file:
                for line in file:
                    words = line.strip().split()
                    for word in words[1:]:
                        if word not in word2id.keys():
                            word2id[word] = len(word2id)
        with open(word2id_path, "wb") as file:
            pickle.dump(word2id, file)
    else:
        with open(word2id_path, "rb") as file:
            word2id = pickle.load(file)
    return word2id

def get_embedding(embedding_path, word2vec_path, word2id, rebuild = False):
    print("Loading embedding...")
    if rebuild or not os.path.exists(embedding_path) or os.path.getsize(embedding_path) == 0:
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