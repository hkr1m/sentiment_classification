# sentiment_classification

## 文件结构

`main.py` 主程序

`model.py` 五个模型（TextCNN，BiRNN，LSTM，GRU，MLP）的实现

`train.py` 训练、验证和测试的过程

`loader.py` 加载词到序列的对应（word2id）和词嵌入（embedding）

`data/` 存放 `train.txt`，`validation.txt`，`test.txt`

`src/` 存放预训练词嵌入模型 `wiki_word2vec_50.bin` 或者已经处理好的 `word2id.pkl` 和 `embedding.npy`

## 运行方式

在当前虚拟环境中安装 Python 依赖库

```bash
pip install numpy torch scipy gensim tensorboard
```

在项目路径下，运行

```bash
python main.py
```

可选参数及解释如下

```plain
usage: main.py [-h] [-m {TextCNN,BiRNN,LSTM,GRU,MLP}] [-l MODEL_LOAD_PATH]
               [-s MODEL_SAVE_PATH] [--device {cuda,mps,cpu,auto}]
               [-b BATCH_SIZE] [--lr LEARNING_RATE] [-e EPOCHS]
               [-d DROPOUT_RATE] [--unemb] [-len MAX_SENT_LEN]
               [-f FEATURE_SIZE] [-w [WINDOW_SIZES ...]] [-n NUM_LAYERS]
               [-hd HIDDEN_DIM] [-hs [HIDDEN_SIZES ...]]

Sentiment Classification

optional arguments:
  -h, --help            show this help message and exit
  -m {TextCNN,BiRNN,LSTM,GRU,MLP}, --model {TextCNN,BiRNN,LSTM,GRU,MLP}
                        指定模型，默认使用 TextCNN
  -l MODEL_LOAD_PATH, --model-load-path MODEL_LOAD_PATH
                        指定模型参数加载路径
  -s MODEL_SAVE_PATH, --model-save-path MODEL_SAVE_PATH
                        指定模型参数保存路径
  --device {cuda,mps,cpu,auto}
                        指定 PyTorch 使用的设备
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        指定 batch 大小
  --lr LEARNING_RATE    指定 learning rate
  -e EPOCHS, --epochs EPOCHS
                        指定要运行的 epoch 数
  -d DROPOUT_RATE, --dropout-rate DROPOUT_RATE
                        指定训练期间使用的 dropout rate
  --unemb               不采用预训练的 embedding
  -len MAX_SENT_LEN, --max-sent-len MAX_SENT_LEN
                        指定最大截取或补全的句子长度
  -f FEATURE_SIZE, --feature-size FEATURE_SIZE
                        指定每个卷积核的特征数，用于 TextCNN
  -w [WINDOW_SIZES ...], --window-sizes [WINDOW_SIZES ...]
                        指定卷积核大小，用于 TextCNN
  -n NUM_LAYERS, --num-layers NUM_LAYERS
                        指定隐含层数，用于 BiRNN / LSTM / GRU
  -hd HIDDEN_DIM, --hidden-dim HIDDEN_DIM
                        指定每个隐含层大小，用于 BiRNN / LSTM / GRU
  -hs [HIDDEN_SIZES ...], --hidden-sizes [HIDDEN_SIZES ...]
                        指定隐含层大小，用于 MLP
```
