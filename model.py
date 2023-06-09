import torch
from torch import nn

class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size,
                                      embedding_dim=config.embedding_dim)
        if config.emb: # 加载预训练 embedding
            self.embedding.weight.data.copy_(config.embedding)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=config.embedding_dim,
                                    out_channels=config.feature_size,
                                    kernel_size=ker),
                           nn.ReLU(),
                           nn.MaxPool1d(kernel_size=config.max_sent_len-ker+1))
            for ker in config.window_sizes
        ]) # 卷积核，用 ReLU 做激活函数，后接 1-最大池化
        self.dropout = nn.Dropout(config.dropout_rate) # dropout 层
        self.fc = nn.Linear(in_features=config.feature_size*len(config.window_sizes),
                            out_features=config.num_class) # 全连接
    
    def forward(self, x):
        embed_x = self.embedding(x).permute(0, 2, 1) # (batch, embed, len)
        y = [conv(embed_x).squeeze(2) for conv in self.convs] # [(batch, feature_size)]
        y = torch.cat(y, dim=1) # (batch, feature_size_sum)
        return self.fc(self.dropout(y))

class _RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(_RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size) # 输入层到隐含层全连接网络
        self.h2h = nn.Linear(hidden_size, hidden_size) # 隐含层到隐含层全连接网络
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden):
        hidden = torch.tanh(self.i2h(input) + self.h2h(hidden))
        hidden = self.dropout(hidden)
        return hidden

class BiRNN(nn.Module):
    def __init__(self, config):
        super(BiRNN, self).__init__()
        self.num_layers = config.num_layers
        self.hidden_dim = config.hidden_dim
        self.config = config
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size,
                                      embedding_dim=config.embedding_dim)
        if config.emb:
            self.embedding.weight.data.copy_(config.embedding)
        
        # 初始化前向 RNN 和后向 RNN，第一层的输入大小为词向量维数
        self.fwd_rnn = nn.ModuleList([_RNNCell(input_size=config.embedding_dim,
                                               hidden_size=config.hidden_dim,
                                               dropout=config.dropout_rate)])
        self.bwd_rnn = nn.ModuleList([_RNNCell(input_size=config.embedding_dim,
                                               hidden_size=config.hidden_dim,
                                               dropout=config.dropout_rate)])
        for _ in range(config.num_layers-1):
            self.fwd_rnn.append(_RNNCell(input_size=config.hidden_dim,
                                         hidden_size=config.hidden_dim,
                                         dropout=config.dropout_rate))
            self.bwd_rnn.append(_RNNCell(input_size=config.hidden_dim,
                                         hidden_size=config.hidden_dim,
                                         dropout=config.dropout_rate))
        
        self.fc = nn.Sequential(nn.Linear(config.hidden_dim * 2, 64),
                                nn.Linear(64, config.num_class)) # 两层全连接

    def forward(self, x):
        batch_size, seq_len = x.size()
        embed_x = self.embedding(x).permute(1, 0, 2) # (len, batch, embed)
        h_fwd = [torch.zeros(batch_size, self.hidden_dim).to(x.device) for _ in range(self.num_layers)] # 前向隐含层
        h_bwd = [torch.zeros(batch_size, self.hidden_dim).to(x.device) for _ in range(self.num_layers)] # 后向隐含层
        for t in range(seq_len):
            for layer in range(self.num_layers):
                if layer == 0:
                    h_fwd[layer] = self.fwd_rnn[layer](embed_x[t], h_fwd[layer])
                    h_bwd[layer] = self.bwd_rnn[layer](embed_x[seq_len-t-1], h_bwd[layer])
                else:
                    h_fwd[layer] = self.fwd_rnn[layer](h_fwd[layer-1], h_fwd[layer])
                    h_bwd[layer] = self.bwd_rnn[layer](h_bwd[layer-1], h_bwd[layer])

        hidden = torch.cat((h_fwd[-1], h_bwd[-1]), dim=1)
        return self.fc(hidden)
    
class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size,
                                      embedding_dim=config.embedding_dim)
        if config.emb:
            self.embedding.weight.data.copy_(config.embedding)
        self.lstm = nn.LSTM( # 使用 Pytorch 的 LSTM 模型
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout_rate,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Sequential(nn.Linear(config.hidden_dim * 2, 64),
                                nn.Linear(64, config.num_class))

    def forward(self, x):
        embed_x = self.embedding(x) # (batch, len, embed)
        _, (hidden, _) = self.lstm(embed_x) # (num_layers * directions, batch, embed)
        hidden = hidden.view(self.config.num_layers, -1, x.size(0), self.config.hidden_dim)
        # (num_layer, directions, batch, embed)
        hidden = torch.cat(hidden[-1].unbind(0), dim=-1) # (batch, hidden_dim * 2)
        return self.fc(hidden)

class GRU(nn.Module):
    def __init__(self, config):
        super(GRU, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size,
                                      embedding_dim=config.embedding_dim)
        if config.emb:
            self.embedding.weight.data.copy_(config.embedding)
        self.gru = nn.GRU( # 使用 Pytorch 的 GRU 模型
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout_rate,
            bidirectional=True,
            batch_first=True
        ) 
        self.fc = nn.Sequential(nn.Linear(config.hidden_dim * 2, 64),
                                nn.Linear(64, config.num_class))

    def forward(self, x):
        embed_x = self.embedding(x)
        _, hidden = self.gru(embed_x)
        hidden = hidden.view(self.config.num_layers, -1, x.size(0), self.config.hidden_dim)
        hidden = torch.cat(hidden[-1].unbind(0), dim=-1)
        return self.fc(hidden)

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size,
                                      embedding_dim=config.embedding_dim)
        if config.emb:
            self.embedding.weight.data.copy_(config.embedding)

        self.flatten = nn.Flatten() # 将句子长度 * 词向量维数的输入打成一维
        num_layer = len(config.hidden_sizes)
        layer_sizes = [config.max_sent_len*config.embedding_dim] + config.hidden_sizes
        self.mlp = nn.Sequential() # 多个全连接层堆叠，用 ReLU 做激活函数
        for i in range(num_layer):
            self.mlp.extend([nn.Linear(layer_sizes[i], layer_sizes[i+1]),
                             nn.ReLU(),
                             nn.Dropout(config.dropout_rate)])
        self.mlp.append(nn.Linear(layer_sizes[-1], config.num_class))
    
    def forward(self, x):
        embed_x = self.embedding(x)
        flatten = self.flatten(embed_x)
        return self.mlp(flatten)
