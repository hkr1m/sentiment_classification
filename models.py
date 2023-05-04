import torch
from torch import nn

class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size,
                                      embedding_dim=config.embedding_dim)
        if config.emb:
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
        embed_x = self.embedding(x).permute(0, 2, 1) # (batch, embed, len)
        y = [conv(embed_x).squeeze(2) for conv in self.convs] # [(batch, feature_size)]
        y = torch.cat(y, dim=1) # (batch, feature_size_sum)
        return self.fc(self.dropout(y))

class _RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(_RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, input, hidden):
        hidden = self.i2h(input) + self.h2h(hidden)
        return torch.tanh(hidden)

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
        self.fwd_rnn = nn.ModuleList([_RNNCell(input_size=config.embedding_dim,
                                                   hidden_size=config.hidden_dim)])
        self.bwd_rnn = nn.ModuleList([_RNNCell(input_size=config.embedding_dim,
                                                    hidden_size=config.hidden_dim)])
        for _ in range(config.num_layers-1):
            self.fwd_rnn.append(_RNNCell(input_size=config.hidden_dim,
                                             hidden_size=config.hidden_dim))
            self.bwd_rnn.append(_RNNCell(input_size=config.hidden_dim,
                                              hidden_size=config.hidden_dim))
        
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc = nn.Sequential(nn.Linear(config.hidden_dim * 2, 64),
                                nn.Linear(64, config.num_class))

    def forward(self, x):
        batch_size, seq_len = x.size()
        embed_x = self.embedding(x).permute(1, 0, 2) # (len, batch, embed)
        h_fwd = [torch.zeros(batch_size, self.hidden_dim).to(x.device) for _ in range(self.num_layers)]
        h_bwd = [torch.zeros(batch_size, self.hidden_dim).to(x.device) for _ in range(self.num_layers)]
        for t in range(seq_len):
            t_fwd, t_bwd = [], []
            for layer in range(self.num_layers):
                if layer == 0:
                    t_fwd.append(self.fwd_rnn[layer](embed_x[t], h_fwd[layer]))
                    t_bwd.append(self.bwd_rnn[layer](embed_x[-t-1], h_bwd[layer]))
                else:
                    t_fwd.append(self.fwd_rnn[layer](t_fwd[layer-1], h_fwd[layer]))
                    t_bwd.append(self.bwd_rnn[layer](t_bwd[layer-1], h_bwd[layer]))
                h_fwd[layer] = self.dropout(t_fwd[layer])
                h_bwd[layer] = self.dropout(t_bwd[layer])
                
        hidden = torch.cat((h_fwd[-1], h_bwd[-1]), dim=1)
        return self.fc(hidden)
    
class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size,
                                      embedding_dim=config.embedding_dim)
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout_rate,
            bidirectional=True,
            batch_first=True
        )
        if config.emb:
            self.embedding.weight.data.copy_(config.embedding)
        self.fc = nn.Sequential(nn.Linear(config.hidden_dim * 2, 64),
                                nn.Linear(64, config.num_class))

    def forward(self, x):
        embed_x = self.embedding(x) # (batch, len, embed)
        _, (hidden, _) = self.lstm(embed_x) # (num_layers * directions, batch, embed)
        hidden = hidden.view(self.config.num_layers, -1, x.size(0), self.config.hidden_dim)
        hidden = torch.cat(hidden[-1].unbind(0), dim=-1)
        return self.fc(hidden)

class GRU(nn.Module):
    def __init__(self, config):
        super(GRU, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size,
                                      embedding_dim=config.embedding_dim)
        self.gru = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout_rate,
            bidirectional=True,
            batch_first=True
        )
        if config.emb:
            self.embedding.weight.data.copy_(config.embedding)
        self.fc = nn.Sequential(nn.Linear(config.hidden_dim * 2, 64),
                                nn.Linear(64, config.num_class))

    def forward(self, x):
        embed_x = self.embedding(x) # (batch, len, embed)
        _, hidden = self.gru(embed_x) # (num_layers * directions, batch, embed)
        hidden = hidden.view(self.config.num_layers, -1, x.size(0), self.config.hidden_dim)
        hidden = torch.cat(hidden[-1].unbind(0), dim=-1)
        return self.fc(hidden)