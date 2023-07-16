import mindspore.nn as nn
import mindspore.ops as ops


class Encoder(nn.Cell):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)  # Embedding层
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)  # 双向GRU层
        self.fc = nn.Dense(enc_hid_dim * 2, dec_hid_dim)  # 全连接层

        self.dropout = nn.Dropout(p=dropout)  # dropout，防止过拟合

    def construct(self, src, src_len):

        # 将输入源序列转化为向量，并进行暂退（dropout）
        # shape = [src len, batch size, emb dim]
        embedded = self.dropout(self.embedding(src))
        # 计算输出
        # shape = [src len, batch size, enc hid dim*2]
        outputs, hidden = self.rnn(embedded, seq_length=src_len)
        # 为适配解码器，合并两个上下文函数
        # shape = [batch size, dec hid dim]
        hidden = ops.tanh(self.fc(ops.concat((hidden[-2, :, :], hidden[-1, :, :]), axis=1)))

        return outputs, hidden
