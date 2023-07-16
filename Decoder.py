import mindspore.nn as nn
import mindspore.ops as ops


class Decoder(nn.Cell):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Dense((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def construct(self, inputs, hidden, encoder_outputs, mask):

        # 为输入增加额外维度
        # shape = [1, batch size]
        inputs = inputs.expand_dims(0)

        # 输入词的embedding输出， d(y_t)
        # shape = [1, batch size, emb dim]
        embedded = self.dropout(self.embedding(inputs))

        # 注意力权重向量, a_t
        # shape = [batch size, src len]
        a = self.attention(hidden, encoder_outputs, mask)

        # 为注意力权重增加额外维度
        # shape = [batch size, 1, src len]
        a = a.expand_dims(1)

        # 将编码器隐藏状态中的第1、2维度进行交换
        # shape = [batch size, src len, enc hid dim * 2]
        encoder_outputs = encoder_outputs.transpose(1, 0, 2)

        # 计算w_t
        # shape = [batch size, 1, enc hid dim * 2]
        weighted = ops.bmm(a, encoder_outputs)

        # 将w_t的第1、2维度进行交换
        # shape = [1, batch size, enc hid dim * 2]
        weighted = weighted.transpose(1, 0, 2)

        # rnn_input shape = [1, batch size, (enc hid dim * 2) + emb dim]
        # output shape = [seq len = 1, batch size, dec hid dim * n directions]
        # hidden shape = [n layers (1) * n directions (1) = 1, batch size, dec hid dim]
        rnn_input = ops.concat((embedded, weighted), axis=2)
        output, hidden = self.rnn(rnn_input, hidden.expand_dims(0))

        # 去除多余的第1维度
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        # 将embedded，weighted和hidden堆叠起来，并输入线性层，预测下一个词
        # shape = [batch size, output dim]
        prediction = self.fc_out(ops.concat((output, weighted, embedded), axis=1))

        return prediction, hidden.squeeze(0), a.squeeze(1)
