import mindspore.nn as nn
import mindspore.ops as ops


class Attention(nn.Cell):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        # attention线性层
        self.attn = nn.Dense((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        # v， 用不带有bias的线性层表示
        # shape = [1, dec hid dim]
        self.v = nn.Dense(dec_hid_dim, 1, has_bias=False)

    def construct(self, hidden, encoder_outputs, mask):

        src_len = encoder_outputs.shape[0]

        # 重复解码器隐藏状态src len次，对齐维度
        # shape = [batch size, src len, dec hid dim]
        hidden = ops.tile(hidden.expand_dims(1), (1, src_len, 1))

        # 将编码器输出中的第1、2维度进行交换，对齐维度
        # shape = [batch size, src len, enc hid dim*2]
        encoder_outputs = encoder_outputs.transpose(1, 0, 2)

        # 计算E_t
        # shape = [batch size, src len, dec hid dim]
        energy = ops.tanh(self.attn(ops.concat((hidden, encoder_outputs), axis=2)))

        # 计算v * E_t
        # shape = [batch size, src len]
        attention = self.v(energy).squeeze(2)

        # 不需要考虑序列中<pad>占位符的注意力权重
        attention = attention.masked_fill(mask == 0, -1e10)

        return ops.softmax(attention, axis=1)
