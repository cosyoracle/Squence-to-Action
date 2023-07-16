import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class Seq2Act(nn.Cell):
    def __init__(self, encoder, decoder, src_pad_idx, teacher_forcing_ratio):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.teacher_forcing_ratio = teacher_forcing_ratio  # 使用teacher forcing的可能性

    def create_mask(self, src):
        mask = (src != self.src_pad_idx).astype(mindspore.int32).swapaxes(1, 0)
        return mask

    def construct(self, src, src_len, trg, trg_len=None):
        if trg_len is None:
            trg_len = trg.shape[0]

        # 存储解码器输出
        outputs = []

        # 编码器（encoder）：
        # 输入：源序列、源序列长度
        # 输出1：编码器中所有前向与反向RNN 的隐藏状态 encoder_outputs
        # 输出2：编码器中前向与反向RNN中最后时刻的隐藏状态放入线性层后的输出 hidden
        encoder_outputs, hidden = self.encoder(src, src_len)

        # 解码器的第一个输入是表示序列开始的占位符<bos>
        inputs = trg[0]

        # 标记源序列中<pad>占位符的位置
        # shape = [batch size, src len]
        mask = self.create_mask(src)

        for t in range(1, trg_len):

            # 解码器（decoder）：
            # 输入：源句子序列 inputs、前一时刻的隐藏状态 hidden、编码器所有前向与反向RNN的隐藏状态
            # 标明每个句子中的<pad>，方便计算注意力权重时忽略该部分
            # 输出：预测结果 output、新的隐藏状态 hidden、注意力权重（忽略）
            output, hidden, _ = self.decoder(inputs, hidden, encoder_outputs, mask)

            # 将预测结果放入之前的存储中
            outputs.append(output)

            # 找出对应预测概率最大的词元
            top1 = output.argmax(1).astype(mindspore.int32)

            if self.training:
                # 如果目前为模型训练状态，则按照之前设定的概率使用teacher forcing
                minval = Tensor(0, mindspore.float32)
                maxval = Tensor(1, mindspore.float32)
                teacher_force = ops.uniform((1,), minval, maxval) < self.teacher_forcing_ratio
                # 如使用teacher forcing，则将目标序列中对应的词元作为下一个输入
                # 如不使用teacher forcing，则将预测结果作为下一个输入
                inputs = trg[t] if teacher_force else top1
            else:
                inputs = top1

        # 将所有输出整合为tensor
        outputs = ops.stack(outputs, axis=0)

        return outputs
