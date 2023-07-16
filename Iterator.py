import mindspore


class Iterator:
    """创建数据迭代器"""
    def __init__(self, dataset, de_vocab, en_vocab, batch_size, max_len=32, drop_reminder=False):
        self.dataset = dataset
        self.de_vocab = de_vocab
        self.en_vocab = en_vocab

        self.batch_size = batch_size
        self.max_len = max_len
        self.drop_reminder = drop_reminder

        length = len(self.dataset) // batch_size
        self.len = length if drop_reminder else length + 1  # 批量数量

    def __call__(self):
        def pad(idx_list, vocab, max_len):
            idx_pad_list, idx_len = [], []
            # 当前序列度超过最大长度时，将超出的部分丢弃；当前序列长度小于最大长度时，用占位符补齐
            for i in idx_list:
                if len(i) > max_len - 2:
                    idx_pad_list.append(
                        [vocab.bos_idx] + i[:max_len-2] + [vocab.eos_idx]
                    )
                    idx_len.append(max_len)
                else:
                    idx_pad_list.append(
                        [vocab.bos_idx] + i + [vocab.eos_idx] + [vocab.pad_idx] * (max_len - len(i) - 2)
                    )
                    idx_len.append(len(i) + 2)
            return idx_pad_list, idx_len

        def sort_by_length(src, trg):
            data = zip(src, trg)
            data = sorted(data, key=lambda t: len(t[0]), reverse=True)
            return zip(*list(data))

        def encode_and_pad(batch_data, max_len):
            # 将当前批量数据中的词元转化为索引
            src_data, trg_data = zip(*batch_data)
            src_idx = [self.de_vocab.encode(i) for i in src_data]
            trg_idx = [self.en_vocab.encode(i) for i in trg_data]

            # 统一序列长度
            src_idx, trg_idx = sort_by_length(src_idx, trg_idx)
            src_idx_pad, src_len = pad(src_idx, self.de_vocab, max_len)
            trg_idx_pad, _ = pad(trg_idx, self.en_vocab, max_len)

            return src_idx_pad, src_len, trg_idx_pad

        for i in range(self.len):
            # 获取当前批量的数据
            if i == self.len - 1 and not self.drop_reminder:
                batch_data = self.dataset[i * self.batch_size:]
            else:
                batch_data = self.dataset[i * self.batch_size: (i+1) * self.batch_size]

            src_idx, src_len, trg_idx = encode_and_pad(batch_data, self.max_len)
            # 将序列数据转换为tensor
            yield mindspore.Tensor(src_idx, mindspore.int32), \
                mindspore.Tensor(src_len, mindspore.int32), \
                mindspore.Tensor(trg_idx, mindspore.int32)

    def __len__(self):
        return self.len