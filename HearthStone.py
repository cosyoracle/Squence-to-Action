import os
import spacy
from functools import partial
from mindnlp.transforms import BasicTokenizer


class HearthStone:

    def __init__(self, path):
        self.data = self._load(path)

    @staticmethod
    def _load(path):
        tokenizer = BasicTokenizer()

        def tokenize(text, spacy_lang):
            # 去除多余空格，统一大小写
            text = text.rstrip()
            # return [tok.text.lower() for tok in spacy_lang.tokenizer(text)]
            return tokenizer(text).tolist()

        # 加载英、德语分词器
        tokenize_de = partial(tokenize, spacy_lang=spacy.load('en_core_web_sm'))
        tokenize_en = partial(tokenize, spacy_lang=spacy.load('en_core_web_sm'))

        members = {i.split('.')[-1]: i for i in os.listdir(path)}
        de_path = os.path.join(path, members['in'])
        en_path = os.path.join(path, members['out'])
        with open(de_path, 'r') as de_file:
            de = de_file.readlines()[:-1]
            de = [tokenize_de(i) for i in de]
        with open(en_path, 'r') as en_file:
            en = en_file.readlines()[:-1]
            en = [tokenize_en(i) for i in en]

        return list(zip(de, en))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
