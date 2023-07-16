import os
import mindspore
from mindspore import nn, ops
from tqdm import tqdm
from Attention import Attention
from Decoder import Decoder
from Encoder import Encoder
from HearthStone import HearthStone
from Iterator import Iterator
from Model import Seq2Act
from Utils import clip_by_norm, translate_sentence
from Vocab import Vocab
from collections import Counter, OrderedDict
from mindspore import save_checkpoint
from mindspore import load_checkpoint, load_param_into_net
from nltk.translate.bleu_score import corpus_bleu

train_path = "./hearthstone/train/"
valid_path = "./hearthstone/valid/"
test_path = "./hearthstone/train/"
train_dataset, valid_dataset, test_dataset = \
    HearthStone(train_path), HearthStone(valid_path), HearthStone(test_path)


def build_vocab(dataset):
    de_words, en_words = [], []
    for de, en in dataset:
        de_words.extend(de)
        en_words.extend(en)
    de_count_dict = OrderedDict(sorted(Counter(de_words).items(), key=lambda t: t[1], reverse=True))
    en_count_dict = OrderedDict(sorted(Counter(en_words).items(), key=lambda t: t[1], reverse=True))
    return Vocab(de_count_dict, min_freq=2), Vocab(en_count_dict, min_freq=2)


de_vocab, en_vocab = build_vocab(train_dataset)
print('Unique tokens in de vocabulary:', len(de_vocab))

train_iterator = Iterator(train_dataset, de_vocab, en_vocab, batch_size=128, max_len=32, drop_reminder=True)
valid_iterator = Iterator(valid_dataset, de_vocab, en_vocab, batch_size=128, max_len=32, drop_reminder=False)
test_iterator = Iterator(test_dataset, de_vocab, en_vocab, batch_size=128, max_len=32, drop_reminder=False)

input_dim = len(de_vocab)  # 输入维度
output_dim = len(en_vocab)  # 输出维度
enc_emb_dim = 256  # Encoder Embedding层维度
dec_emb_dim = 256  # Decoder Embedding层维度
enc_hid_dim = 512  # Encoder 隐藏层维度
dec_hid_dim = 512  # Decoder 隐藏层维度
enc_dropout = 0.5
dec_dropout = 0.5
src_pad_idx = de_vocab.pad_idx
trg_pad_idx = en_vocab.pad_idx

attn = Attention(enc_hid_dim, dec_hid_dim)
encoder = Encoder(input_dim, enc_emb_dim, enc_hid_dim, dec_hid_dim, enc_dropout)
decoder = Decoder(output_dim, dec_emb_dim, enc_hid_dim, dec_hid_dim, dec_dropout, attn)
model = Seq2Act(encoder, decoder, src_pad_idx, 0.5)

opt = nn.Adam(model.trainable_params(), learning_rate=0.001)  # 损失函数
loss_fn = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)


def forward_fn(src, src_len, trg):
    """前向网络"""
    src = src.swapaxes(0, 1)
    trg = trg.swapaxes(0, 1)

    output = model(src, src_len, trg)
    output_dim = output.shape[-1]
    output = output.view(-1, output_dim)
    trg = trg[1:].view(-1)
    loss = loss_fn(output, trg)

    return loss


def evaluate(iterator):
    """模型验证"""
    model.set_train(False)
    num_batches = len(iterator)
    total_loss = 0  # 所有batch训练loss的累加
    total_steps = 0  # 训练步数

    with tqdm(total=num_batches) as t:
        for src, src_len, trg in iterator():
            loss = forward_fn(src, src_len, trg)  # 当前batch的loss
            total_loss += loss.asnumpy()
            total_steps += 1
            curr_loss = total_loss / total_steps  # 当前的平均loss
            t.set_postfix({'loss': f'{curr_loss:.2f}'})
            t.update(1)

    return total_loss / total_steps


# 反向传播计算梯度
grad_fn = mindspore.value_and_grad(forward_fn, None, opt.parameters)


def train_step(src, src_len, trg, clip):
    """单步训练"""
    loss, grads = grad_fn(src, src_len, trg)
    grads = ops.HyperMap()(ops.partial(clip_by_norm, clip), grads)  # 梯度裁剪
    opt(grads)  # 更新网络参数

    return loss


def train(iterator, clip, epoch=0):
    """模型训练"""
    model.set_train(True)
    num_batches = len(iterator)
    total_loss = 0  # 所有batch训练loss的累加
    total_steps = 0  # 训练步数

    with tqdm(total=num_batches) as t:
        t.set_description(f'Epoch: {epoch}')
        for src, src_len, trg in iterator():
            loss = train_step(src, src_len, trg, clip)  # 当前batch的loss
            total_loss += loss.asnumpy()
            total_steps += 1
            curr_loss = total_loss / total_steps  # 当前的平均loss
            t.set_postfix({'loss': f'{curr_loss:.2f}'})
            t.update(1)

    return total_loss / total_steps


num_epochs = 128  # 训练迭代数
clip = 1.0  # 梯度裁剪阈值
best_valid_loss = float('inf')  # 当前最佳验证损失
ckpt_file_name = os.path.join('./model', 'seq2act.ckpt')  # 模型保存路径

for i in range(num_epochs):
    # 模型训练，网络权重更新
    train_loss = train(train_iterator, clip, i)
    # 网络权重更新后对模型进行验证
    valid_loss = evaluate(valid_iterator)

    # 保存当前效果最好的模型
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        save_checkpoint(model, ckpt_file_name)

# 加载之前训练好的模型
param_dict = load_checkpoint(ckpt_file_name)
load_param_into_net(model, param_dict)
# 以测试数据集中的第一组语句为例，进行测试
example_idx = 0
in_put = test_dataset[example_idx][0]
out_put = test_dataset[example_idx][1]

print(f'in_put = {in_put}')
print(f'out_put = {out_put}')

translation = translate_sentence(in_put, de_vocab, en_vocab, model)

print(f'predicted out_put = {translation}')


def calculate_bleu(dataset, de_vocab, en_vocab, model, max_len=50):
    trgs = []
    pred_trgs = []
    for data in dataset:
        src = data[0]
        trg = data[1]
        # 获取模型预测结果
        pred_trg = translate_sentence(src, de_vocab, en_vocab, model, max_len)
        pred_trgs.append(pred_trg)
        trgs.append([trg])

    return corpus_bleu(trgs, pred_trgs)


# 计算BLEU Score
bleu_score = calculate_bleu(test_dataset, de_vocab, en_vocab, model)

print(f'BLEU score = {bleu_score * 100:.2f}')
