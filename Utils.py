import mindspore.ops as ops
import spacy
import mindspore


def clip_by_norm(clip_norm, t, axis=None):

    # 计算L2-norm
    t2 = t * t
    l2sum = t2.sum(axis=axis, keepdims=True)
    pred = l2sum > 0
    # 将加和中等于0的元素替换为1，避免后续出现NaN
    l2sum_safe = ops.select(pred, l2sum, ops.ones_like(l2sum))
    l2norm = ops.select(pred, ops.sqrt(l2sum_safe), l2sum)
    # 比较L2-norm和clip_norm，如L2-norm超过阈值，进行裁剪
    # 剪裁方法：output(x) = (x * clip_norm)/max(|x|, clip_norm)
    intermediate = t * clip_norm
    cond = l2norm > clip_norm
    t_clip = ops.identity(intermediate / ops.select(cond, l2norm, clip_norm))
    return t_clip


def translate_sentence(sentence, de_vocab, en_vocab, model, max_len=32):
    model.set_train(False)
    if isinstance(sentence, str):
        spacy_lang = spacy.load('en_core_web_sm')
        tokens = [token.text.lower() for token in spacy_lang(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    if len(tokens) > max_len - 2:
        src_len = max_len
        tokens = ['<bos>'] + tokens[:max_len - 2] + ['<eos>']
    else:
        src_len = len(tokens) + 2
        tokens = ['<bos>'] + tokens + ['<eos>'] + ['<pad>'] * (max_len - src_len)

    src = de_vocab.encode(tokens)
    src = mindspore.Tensor(src, mindspore.int32).expand_dims(1)
    src_len = mindspore.Tensor([src_len], mindspore.int32)
    trg = mindspore.Tensor([en_vocab.bos_idx], mindspore.int32).expand_dims(1)
    outputs = model(src, src_len, trg, max_len)
    trg_indexes = [int(i.argmax(1).asnumpy()) for i in outputs]
    eos_idx = trg_indexes.index(en_vocab.eos_idx) if en_vocab.eos_idx in trg_indexes else -1
    trg_tokens = en_vocab.decode(trg_indexes[:eos_idx])
    return trg_tokens
