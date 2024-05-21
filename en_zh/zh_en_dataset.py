"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/5/6-13:46
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import jieba
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class myDataset(Dataset):

    def __init__(self, json_data, tokenizer_src, tokenizer_tgt, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.json_data = json_data.dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        # 得到目标语言语句的开头，结束和填充字符索引
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        src_target_pair = list(self.json_data.items())[idx]
        src_text = src_target_pair[0]  # 得到英语句子
        tgt_text = src_target_pair[1]  # 得到中文句子

        jieba_text = [str(t) for t in list(jieba.cut(tgt_text))]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids  # 得到对应英语句子中单词索引
        #注意这里的中文已经jieba分词工具给划分了，所以对其进行编码的格式如下，is_pretokenized表示预先已经进行了划分
        dec_input_tokens = self.tokenizer_tgt.encode(jieba_text,is_pretokenized = True).ids  # 得到对应中文句子中词索引

        # 以下分别得到对于英语token以及意大利token还需要填充的大小
        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s>
        # We will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            # raise ValueError("Sentence is too long")
            if enc_num_padding_tokens < 0:
                enc_input_tokens = enc_input_tokens[:self.seq_len - 2]
                enc_num_padding_tokens = 0
            if dec_num_padding_tokens < 0:
                dec_input_tokens = dec_input_tokens[:self.seq_len - 1]
                dec_num_padding_tokens = 0

        # Add <s> and </s> token 得到英语句子的填充+开头+结尾标识符之后token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only <s> token 用于decoder输入的意大利语句，只需要加上开头和填充标识符，不需要加上结束标识符
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Add only </s> token 用于decoder最后输出的结果计算预测，以结束标识符为作为结尾
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()
        decoder_input_mask = (decoder_input != self.pad_token).unsqueeze(0).int()  # (1, seq_len)
        decoder_input_size_mask = causal_mask(decoder_input.size(0))  # (1, seq_len, seq_len),
        decoder_mask = decoder_input_mask & decoder_input_size_mask
        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": encoder_mask,  # (1, 1, seq_len)
            "decoder_mask": decoder_mask,
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(size):
    # diagonal： 表示获得上三角部分的元素，其他部分使用0填充
    mask = torch.triu(
        torch.ones((1, size, size)), diagonal=1
    ).type(torch.int)
    return mask == 0


if __name__ == '__main__':
    input = torch.ones(size=(1, 5), dtype=torch.int64)
    mask = torch.triu(
        torch.ones((1, 5, 5)), diagonal=1
    ).type(torch.int)
    no_mask = (mask == 0)
    print('mask: {}'.format(mask))
    print('no mask: {}'.format(no_mask))
    print('no mask & input: {}'.format(input & no_mask))