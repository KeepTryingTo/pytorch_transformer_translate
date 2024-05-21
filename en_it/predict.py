"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/5/5-21:50
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import os
import torch

from torch_transformer_translate.en_it.dataset import causal_mask
from torch_transformer_translate.en_it.config import get_config
from datasets import load_from_disk

from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch_transformer_translate.en_it.model import build_transformer

device = 'cpu' if torch.cuda.is_available() else 'cpu'

def greedy_decode(
        model, source, source_mask,
        tokenizer_src, tokenizer_tgt, max_len, device
):
    """
    source: 输入的原句子对应索引
    source_mask: 原句子索引对应的mask
    推理阶段，解码器的输入是一个单词一个单词的输入，然后进行翻译
    """
    #得到开始标识符和结尾标识符
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step 编码器输出结果
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    # 最开始的时候编码器输入的是一个开始标识符
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        # calculate output （encoder_output, encoder_mask, decoder_input, decoder_mask）
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        # get next token  根据前面预测的结果，得到下一个词的输出概率
        prob = model.project(out[:, -1])
        #得到预测概率最大的单词索引next_word
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)
            ], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        #关于Tokenizer有哪些属性可用：https://huggingface.co/docs/tokenizers/api/tokenizer
        tokenizer = Tokenizer(model=WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        #get_all_sentences得到所有的句子
        tokenizer.train_from_iterator(iterator=get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        src_vocab_size=vocab_src_len, tgt_vocab_size=vocab_tgt_len,
        src_seq_len=config["seq_len"],
        tgt_seq_len=config['seq_len'], d_model=config['d_model']
    )
    return model


def predict(en_sentence = 'I have a dream',model_name = r'opus_books_weights/tmodel_720.pt'):
    config = get_config()
    data_dir = r'D:\conda3\Transfer_Learning\NLP\projects\torch\torch_transformer_translate'
    # 从本地磁盘加载数据集
    ds_raw = load_from_disk(
        dataset_path=os.path.join(data_dir,r'en_it/dataset/')
    )

    # Build tokenizers 分别得到英文和意大利的分词器
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # 加载模型
    model = get_model(
        config=config, vocab_src_len=tokenizer_src.get_vocab_size(),
        vocab_tgt_len=tokenizer_tgt.get_vocab_size()
    ).to(device)

    checkpoint = torch.load(
        os.path.join(r'D:\conda3\Transfer_Learning\NLP\projects\torch\torch_transformer_translate\en_it',model_name),
        map_location='cpu'
    )['model_state_dict']
    model.load_state_dict(checkpoint)

    #对要翻译的句子进行向量化
    # 得到目标语言语句的开头，结束和填充字符索引
    sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
    eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
    pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    # Transform the text into tokens
    enc_input_tokens = tokenizer_src.encode(en_sentence).ids  # 得到对应英语句子中单词索引
    # dec_input_tokens = tokenizer_tgt.encode(tgt_text).ids  # 得到对应意大利句子中单词索引

    # 以下分别得到对于英语token以及意大利token还需要填充的大小
    # Add sos, eos and padding to each sentence
    enc_num_padding_tokens = config['seq_len'] - len(enc_input_tokens) - 2  # We will add <s> and </s>
    # We will only add <s>, and </s> only on the label
    # dec_num_padding_tokens = config['seq_len'] - len(dec_input_tokens) - 1

    # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
    if enc_num_padding_tokens < 0:
        raise ValueError("Sentence is too long")

    # Add <s> and </s> token 得到英语句子的填充+开头+结尾标识符之后token
    encoder_input = torch.cat(
        [
            sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            eos_token,
            torch.tensor([pad_token] * enc_num_padding_tokens, dtype=torch.int64),
        ],
        dim=0,
    )
    encoder_mask = (encoder_input != pad_token).unsqueeze(0).unsqueeze(0).int()
    model.eval()

    model_out = greedy_decode(
        model=model, source=encoder_input, source_mask=encoder_mask,
        tokenizer_src=tokenizer_src, tokenizer_tgt=tokenizer_tgt, max_len=config['seq_len'], device=device
    )

    # 根据目标分词器解码模型输出结果，得到翻译之后的文本
    model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

    print('source sentence: {}'.format(en_sentence))
    print('prediction sentence: {}'.format(model_out_text))

    return model_out_text

if __name__ == '__main__':
    """
    en: how are you?
    
    en: It's a beautiful day. Do you want to go out for a walk?  
    it: È una bella giornata. Vuoi uscire per fare una passeggiata?
    
    en: You are very good. You have persevered under so many difficulties.
    it: Sei molto bravo. Lei ha perseverato in tante difficoltà.
    
    en: If you could have one wish, what would it be? Wealth, wisdom, and eternal life.
    it: Se poteste avere un desiderio, cosa sarebbe? Ricchezza, saggezza e vita eterna.
    """
    predict(en_sentence="If you could have one wish, what would it be? Wealth, wisdom, and eternal life.")
    pass