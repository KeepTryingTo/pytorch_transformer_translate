from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

# import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_from_disk

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

os.environ["http_proxy"] = "http://127.0.0.1:9910"
os.environ["https_proxy"] = "http://127.0.0.1:9910"


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


def run_validation(
        model, validation_ds, tokenizer_src,
        tokenizer_tgt, max_len, device, print_msg,
        global_step, writer, num_examples=2
):
    """
    tokenizer_src: 表示源句子的分词器
    tokenizer_tgt: 表示目标句子的分词器
    max_len: 表示输入模型句子的最大长度
    """
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen(cmd='stty size', mode='r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(
                model=model, source=encoder_input, source_mask=encoder_mask,
                tokenizer_src=tokenizer_src, tokenizer_tgt=tokenizer_tgt, max_len=max_len, device=device
            )

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            #根据目标分词器解码模型输出结果，得到翻译之后的文本
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

"""
[UNK]：该标记用于识别序列中的未知单词。
[PAD]：填充标记以确保批次中的所有序列具有相同的长度，因此我们用此标记填充较短的句子。
    我们使用注意力掩码“告诉”模型在训练期间忽略填充的标记，因为它们对任务没有任何实际意义。
[SOS]：这是一个用于表示句子开始的标记。
[EOS]：这是一个用于表示句子结束的标记。
"""
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

def get_ds(config):
    # It only has the train split, so we divide it overselves 加载数据集
    # ds_raw = load_dataset(
    #     path=f"{config['datasource']}",
    #     name=f"{config['lang_src']}-{config['lang_tgt']}",
    #     split='train'
    # )
    # # 保存至本地
    # ds_raw.save_to_disk(dataset_path=r'dataset')
    #从本地磁盘加载数据集
    ds_raw = load_from_disk(dataset_path=r'dataset/')

    # Build tokenizers 分别得到英文和意大利的分词器
    tokenizer_src = get_or_build_tokenizer(config, ds=ds_raw, lang=config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% for training, 10% for validation 划分训练集和验证集
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(
        ds=train_ds_raw, tokenizer_src=tokenizer_src, tokenizer_tgt=tokenizer_tgt,
        src_lang=config['lang_src'], tgt_lang=config['lang_tgt'], seq_len=config['seq_len']
    )
    val_ds = BilingualDataset(
        ds=val_ds_raw, tokenizer_src=tokenizer_src, tokenizer_tgt=tokenizer_tgt,
        src_lang=config['lang_src'], tgt_lang=config['lang_tgt'], seq_len=config['seq_len']
    )

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    #计算所有英语句子和意大利句子中最大长度
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        src_vocab_size=vocab_src_len, tgt_vocab_size=vocab_tgt_len,
        src_seq_len=config["seq_len"],
        tgt_seq_len=config['seq_len'], d_model=config['d_model']
    )
    return model

def train_model(config):
    # Define the device
    # device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    # device = torch.device(device)
    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    #加载模型
    model = get_model(
        config=config, vocab_src_len=tokenizer_src.get_vocab_size(),
        vocab_tgt_len=tokenizer_tgt.get_vocab_size()
    ).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')
    #针对[PAD]不计算损失值
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]')).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            #最后输出映射道vocab_size大小，表示预测词库中的每一个词概率
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy ； label: [seq_len]
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(
            model=model, validation_ds=val_dataloader, tokenizer_src=tokenizer_src,
            tokenizer_tgt=tokenizer_tgt, max_len=config['seq_len'], device=device,
            print_msg=lambda msg: batch_iterator.write(msg),
            global_step=global_step, writer=writer
        )

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
