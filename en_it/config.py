from pathlib import Path

def get_config():
    return {
        "batch_size": 32,
        "num_epochs": 2000,
        "lr": 10**-4,
        "seq_len": 350, #表示句子最大长度（包含填充部分pad）
        "d_model": 512, #每一词表示的词嵌入向量大小
        "datasource": r'opus_books', #加载的数据集
        "lang_src": "en", #英语数据集
        "lang_tgt": "it", #意大利数据
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": r"D:\conda3\Transfer_Learning\NLP\projects\torch\torch_transformer_translate\en_it\dataset/tokenizer_{0}.json",#分词器文件保存位置
        "experiment_name": r"./runs/"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('..') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

if __name__ == '__main__':
    from tqdm import tqdm
    import time
    # 创建一个范围为10的进度条
    for i in tqdm(range(10)):
        # 在每个迭代周期内使用tqdm.write()输出
        tqdm.write(f"Processing item {i + 1}")
        # 模拟一些处理时间
        time.sleep(1)