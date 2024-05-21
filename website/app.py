"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/5/20-20:34
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import gradio as gr
from torch_transformer_translate.en_it.predict import predict as predict_en_it
from torch_transformer_translate.en_zh.predict_en_zh import predict as predict_en_zh

#其中greet函数的参数和inputs参数是对应的，greet额输出结果多少和outputs是对应的
def greet(translate_type,translate_model,translate_text):
    default_model = 'en_it_transformer_720'
    dataset_index = 1
    if translate_model == 'en_it_transformer_720':
        default_model = r'opus_books_weights/tmodel_720.pt'
    elif translate_model == 'en_zh_transformer_146':
        default_model = r'en_zh01_weights/tmodel_146.pt'
    elif translate_model == 'en_zh_transformer_14':
        #使用的是数据集2训练的模型
        default_model = r'en_zh02_weights/tmodel_14.pt'
        dataset_index = 2

    if translate_type == 'en_zh(英语-中文)':
        return predict_en_zh(en_sentence=translate_text,model_name = default_model,dataset_index=dataset_index)
    elif translate_type == 'en_it(英语-意大利语)':
        return predict_en_it(en_sentence=translate_text,model_name = default_model)

if __name__ == '__main__':
    demo = gr.Interface(
        fn=greet,
        inputs=[
            gr.Dropdown(
                choices=['en_it(英语-意大利语)','en_zh(英语-中文)'],
                label='源句子-目标句子'
            ),
            gr.Dropdown(
                choices=['en_it_transformer_720','en_zh_transformer_146','en_zh_transformer_14'],
                label='模型选择'
            ),
            gr.Textbox(lines=5,placeholder='输入翻译的句子...',label='源句子(要翻译的句子)')
        ],
        outputs=[
            gr.Textbox(lines=5,placeholder='输出翻译的结果...',label='翻译结果',)
        ],
        title='基于Transformer的机器翻译',
        submit_btn='点击翻译',
        clear_btn='清除文本框内容'
    )
    demo.launch()