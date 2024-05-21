"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2024/5/6-13:47
@CSDN   : https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421
"""

import os
import json
import numpy as np

#txt => json
def process():
    file = r'zh_en.txt'
    with open(file,'r',encoding='utf-8') as fp:
        datasets = fp.readlines()
    ZH_EN_Dict = {}
    for line in datasets:
        list_data = line.split('\t')
        ZH_EN_Dict[list_data[0]] = list_data[1]

    # for source,target in ZH_EN_Dict.items():
    #     print('source: {}'.format(source))
    #     print('target: {}'.format(target))
    # with open('zh_en.json','w',encoding='utf-8') as fn:
    #     json.dump(ZH_EN_Dict,fn,ensure_ascii=False)
    with open('zh_en.json', 'r', encoding='utf-8') as fn:
        lines = json.load(fn)
    print(list(lines.items())[0])
    # for line in lines.items():
    #     print('source: {}'.format(line[0]))
    #     print('target: {}'.format(line[1]))
    print('size: {}'.format(datasets.__len__()))
if __name__ == '__main__':
    process()
    pass
