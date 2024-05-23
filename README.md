<p align = "center">
	<a href = "https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421"><img src = "https://img.shields.io/badge/Pytorch-NLP-%23CC05FF"/></a>
	<a href = "https://blog.csdn.net/Keep_Trying_Go?spm=1010.2135.3001.5421"><img src = "https://img.shields.io/badge/Pytorch-Transformer-door"/></a>
</p>
<hr style="border : 1px dashed blue;" />

# pytorch_transformer_translate
<h>使用pytorch深度学习框架，基于Transformer的机器翻译，并且使用gradio框架写了一个页面用于输入句子进行翻译</h>

<p>
  This project's reference to <a href ="https://arxiv.org/pdf/1706.03762v7.pdf" title = "Attention Is All You Need">Attention Is All You Need</a> and <a href = "https://github.com/hkproj/pytorch-transformer" title = "基于Pytorch实现Transoformer的机器翻译">Pytorch-based release Transoformer's machine translator</a>.
</p>
<p><strong>Project Environment</strong></p>

```
Device: Server
NVIDIA：GA102[GeForce RTX 3090]
Anaconda Environment:Pyhon 3.6.13  Pytorch 1.9.0+cuda1.1.1  Tokenizers 0.12.1  Transformers 4.18.0
```

<p><strong>1.Project structure</strong></p>

````

en_it
	configs(include some dataset's information,in fact we don't use it)
	dataset(include train's dataset)
		data-00000-of-00001.arrow(from HuggingFace download dataset)
		dataset_info.json（include the download dataset some information）
		state.json
		tokenizer_en.json
		tokenizer_it.json
		tokenizer_zh.json
	opus_books_weights(save the train's weights file)
	runs(save the train's log)
	config.py(train's configuration information)
	dataset.py(read the train and test dataset)
	model.py(Transformer's all structure)
	predict.py(translate the single sentence)
	train.py
en_zh
	dataset
		zh_en_dataset
			myProcess
				zh_en01（the first dataset, from English to Chinese）
					zh_en.json（from .txt file transform to JSON file save the dataset）
					zh_en.txt
					zh_en_process.py（process the source dataset）
				zh_en02（the second dataset, from English to Chinese）
					- with zh_en01 equal
			zh_en(in fact,we don't use it)
			tokenizer_en.json(generate the English tokenizer)
			tokenizer_zh.json(成generate the Chinese tokenizer)
	runs（save the train's log）
		en_zh01
		en_zh02
	weights（save the train's weights file）
		en_zh01_weights
		en_zh02_weights
	predict_en_zh.py
	zh_en_config.py
	zh_en_dataset.py
	zh_en_train.py
website
	flagged
	app.py(web）
train_wb.py(with train.py not difference)
translate.py(input a single sentence and translate it)
 
````
<p>Notice: above give some files,when you train youselves dataset,please chanage the path</p>
<hr style="border : 1px dashed blue;" />
<p><strong>2.Train en_it</strong></p>
train process is not complicate，configuration information in config.py file，here give the train 720 epoch's weights file：

link：[https://pan.baidu.com/s/1g5Y38okBPb4AnE7A2RFaww ](https://pan.baidu.com/s/1g5Y38okBPb4AnE7A2RFaww )
Extract code：1n54

<hr style="border : 1px dashed blue;" />
<p><strong>3.Train en_zh</strong></p>

train process is not complicate，configuration information in zh en config.py's file，here give the 146 and 14 epoch's weights file：

train the zh_en01 dataset save weights file：

link：[https://pan.baidu.com/s/1mj_qZ4xadH9T7WtJYQ_L-A ](https://pan.baidu.com/s/1mj_qZ4xadH9T7WtJYQ_L-A )

Extract code：wjgk

train the zh_en02 dataset save weights file：

link：[https://pan.baidu.com/s/1FduLVLHnnkf2vXMf39lDgQ](https://pan.baidu.com/s/1FduLVLHnnkf2vXMf39lDgQ) 

Extract code：evhe

<p><strong>4.Inference</strong></p>
train app.py(please change youselves file path)

run and then output a url：
[http://127.0.0.1:7860](http://127.0.0.1:7860)

（1）from English to Italian's translate：


（2）from English to Chinese's translate：


<hr style="border : 1px dashed blue;" />

<h1>中文翻译(和英文说明不一定一一对应)</h1>
<p>
  该项目是参考了<a href ="https://arxiv.org/pdf/1706.03762v7.pdf" title = "Attention Is All You Need">Attention Is All You Need</a> 和 <a href = "https://github.com/hkproj/pytorch-transformer" title = "基于Pytorch实现Transoformer的机器翻译">采用Pytorch深度学习框架实现Transformer的机器翻译.</a>.
</p>
<p><strong>1.项目结构如下</strong></p>

<hr style="border : 1px dashed blue;" />

````

en_it
	configs(主要包含了一些数据集的信息，实际上并没有使用到)
	dataset(保存数据集的文件)
		data-00000-of-00001.arrow(从HuggingFace下载的数据集)
		dataset_info.json（数据集的一些信息）
		state.json
		tokenizer_en.json
		tokenizer_it.json
		tokenizer_zh.json
	opus_books_weights(保存训练过程的权重文件)
	runs(保存训练的日志记录)
	config.py(训练的一些配置信息)
	dataset.py(读取数据集的文件)
	model.py(Transformer整体结构)
	predict.py(用于单个语句的翻译预测)
	train.py(英文到意大利语的训练)
en_zh
	dataset
		zh_en_dataset
			myProcess
				zh_en01（第一个英文到中文翻译的数据）
					zh_en.json（从.txt数据转换到使用JSON文件保存）
					zh_en.txt
					zh_en_process.py（处理数据集文件）
				zh_en02（第二个英文到中文翻译的数据）
					- 和zh_en01的结构一样
			zh_en(可以不用看，只是当时使用到了Bert中的预训练模型对句子划分tokens，但是实际上没有用到)
			tokenizer_en.json(生成得到的英语分词器)
			tokenizer_zh.json(成得到的中文分词器)
	runs（保存两个数据集训练日志）
		en_zh01
		en_zh02
	weights（保存两个数据集训练的权重文件）
		en_zh01_weights
		en_zh02_weights
	predict_en_zh.py
	zh_en_config.py
	zh_en_dataset.py
	zh_en_train.py
website
	flagged
	app.py（网页的界面文件）
train_wb.py(和上面给出的train.py的内容差不多，区别在于train_wb.py采用了wandb记录训练过程)
translate.py(将训练好的模型用于用户输入句子的翻译)
 
````
<p>注意：以上的给出的文件目录，在自己训练数据集的过程中，有些路径自己修改一下即可训练自己的模型</p>
<hr style="border : 1px dashed blue;" />
<p><strong>2.训练en_it目录下英文到意大利语言的翻译</strong></p>
训练并不复杂，参数也不多，都在**config.py**文件中，因此这里给出训练了**720个epoch**的权重文件：

链接：[https://pan.baidu.com/s/1g5Y38okBPb4AnE7A2RFaww ](https://pan.baidu.com/s/1g5Y38okBPb4AnE7A2RFaww )
提取码：1n54

<hr style="border : 1px dashed blue;" />
<p><strong>3.训练en_zh目录下英文到意大利语言的翻译</strong></p>

训练并不复杂，参数也不多，都在zh en config.py文件中，因此这里给出训练了720个epoch的权重文件：

训练zh_en01数据集下得到的权重文件：

链接：[https://pan.baidu.com/s/1mj_qZ4xadH9T7WtJYQ_L-A ](https://pan.baidu.com/s/1mj_qZ4xadH9T7WtJYQ_L-A )

提取码：wjgk

训练zh_en02数据集下得到的权重文件：

链接：[https://pan.baidu.com/s/1FduLVLHnnkf2vXMf39lDgQ](https://pan.baidu.com/s/1FduLVLHnnkf2vXMf39lDgQ) 

提取码：evhe

<p><strong>4.测试</strong></p>
运行app.py(注意里面是否需要修改权重文件路径)

运行之后得到一个链接，点击它：
[http://127.0.0.1:7860](http://127.0.0.1:7860)

（1）测试从英语到意大利语的翻译结果：
<img src = "https://github.com/KeepTryingTo/pytorch_transformer_translate/blob/main/images/en_it.png"/>

（2）测试从英语到中文的翻译结果：
<img src = "https://github.com/KeepTryingTo/pytorch_transformer_translate/blob/main/images/en_zh.png"/>
<img src = "https://github.com/KeepTryingTo/pytorch_transformer_translate/blob/main/images/en_zh02.png"/>
