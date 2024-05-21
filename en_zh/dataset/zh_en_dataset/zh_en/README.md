---
language:
- en
- zh
task_categories:
- translation

---
# AutoTrain Dataset for project: opus-mt-en-zh_hanz

## Dataset Description

This dataset has been automatically processed by AutoTrain for project opus-mt-en-zh_hanz.

### Languages

The BCP-47 code for the dataset's language is en2zh.

## Dataset Structure

### Data Instances

A sample from this dataset looks as follows:

```json
[
  {
    "source": "And then I hear something.",
    "target": "\u63a5\u7740\u542c\u5230\u4ec0\u4e48\u52a8\u9759\u3002",
    "feat_en_length": 26,
    "feat_zh_length": 9
  },
  {
    "source": "A ghostly iron whistle blows through the tunnels.",
    "target": "\u9b3c\u9b45\u7684\u54e8\u58f0\u5439\u8fc7\u96a7\u9053\u3002",
    "feat_en_length": 49,
    "feat_zh_length": 10
  }
]
```

### Dataset Fields

The dataset has the following fields (also called "features"):

```json
{
  "source": "Value(dtype='string', id=None)",
  "target": "Value(dtype='string', id=None)",
  "feat_en_length": "Value(dtype='int64', id=None)",
  "feat_zh_length": "Value(dtype='int64', id=None)"
}
```

### Dataset Splits

This dataset is split into a train and validation split. The split sizes are as follow:

| Split name   | Num samples         |
| ------------ | ------------------- |
| train        | 16350 |
| valid        | 4088 |
