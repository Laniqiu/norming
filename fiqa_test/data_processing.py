"""
BertForSequenceClassification情感分析，三分类
flow：
    定义数据、读取数据、数据处理（tookenization和padding）
    定义模型（参数？ basic bert？）
"""
import torch
from torch.utils.data import Dataset
import pandas as pd

# 超参数
hidden_dropout_prob = 0.3
num_labels = 3
learning_rate = 1e-5
weight_decay = 1e-2
epochs = 2
batch_size = 1  # todo 改成1？


class SentimentDataset(Dataset):
    """数据类，继承自torch的Dataset"""
    def __init__(self, path_to_file):
        self.dataset = pd.read_csv(path_to_file, sep="\t", names=["Sentence", "Sentiment"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset.loc[idx, "Sentence"]
        label = self.dataset.loc[idx, "Sentiment"]
        sample = {"text": text, "label": label}
        return sample


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    data_path = "/mnt/d/wsl2/absa/FiQA_PhraseBank.csv"  # todo path，后面改成参数配置？
    # 加载数据集, train、valid、test
    # todo
    train_set = SentimentDataset(data_path)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    valid_set = SentimentDataset(data_path)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2)
