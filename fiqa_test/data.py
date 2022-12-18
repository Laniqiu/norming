"""
BertForSequenceClassification情感分析，三分类
flow：
    定义数据、读取数据、数据处理（tookenization和padding）
    定义模型（参数？ basic bert？）
"""

import os
import logging
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from sklearn.model_selection import KFold
import pandas as pd

from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig

logging.basicConfig(level=logging.DEBUG,
                    filename='test.log',
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')

# hyper params
hidden_dropout_prob = 0.3
num_labels = 3
learning_rate = 1e-5
weight_decay = 1e-2
epochs = 2
batch_size = 8  # todo 改成1？

# Configuration options
k_folds = 5
num_epochs = 2
# loss_function = nn.CrossEntropyLoss()
data_path = "/mnt/d/wsl2/absa/FiQA_PhraseBank.csv"


class SentimentDataset(Dataset):
    """数据类，继承自torch的Dataset"""
    label_dict = {"positive": 0,
                  "neutral": 1,
                  "negative": 2}
    def __init__(self, path_to_file):
        self.dataset = pd.read_csv(path_to_file, names=["Sentence", "Sentiment"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset.loc[idx, "Sentence"]
        label = self.dataset.loc[idx, "Sentiment"]  # positive, neutral, negative
        label_id = self.label_dict.get(label, 1)
        sample = {"text": text, "label": label_id}
        return sample

# define tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# todo 暂时只能在cpu上跑
# 通过model.to(device)的方式使用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = BertConfig.from_pretrained("bert-base-uncased", num_labels=num_labels, hidden_dropout_prob=hidden_dropout_prob)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", config=config)
model.to(device)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
criterion = nn.CrossEntropyLoss()


def convert_text_to_ids(tokenizer, text, max_len=100):
    if isinstance(text, str):
        tokenized_text = tokenizer.encode_plus(text,
                                               max_length=max_len,
                                               add_special_tokens=True,
                                               truncation=True,
                                               truncation_strategy='only_second')
        input_ids = tokenized_text["input_ids"]
        token_type_ids = tokenized_text["token_type_ids"]
    elif isinstance(text, list):
        input_ids = []
        token_type_ids = []
        for t in text:
            tokenized_text = tokenizer.encode_plus(t,
                                               max_length=max_len,
                                               add_special_tokens=True,
                                               truncation=True,
                                               truncation_strategy='only_second')
            input_ids.append(tokenized_text["input_ids"])
            token_type_ids.append(tokenized_text["token_type_ids"])
    else:
        input_ids, token_type_ids = None, None
    return input_ids, token_type_ids


def seq_padding(tokenizer, X):
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
    if len(X) <= 1:
        return torch.tensor(X)
    L = [len(x) for x in X]
    ML = max(L)
    X = torch.Tensor([x + [pad_id] * (ML - len(x)) if len(x) < ML else x for x in X])
    return X


def train(model, iterator, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for i, batch in enumerate(iterator):
        label = batch["label"]  # 需要转成id
        text = batch["text"]
        input_ids, token_type_ids = convert_text_to_ids(tokenizer, text)
        if not input_ids and not token_type_ids:
            logging.error("unexpected input")
            continue
        input_ids = seq_padding(tokenizer, input_ids)
        token_type_ids = seq_padding(tokenizer, token_type_ids)
        # 标签形状为 (batch_size, 1)
        label = label.unsqueeze(1)
        # 需要 LongTensor
        input_ids, token_type_ids, label = input_ids.long(), token_type_ids.long(), label.long()
        # 梯度清零
        optimizer.zero_grad()
        # 迁移到GPU?? todo
        input_ids, token_type_ids, label = input_ids.to(device), token_type_ids.to(device), label.to(device)
        output = model(input_ids=input_ids, token_type_ids=token_type_ids, labels=label)
        y_pred_prob = output[1]
        y_pred_label = y_pred_prob.argmax(dim=1)
        # 计算loss
        # 这个 loss 和 output[0] 是一样的
        # loss = criterion(y_pred_prob.view(-1, 2), label.view(-1))
        loss = output[0]
        # 计算acc
        acc = ((y_pred_label == label.view(-1)).sum()).item()
        # 反向传播
        loss.backward()
        optimizer.step()
        # epoch 中的 loss 和 acc 累加
        epoch_loss += loss.item()
        epoch_acc += acc
        if i % 200 == 0:
            print("current loss:", epoch_loss / (i+1), "\t", "current acc:", epoch_acc / ((i+1)*len(label)))
    # return epoch_loss / len(iterator), epoch_acc / (len(iterator) * iterator.batch_size)
    # 经评论区提醒修改
    return epoch_loss / len(iterator), epoch_acc / len(iterator.dataset.dataset)


def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for _, batch in enumerate(iterator):
            label = batch["label"]
            text = batch["text"]
            input_ids, token_type_ids = convert_text_to_ids(tokenizer, text)
            input_ids = seq_padding(tokenizer, input_ids)
            token_type_ids = seq_padding(tokenizer, token_type_ids)
            label = label.unsqueeze(1)
            input_ids, token_type_ids, label = input_ids.long(), token_type_ids.long(), label.long()
            input_ids, token_type_ids, label = input_ids.to(device), token_type_ids.to(device), label.to(device)
            output = model(input_ids=input_ids, token_type_ids=token_type_ids, labels=label)
            y_pred_label = output[1].argmax(dim=1)
            loss = output[0]
            acc = ((y_pred_label == label.view(-1)).sum()).item()
            epoch_loss += loss.item()
            epoch_acc += acc
    # return epoch_loss / len(iterator), epoch_acc / (len(iterator) * iterator.batch_size)
    # 经评论区提醒修改
    return epoch_loss / len(iterator), epoch_acc / len(iterator.dataset.dataset)


if __name__ == '__main__':
    # For fold results
    results = {}

    # Prepare dataset
    dataset = SentimentDataset(data_path)
    logging.info("data size: %d" % len(dataset))

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in each fold
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_subsampler)

        # Run the training loop for defined number of epochs
        for epoch in range(num_epochs):
            # Print epoch
            print(f'Starting epoch {epoch + 1}')
            train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
            logging.info("training loss: %f, accuracy: %f" % (train_loss, train_acc))
            valid_loss, valid_acc = evaluate(model, test_loader, criterion, device)
            print("valid loss: ", valid_loss, "\t", "valid acc:", valid_acc)
