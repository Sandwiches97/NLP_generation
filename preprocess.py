import re
import pandas as pd
import numpy as np
import string
import torch
from zhon.hanzi import punctuation
from datasets import Dataset
import jieba


def dataLoad(name: str):
    path = "data/" + name + '.json'
    data = pd.read_json(path)
    contents = data["content"]
    data["content"] = [re.sub("[{}]+".format(punctuation), '', content) for content in contents]
    data["content"] = [re.sub("['\n', ' ', '\xa0']", '', content) for content in contents]
    data["title"] = [re.sub("[{}]+".format(punctuation), '', title) for title in data["title"] ]

    # contents[:] = [jieba.lcut(content) for content in contents]
    # titles[:] = [jieba.lcut(title) for title in titles]
    return data


class Data:
    def __init__(self, tokenizer):
        self.max_input_length = 512
        self.max_target_length = 64
        self.tokenizer = tokenizer

    def preProcess(self):
        train_dic = dataLoad('train')
        test_dic = dataLoad('dev')

        def preprocess_function(example):
            inputs = example['content']
            model_inputs = self.tokenizer(inputs, max_length=self.max_input_length,
                                          padding='max_length', truncation=True)
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(example['title'], max_length=self.max_target_length,
                                        padding='max_length', truncation=True)
            model_inputs['labels'] = labels['input_ids']
            return model_inputs

        train_dataset = Dataset.from_dict(train_dic)
        test_dataset = Dataset.from_dict(test_dic)
        tokenized_train_dataset = train_dataset.map(preprocess_function)
        tokenized_test_dataset = test_dataset.map(preprocess_function)
        return tokenized_train_dataset, tokenized_test_dataset


if __name__ == "__main__":
    data = dataLoad("train")
    data["content"] = data["content"].apply(lambda x: jieba.lcut(x))

    print(data)