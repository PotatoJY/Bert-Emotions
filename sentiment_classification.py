# -*- coding: utf-8 -*-
# @Time    : 2024-4-26 10:58
# @Author  : jizhengyu
# @File    : sentiment_classification.py
# @Software: PyCharm
import torch
import numpy as np
from datasets import load_dataset
from collections import Counter
from imblearn.over_sampling import SMOTE  # 导入过采样方法SMOTE
from transformers import AutoModelForSequenceClassification,AutoTokenizer,AutoModel
from transformers import Trainer, TrainingArguments

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import f1_score,accuracy_score
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# import requests
# requests.head("https://www.dropbox.com/s/1pzkadrvffbqw6o/train.txt?dl=1")
# from transformers import TFAutoModel
# tf_model = TFAutoModel.from_pretrained(model_ckpt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

model = AutoModel.from_pretrained('distilbert-base-uncased')



def label_int2str(row):
    return emotions['train'].features["label"].int2str(row)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

def extract_hidden_states(batch):
    # Place model inputs on the GPU
    inputs = {k:v.to(device) for k,v in batch.items()
              if k in tokenizer.model_input_names}
    # Extract last hidden states
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    # Return vector for [CLS] token
    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}


def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

if __name__ == '__main__':
    emotions = load_dataset('emotion')
    # emotions.set_format(type="pandas")
    # df = emotions['train'][:]
    # '''转换并生成一列label name'''
    # df["label_name"] = df["label"].apply(label_int2str)
    # print(df.head())
    # emotions.reset_format()
    # print("trian length:{}\nvalid_length:{}\ntest_length:{}".format(len(train_dataset),len(valid_dataset),len(test_dataset)))


    '''Encode'''
    emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
    print(emotions_encoded["train"].column_names)

    emotions_encoded.set_format("torch",
                                columns=["input_ids", "attention_mask", "label"])
    emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True,batch_size=1000)
    print(emotions_encoded.column_names)

    '''ValueError: could not convert string to float: 'i didnt feel humiliated'''
    # smote = SMOTE(random_state=0)  # 建立SMOTE模型对象
    # X_resampled, Y_resampled = smote.fit_resample(train_dataset['text'],train_dataset['label'] )  # 输入数据并作过采样处理
    # print(Counter(Y_resampled))


    # print(emotions_hidden["train"].column_names)

    X_train = np.array(emotions_hidden["train"]["hidden_state"])
    X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
    y_train = np.array(emotions_hidden["train"]["label"])
    y_valid = np.array(emotions_hidden["validation"]["label"])
    print(X_train.shape, X_valid.shape)

    '''Train 模型训练'''

    num_labels = 6
    model = (AutoModelForSequenceClassification
             .from_pretrained('distilbert-base-uncased', num_labels=num_labels)
             .to(device))

    batch_size = 64
    logging_steps = len(emotions_encoded["train"]) // batch_size
    model_name = f"{'distilbert-base-uncased'}-finetuned-emotion"
    training_args = TrainingArguments(output_dir=model_name,
                                      num_train_epochs=2,
                                      learning_rate=2e-5,
                                      per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      weight_decay=0.01,
                                      evaluation_strategy="epoch",
                                      disable_tqdm=False,
                                      logging_steps=logging_steps,
                                      push_to_hub=True,
                                      log_level="error")

    trainer = Trainer(model=model, args=training_args,
                       compute_metrics=compute_metrics,
                      train_dataset=emotions_encoded["train"],
                      eval_dataset=emotions_encoded["validation"],
                      tokenizer=tokenizer)
    trainer.train()

    preds_output = trainer.predict(emotions_encoded["validation"])
    print(preds_output.metrics)

    y_preds = np.argmax(preds_output.predictions, axis=1)

    labels = emotions["train"].features["label"].names
    plot_confusion_matrix(y_preds, y_valid, labels)


    '''误差分析'''
    from torch.nn.functional import cross_entropy

    def forward_pass_with_label(batch):
        # Place all input tensors on the same device as the model
        inputs = {k: v.to(device) for k, v in batch.items()
                  if k in tokenizer.model_input_names}

        with torch.no_grad():
            output = model(**inputs)
            pred_label = torch.argmax(output.logits, axis=-1)
            loss = cross_entropy(output.logits, batch["label"].to(device),
                                 reduction="none")

        # Place outputs on CPU for compatibility with other dataset columns
        return {"loss": loss.cpu().numpy(),
                "predicted_label": pred_label.cpu().numpy()}


    # 使用map()方法，获取所有样本的损失
    # Convert our dataset back to PyTorch tensors
    emotions_encoded.set_format("torch",
                                columns=["input_ids", "attention_mask", "label"])
    # Compute loss values
    emotions_encoded["validation"] = emotions_encoded["validation"].map(
        forward_pass_with_label, batched=True, batch_size=16)

    '''创建一个带有文本、损失和预测/真实标签的“DataFrame”'''
    emotions_encoded.set_format("pandas")
    cols = ["text", "label", "predicted_label", "loss"]
    df_test = emotions_encoded["validation"][:][cols]
    df_test["label"] = df_test["label"].apply(label_int2str)
    df_test["predicted_label"] = (df_test["predicted_label"]
                                  .apply(label_int2str))