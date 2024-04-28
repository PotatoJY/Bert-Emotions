import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
#导入huggingface的数据集
# from datasets import list_datasets
# all_datasets = list_datasets()
# print(f"There are {len(all_datasets)} datasets currently available on the Hub")
# print(f"The first 10 are: {all_datasets[:10]}")

#我们已经了解了如何使用 数Datasets加载和检查数据
print('-----------加载情绪数据----------------')
import requests
requests.head("https://www.dropbox.com/s/1pzkadrvffbqw6o/train.txt?dl=1")
# print(requests)
from datasets import load_dataset
emotions = load_dataset("emotion")
print('-----------查看情绪数据--------------')
# print(emotions)
# train_ds = emotions["train"]
# print('--------------emotions训练集--------------\n',train_ds)
# print('--------------emotions训练集长度-----------\n',len(train_ds))
# print('--------------emotions训练集第一条数据---------\n',train_ds[0])
# print('--------------emotions训练集数据集格式---------\n',train_ds.column_names)#数据是推文文本和情感标签
# print('--------------emotions训练集数据集类型---------\n',type(train_ds['label']))
# print('--------------emotions训练集train_ds数据类型---------\n',train_ds.features)
# print('--------------emotions训练集train_ds使用切片访问多行-----\n',train_ds[:5])
# print('--------------emotions训练集train_ds按名称获取完整列-----\n',train_ds["text"][:5])

print('==============将Datasets对象转为DataFrame==============\n')
#允许我们更改 Dataset 的输出格式
import pandas as pd

emotions.set_format(type="pandas")
df = emotions["train"][:]
# print('--------------数据格式---------\n',df.head(1))

# 但是，label表示为整数让我们使用标签功能的 int2str() 方法在 DataFrame 中创建一个具有相应标签名称的新列
def label_int2str(row):
    return emotions["train"].features["label"].int2str(row)

df["label_name"] = df["label"].apply(label_int2str)
# print('--------------数据格式更新---------\n',df.head(1))

'''处理文本分类问题时，检查数据集中类别分布'''
print('-------------可视化类分布-------------')
import matplotlib.pyplot as plt

# df["label_name"].value_counts(ascending=True).plot.barh()
# plt.title("Frequency of Classes")
# plt.show()

'''通过查看每条推文的单词分布，我们可以粗略估计每种情绪的推文长度'''
# df["Words Per Tweet"] = df["text"].str.split().apply(len) # 按空格切分，获取雷彪长度
# df.boxplot("Words Per Tweet", by="label_name", grid=False, showfliers=False,
#            color="black")
# plt.suptitle("")
# plt.xlabel("")
# plt.show()
# 重置数据集的输出格式，因为我们不再需要 DataFrame 格式
emotions.reset_format()

print('---------------从文本到分词----------------')
# 像 DistilBERT 这样的 Transformer 模型不能接收原始字符串作为输入
# 查看用于 DistilBERT 的分词器之前，让我们考虑两种常见情况：_character_ 和 word 分词。
print('------------Character Tokenization----------')
text = "Tokenizing text is a core task of NLP."
tokenized_text = list(text)
# print(tokenized_text)#快速实现字符级标记化

token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
# print(token2idx)#使用唯一整数对每个唯一标记（在本例中为字符）进行编码

input_ids = [token2idx[token] for token in tokenized_text]
# print(input_ids)# 现在可以使用 token2idx 将标记化的文本转换为整数列表
# input_ids 转换为 one-hot 向量的 2D 张量
categorical_df = pd.DataFrame(
    {"Name": ["Bumblebee", "Optimus Prime", "Megatron"], "Label ID": [0,1,2]})
# print(categorical_df)
# pd.get_dummies(categorical_df["Name"])

import torch
import torch.nn.functional as F
input_ids = torch.tensor(input_ids)
one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))
# print(one_hot_encodings.shape)

#通过检查第一个向量，我们可以验证 1 出现在 input_ids[0] 指示的位置
# print(f"Token: {tokenized_text[0]}")
# print(f"Tensor index: {input_ids[0]}")
# print(f"One-hot: {one_hot_encodings[0]}")

print('------------Word Tokenization-----------')
# 一种简单的分词方法就是使用空格来标记文本
tokenized_text = text.split()
# print(tokenized_text)

print('------------Subword Tokenization-----------')
#Subword分词背后的基本思想是结合字符和词标记化的最佳应用

print('------------加载distilbert模型-------------')
# 加载distilbert模型
from transformers import AutoTokenizer

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

#手动加载distilbert模型
# from transformers import DistilBertTokenizer
#
# distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)
encoded_text = tokenizer(text)
# print(encoded_text)#简单的“文本分词是 NLP 的核心任务”来检查这个分词模块是如何工作的

# 我们可以使用分词器的 convert_ids_to_tokens() 方法将它们转换回原始字符
tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
# print(tokens)

# 添加了一些特殊的 [CLS] 和 [SEP] 标记
# print(tokenizer.convert_tokens_to_string(tokens))

#可以检查词汇量
# print(tokenizer.vocab_size)

# 相应模型的最大上下文最大长度
# print(tokenizer.model_max_length)

#模型在其前向传递中需要输入的字段名称
# print(tokenizer.model_input_names)

print('--------------对整个数据集进行分词---------------')

# 一个处理函数来分词我们的文本 DatasetDict 对象的map()方法
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

# print(tokenize(emotions["train"][:2]))#从训练集中传递含有两条数据的batch

#在这里我们可以看到填充的结果
tokens2ids = list(zip(tokenizer.all_special_tokens, tokenizer.all_special_ids))
data = sorted(tokens2ids, key=lambda x : x[-1])
# df = pd.DataFrame(data, columns=["Special Token", "Special Token ID"])
# print(df.T)

# 它应用于语料库中的所有拆分
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

#默认情况下，map() 方法对语料库中的每个示例单独运行，因此设置 batched=True 将对推文进行批量编码
# print(emotions_encoded["train"].column_names)


print('-----训练一个分类器DistilBERT这样的基于编码器的模型的架构-------')
# 两种选择可以在 Twitter 数据集上训练这样的模型：
#
# 特征提取:: 我们使用隐藏状态作为特征，只在它们上训练一个分类器，而不修改预训练模型。
# Fine-tuning:: 我们端到端训练整个模型，这也更新了预训练模型的参数。

print('-------------Transformers作为特征提取工具------------')

#1.使用预训练模型
# hide_output加载 DistilBERT 检查点
from transformers import AutoModel

model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)

#Pytorch与TensorFlow框架切换
# 我们可以使用 TFAutoModel 类在 TensorFlow 中加载 DistilBERT
# from transformers import TFAutoModel
# tf_model = TFAutoModel.from_pretrained(model_ckpt)
# tf_xlmr = TFAutoModel.from_pretrained("xlm-roberta-base", from_pt=True)

# 2.提取最后的隐藏状态 为了热身，让我们检索单个字符串的最后隐藏状态

#对字符串进行编码并将标记转换为 PyTorch 张量
text = "this is a test"
inputs = tokenizer(text, return_tensors="pt")
# print(f"Input tensor shape: {inputs['input_ids'].size()}")

#现在我们将编码作为张量，最后一步是将它们放在与模型相同的设备上
inputs = {k:v.to(device) for k,v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
# print(outputs)

# 我们使用了 torch.no_grad() 上下文管理器来禁用梯度的自动计算。 这对于推理很有用，因为它减少了计算的内存占用

outputs.last_hidden_state.size()
print('-----------------查看隐藏状态张量----------\n',torch.Size([1, 6, 768]))

outputs.last_hidden_state[:,0].size()

# print(torch.Size([1, 768]))#通过简单地索引到 outputs.last_hidden_state 来提取它

#我们将最终的隐藏状态作为 NumPy 数组放回 CPU
def extract_hidden_states(batch):
    # Place model inputs on the GPU
    inputs = {k:v.to(device) for k,v in batch.items()
              if k in tokenizer.model_input_names}
    # Extract last hidden states
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    # Return vector for [CLS] token
    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}


#由于我们的模型需要张量作为输入，接下来要做的是将 input_ids 和 attention_mask 列转换为 "torch" 格式
emotions_encoded.set_format("torch",
                            columns=["input_ids", "attention_mask", "label"])
#然后我们可以继续并一次性提取所有拆分中的隐藏状态：
#hide_output
emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)

#向我们的数据集添加了一个新的 hidden_state 列
# print(emotions_hidden["train"].column_names)

#现在我们已经有了与每条推文相关的隐藏状态向量，下一步是在它们上训练一个分类器。 为此，我们需要一个特征矩阵——让我们来看看

#3.创建特征矩阵
import numpy as np

X_train = np.array(emotions_hidden["train"]["hidden_state"])
X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
y_train = np.array(emotions_hidden["train"]["label"])
y_valid = np.array(emotions_hidden["validation"]["label"])
# print(X_train.shape, X_valid.shape)

# 在我们在隐藏状态上训练模型之前确保它们为我们想要分类的情绪提供有用的表示

#4.可视化训练集
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler

# Scale features to [0,1] range
X_scaled = MinMaxScaler().fit_transform(X_train)
# Initialize and fit UMAP
mapper = UMAP(n_components=2, metric="cosine").fit(X_scaled)
# Create a DataFrame of 2D embeddings
df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
df_emb["label"] = y_train
df_emb.head()

#进一步研究压缩数据并分别绘制每个类别的点密度
fig, axes = plt.subplots(2, 3, figsize=(7,5))
axes = axes.flatten()
cmaps = ["Greys", "Blues", "Oranges", "Reds", "Purples", "Greens"]
labels = emotions["train"].features["label"].names

for i, (label, cmap) in enumerate(zip(labels, cmaps)):
    df_emb_sub = df_emb.query(f"label == {i}")
    axes[i].hexbin(df_emb_sub["X"], df_emb_sub["Y"], cmap=cmap,
                   gridsize=20, linewidths=(0,))
    axes[i].set_title(label)
    axes[i].set_xticks([]), axes[i].set_yticks([])
#这些只是对低维空间的投影相反，如果它们在投影空间中是可分离的，那么它们在原始空间中也是可分离的。
plt.tight_layout()
plt.show()




#现在我们已经对数据集的特征有了一些了解，让我们最终训练一个模型吧
# 1.训练一个简单的分类器
#hide_output 逻辑回归模型似乎比随机模型好一点
# We increase `max_iter` to guarantee convergence
from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(max_iter=3000)
lr_clf.fit(X_train, y_train)

LogisticRegression(max_iter=3000)
# print(lr_clf.score(X_valid, y_valid))

# 在 Scikit-Learn 中有一个 DummyClassifier用于构建具有简单启发式的分类器
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
# print(dummy_clf.score(X_valid, y_valid))

#因此，我们的带有 DistilBERT 嵌入的简单分类器明显优于我们的基线
# 可以通过查看分类器的混淆矩阵来进一步研究模型的性能，它告诉我们真实标签和预测标签之间的关系
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()
y_preds = lr_clf.predict(X_valid)
plot_confusion_matrix(y_preds, y_valid, labels)


print('----------------Transformers微调Fine-tuning-------------------')
# import huggingface_hub
# huggingface_hub.login()
#1.加载预训练模型
# hide_output
from transformers import AutoModelForSequenceClassification
#AutoModelForSequenceClassification在预训练模型输出之上有一个分类头，可以使用基本模型轻松训练。
# os.environ['HF_API_TOKEN'] = "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"
num_labels = 6
model = (AutoModelForSequenceClassification
         .from_pretrained(model_ckpt, num_labels=num_labels)
         .to(device))


#微调期间用于评估模型性能的指标
#2.定义性能指标
# 为了在训练期间监控指标，我们需要为“Trainer”定义一个“compute_metrics()”函数。2
# 该函数接收一个“EvalPrediction”对象（它是一个具有“predictions”和“label_ids”属性的命名元组），并需要返回一个字典，将每个指标的名称映射到它的值。
from sklearn.metrics import accuracy_score, f1_score
#计算 F1-score 和模型的准确性
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

#在定义“Trainer”类之前，我们只需要处理最后两件事
# 在 Hugging Face Hub 上登录我们的帐户。 这将使我们能够将微调后的模型推送到我们在 Hub 上的帐户并与社区共享。
# 定义训练运行的所有超参数。

#要定义训练参数，我们使用“TrainingArguments”类。
from transformers import Trainer, TrainingArguments

batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size
model_name = f"{model_ckpt}-finetuned-emotion"
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

#在这里，我们还设置了批量大小、学习率和 epoch 数，并指定在训练运行结束时加载最佳模型。 有了这个最终成分
from transformers import Trainer

trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded["train"],
                  eval_dataset=emotions_encoded["validation"],
                  tokenizer=tokenizer)
trainer.train()
# F:\Projects\bert-nlp-with-transformers\notebooks\distilbert-base-uncased-finetuned-emotion is already a clone of https://huggingface.co/quincyqiang/distilbert-base-uncased-finetuned-emotion. Make sure you pull the latest changes with `repo.git_pull()`.
# Failed to detect the name of this notebook, you can set it manually with the WANDB_NOTEBOOK_NAME environment variable to enable code saving.
# [34m[1mwandb[0m: Currently logged in as: [33mquincyqiang[0m (use `wandb login --relogin` to force relogin)
# [34m[1mwandb[0m: wandb version 0.12.11 is available!  To upgrade, please run:
# [34m[1mwandb[0m:  $ pip install wandb --upgrade

# 查看日志，我们可以看到我们的模型在验证集上的
# -score 约为 92% - 这是对基于特征的方法的显着改进！
preds_output = trainer.predict(emotions_encoded["validation"])
# print(preds_output)#返回几个我们可以用于评估的有用对象

# print(preds_output.metrics)

#贪婪地解码预测
y_preds = np.argmax(preds_output.predictions, axis=1)
print(y_valid)
#通过预测，我们可以再次绘制混淆矩阵：
plot_confusion_matrix(y_preds, y_valid, labels)

# print('----------------使用Keras进行微调-----------------')
# 使用 TensorFlow，也可以使用 Keras API 微调您的模型。
# 首先将 DistilBERT 作为 TensorFlow 模型加载
# #hide_output
# from transformers import TFAutoModelForSequenceClassification

# tf_model = (TFAutoModelForSequenceClassification
#             .from_pretrained(model_ckpt, num_labels=num_labels))

# 我们将数据集转换为 tf.data.Dataset 格式
# # The column names to convert to TensorFlow tensors
# tokenizer_columns = tokenizer.model_input_names

# tf_train_dataset = emotions_encoded["train"].to_tf_dataset(
#     columns=tokenizer_columns, label_cols=["label"], shuffle=True,
#     batch_size=batch_size)
# tf_eval_dataset = emotions_encoded["validation"].to_tf_dataset(
#     columns=tokenizer_columns, label_cols=["label"], shuffle=False,
#     batch_size=batch_size)

# 我们还对训练集进行了洗牌，并为其定义了批量大小和验证集
# 编译和训练模型
# #hide_output
# import tensorflow as tf

# tf_model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=tf.metrics.SparseCategoricalAccuracy())

# tf_model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=2)

#4.误差分析
# 要了解它是如何工作的，我们首先将 DistilBERT 作为 TensorFlow 模型加载
from torch.nn.functional import cross_entropy

def forward_pass_with_label(batch):
    # Place all input tensors on the same device as the model
    inputs = {k:v.to(device) for k,v in batch.items()
              if k in tokenizer.model_input_names}

    with torch.no_grad():
        output = model(**inputs)
        pred_label = torch.argmax(output.logits, axis=-1)
        loss = cross_entropy(output.logits, batch["label"].to(device),
                             reduction="none")

    # Place outputs on CPU for compatibility with other dataset columns
    return {"loss": loss.cpu().numpy(),
            "predicted_label": pred_label.cpu().numpy()}

#再次使用 map() 方法，我们可以应用此函数来获取所有样本的损失：
#hide_output
# Convert our dataset back to PyTorch tensors
emotions_encoded.set_format("torch",
                            columns=["input_ids", "attention_mask", "label"])
# Compute loss values
emotions_encoded["validation"] = emotions_encoded["validation"].map(
    forward_pass_with_label, batched=True, batch_size=16)

# 最后，我们创建一个带有文本、损失和预测/真实标签的“DataFrame”：
emotions_encoded.set_format("pandas")
cols = ["text", "label", "predicted_label", "loss"]
df_test = emotions_encoded["validation"][:][cols]
df_test["label"] = df_test["label"].apply(label_int2str)
df_test["predicted_label"] = (df_test["predicted_label"]
                              .apply(label_int2str))
# 5.保存模型
#hide_output
trainer.push_to_hub(commit_message="Training completed!")

#使用微调模型对新推文进行预测
#hide_output
from transformers import pipeline

# Change `transformersbook` to your Hub username
model_id = "transformersbook/distilbert-base-uncased-finetuned-emotion"
classifier = pipeline("text-classification", model=model_id)

#示例推文测试管道
custom_tweet = "I saw a movie today and it was really good."
preds = classifier(custom_tweet, return_all_scores=True)

#在条形图中绘制每个类别的概率。
preds_df = pd.DataFrame(preds[0])
plt.bar(labels, 100 * preds_df["score"], color='C0')
plt.title(f'"{custom_tweet}"')
plt.ylabel("Class probability (%)")
plt.show()