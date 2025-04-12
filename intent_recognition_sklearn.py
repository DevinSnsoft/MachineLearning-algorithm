import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, Perceptron
from sklearn.metrics import accuracy_score, classification_report

# import warnings
# warnings.filterwarnings("ignore")
# 示例数据，使用字典形式
data = {
    "text" : [],
    'label':[]
}

with open('text.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    for i in lines:
        data['text'].append(json.loads(i)['text'])
        data['label'].append(int(json.loads(i)['label']))

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 特征和标签
X = df['text']
y = df['label'].astype(int)  # 转换为整型标签

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 向量化文本数据
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 训练逻辑回归模型
# model = LogisticRegression()
# model = SGDClassifier()
# model = RidgeClassifier()
model = Perceptron()
model.fit(X_train_tfidf, y_train)

# 进行预测
y_pred = model.predict(X_test_tfidf)

# 输出结果
print("预测标签:", y_pred)
# print("实际标签:", y_test.values)
print("准确率:", accuracy_score(y_test, y_pred))
print("分类报告:")
print(classification_report(y_test, y_pred))