import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer # 词频统计
from sklearn.neighbors import KNeighborsClassifier # KNN
from openai import OpenAI
from typing import Union
from llm_config import DEEPSEEK_CONFIG

from fastapi import FastAPI

app = FastAPI()

dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)
print(dataset[1].value_counts())

input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理

vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词， 不是模型
vector.fit(input_sententce.values) # 统计词表
input_feature = vector.transform(input_sententce.values) # 进行转换 100 * 词表大小

model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)


client = OpenAI(
    api_key=DEEPSEEK_CONFIG["api_key"],
    base_url=DEEPSEEK_CONFIG["base_url"],
)

@app.get("/text-cls/ml")
def text_calssify_using_ml(text: str) -> str:
    """
    文本分类（机器学习），输入文本完成类别划分
    """
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]

@app.get("/text-cls/llm")
def text_calssify_using_llm(text: str) -> str:
    """
    文本分类（大语言模型），输入文本完成类别划分
    """
    completion = client.chat.completions.create(
        model="deepseek-v4-flash",  # 模型的代号

        messages=[
            {"role": "user", "content": f"""帮我进行文本分类：{text}

输出的类别只能从如下中进行选择， 除了类别之外下列的类别，请给出最合适的类别。
FilmTele-Play            
Video-Play               
Music-Play              
Radio-Listen           
Alarm-Update        
Travel-Query        
HomeAppliance-Control  
Weather-Query          
Calendar-Query      
TVProgram-Play      
Audio-Play       
Other             
"""},  # 用户的提问
        ]
    )
    return completion.choices[0].message.content

# http://0.0.0.0:8000/text-cls/llm?text=%E5%B8%AE%E6%88%91%E5%AF%BC%E8%88%AA%E5%88%B0%E7%8E%8B%E5%BA%9C%E4%BA%95

"""
步骤1: 数据集的读取和了解
步骤2： 尝试了不同的方法
    方案1: 机器学习的方法， 提取特征+knn
    方案2: llm的提示词方法，写提示词
步骤3: 服务fastpi部署，接口文档
"""