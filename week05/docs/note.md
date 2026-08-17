# 作业1: 完整运行下 10_BERT文本分类.py， 理解微调过程。可以调整超参数（学习率或者batch size），对比模型精度。 截图

> Owen 3.7 3.8的版本 -》 本地部署 / 本地微调 -〉 超参数对最终精度的影响
>
> 超参数 -》 人工设置 / 理想范围



| 参数             | 推荐搜索范围 （对于bert模型） | 主要目的                   |
| ---------------- | ----------------------------- | -------------------------- |
| `learning_rate`  | `1e-5 ~ 5e-5`                 | 控制收敛效果               |
| `batch_size`     | `16 / 32 / 64`                | 平衡效果、吞吐和显存       |
| `max_seq_length` | `128 / 256 / 512`             | 平衡信息完整度和计算成本   |
| `epochs`         | `2 / 3 / 4`                   | 控制训练充分程度           |
| `weight_decay`   | `0 / 0.01 / 0.05`             | 控制过拟合                 |
| `warmup_ratio`   | `0 / 0.05 / 0.1`              | 提升训练稳定性，动态学习率 |
| `freeze_layers`  | `0 / 3 / 6 / 9`               | 平衡效果和训练成本         |



最初的bert模型 ，使用模型默认的超参数 -》 94%

超参数调整， 使用调整后的超参数 -》 96%



# 作业2: 基于01 意图识别项目，尝试自己vibe coding 写出完整过程。 提交代码结构截图

- token plan / coding plan -》 按照时间时间为范围，限制小时 / 天/ 周 消耗额度， 包年包月 会员

- api 计费 -〉 按照token消耗 直接计算，提前充值



确定项目需求和核心功能，明确用户输入、系统处理过程以及最终输出结果 -》 **生成 CLAUDE.md 项目全局的记忆 -〉 给大模型看**

确定数据结构，通过 Pydantic 定义请求和响应 Schema，进而确定前后端交互逻辑。

```
class FAQBase(BaseModel):
    """FAQ 基础模型"""
    title: str = Field(..., max_length=255, description="标准问标题")
    category_id: int = Field(..., description="关联类目ID")
    similar_queries: Optional[List[str]] = Field(default_factory=list, description="相似问列表")
    related_ids: Optional[List[int]] = Field(default_factory=list, description="关联问题ID列表")
    tags: Optional[List[str]] = Field(default_factory=list, description="标签列表")
    is_permanent: bool = Field(default=True, description="是否永久生效")
```



basedpyright --level error



【初步编码，不能保证最终效果】确定 API 接口，包括接口路径、请求方法、输入字段、返回字段以及异常返回形式。

确定代码整体框架，规划项目目录、每个文件的组成以及模块之间的职责划分。

搭建最小可运行项目，优先完成程序入口、API、Schema 和 Mock 数据，使核心链路能够先运行起来。

分模块进行 Vibe Coding，按照 API、Service、Model、前端页面等模块逐步让 Coding Agent 完成代码实现。

每完成一个模块立即进行运行验证，检查 Import、语法、接口调用以及模块之间的数据传递是否正常。



【效果验证】实现核心意图识别逻辑，将模型推理与 API、业务逻辑解耦，并通过统一接口返回意图和置信度。

完成前后端联调，通过真实请求验证前端参数、Pydantic Schema、后端返回结构是否完全一致。

补充异常处理，验证空输入、非法参数、未知意图、模型调用失败等异常场景是否能够正确处理。

编写单元测试和接口测试，覆盖正常 Case、边界 Case 和异常 Case，并运行测试验证核心功能。

根据测试失败结果让 Coding Agent 定位问题并修改代码，形成“修改代码 → 运行测试 → 分析错误 → 修复代码 → 再次测试”的迭代闭环。





# bert （预训练 和 微调 ，网络结构不一样， 训练任务不同）

- 判别式 / 编码任务
- 使用场景：
  - 输入文本，输出 类别
  - 输入 文本，输出 每个token类别
  - 输入 文本1 文本2， 输出 句子对 类别



![img](https://raw.githubusercontent.com/huggingface/sentence-transformers/main/docs/img/Bi_vs_Cross-Encoder.png)

bi - encoder ， 双塔结构 ，孪生网络： 文本编码

cross - encoder，原声bert nsp 的用法： 文本对的分类

embedding模型，目标：数据库中有100个文本， 用户有一个待查询文本， 需要从数据库中找到与 待查询最相似。

步骤1: bert 微调后的模型 对100 + 1 编码

步骤2: 计算相似度

复杂度： **数据库中有100个文本** 可以提前做编码（100 * 256）， **待查询文本 实时编码 （1 * 256）， 一次矩阵乘法**

https://huggingface.co/BAAI/bge-small-zh-v1.5



rerank模型，目标：数据库中有100个文本， 用户有一个待查询文本， 需要从数据库中找到与 待查询最相似。

步骤1:  100个文本后待查询 拼接作为输入

步骤2: 进行100次bert 正向传播，得到打分

复杂度：100次实时编码

https://huggingface.co/BAAI/bge-reranker-base



# GPT（预训练 和 微调，网络结构一样）

- 生成式
- 使用场景：
  - 输入 文本， 输出 文本的翻译
  - 谁让 文本，输出 文本的下一句



**预测下一个词（Next Token Prediction）**

给定一个完整的文本序列（例如“我今天吃了”），模型通过掩码自注意力，在一次前向传播中，同时预测每个位置的下一个词（“我”→“今”，“我今天”→“吃”……）。它学会的是**统计上的条件概率** P(下一个词∣之前所有词)*P*(下一个词∣之前所有词)。



<eos> -> 句子结束



输入提示词 `“法国的首都是”`
→ 生成 `“巴”`（输入变为“法国的首都是巴”）
→ 生成 `“黎”`（输入变为“法国的首都是巴黎”）
→ 生成 `<EOS>` 结束。



强化学习（RL） 有监督微调（SFT）



# https://help.aliyun.com/zh/document_detail/421161.html



![1](https://help-static-aliyun-doc.aliyuncs.com/assets/img/zh-CN/0982367261/p299627.png)

![img](https://help-static-aliyun-doc.aliyuncs.com/assets/img/zh-CN/1982367261/p299629.png)



- uvicorn， 运行和托管现代异步 Python Web 应用程序

- fastapi router （子模块）， 用于将相关的路径操作（路由）组织在一起

- sqlalchemy， orm 框架，ORM（Object-Relational Mapping，对象关系映射）， class 与 数据库表 进行映射
- sentence_transformers， 本地的bert /gpt 类型模型的加载，进行文本编码
- Elasticsearch，是一个文档数据库， 类似 mysql 的中间件； ES 可以存储文档，也可以存储向量
  - **Domain-Specific Language**（领域特定语言）
  - es 支持 倒排，支持 tfidf 和 bm25打分 -》 很适合文档检索
  - es 支持 向量检索，支持近似检索的算法 -〉适合向量检索（语义检索 / 近似检索）

![What is ORM? A Beginner's Guide to Object-Relational Mapping | by Jesse  Onoyeyan | Medium](https://miro.medium.com/v2/resize:fit:1400/1*DDXrDtTtnKsdu60IXvTTiQ.png)

![API: Beginner to Advance: Understanding Object-Relational Mappers (ORMs) in  Python Applications- Part 6 | by Talib | Bootcamp | Medium](https://miro.medium.com/v2/resize:fit:1400/1*gkGdsM-diyWasjcISU222g.png)



如下代码，理解即可：

-rw-r--r--@ 1 lyz  staff   3.2K Aug 16 11:26 01_激活函数.py
-rw-r--r--@ 1 lyz  staff   1.0K Aug 16 11:01 02_归一化层.py
-rw-r--r--@ 1 lyz  staff   4.5K Feb  8  2026 03_BERT进阶.py
-rw-r--r--@ 1 lyz  staff    10K Aug 16 14:02 05_GPT预训练.py
-rw-------@ 1 lyz  staff    10K Aug 16 14:05 05_RLHF.py



偏向实操，可以运行一下：

-rw-r--r--@ 1 lyz  staff   1.6K Feb  8  2026 04_SBERT使用.py
-rw-r--r--@ 1 lyz  staff   2.5K Aug 16 14:13 06_ChatGPT_HTTP_API.py
-rw-r--r--@ 1 lyz  staff   827B Feb  8  2026 07_ChatGPT_OpenAI_API.py
-rw-r--r--@ 1 lyz  staff   3.2K Mar 15 14:52 07_ChatGPT_进阶API.py
-rw-r--r--@ 1 lyz  staff   730B Sep  4  2025 08_Qwen-Kimi-DeepSeek使用.py
-rw-r--r--@ 1 lyz  staff   826B Aug 16 14:31 09_本地Qwen大模型.py



# 作业

1. 本地安装下 sentence-transformer库，使用bge模型进行文本检索，不需要es

```
modelscope download --model BAAI/bge-small-zh-v1.5  --local_dir BAAI/bge-small-zh-v1.5
```

待检索的文本：我今天很开心

数据库文本：

- 我喜欢机器学习
- 我喜欢深度学习
- 我今天心情很不错



2. 本地安装Ollama， 本地运行 qwen3-0.6b 完成 sdk调用

```
from openai import OpenAI

# 初始化客户端，指向 Ollama 的本地服务
client = OpenAI(
    base_url="http://localhost:11434/v1",  # Ollama API 地址
    api_key="1111"  # Ollama 默认无需真实 API Key，填任意值即可
)

# 发送请求
response = client.chat.completions.create(
    model="qwen3:0.6b",  # 指定模型
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "你好"}
    ],
    temperature=0.7,  # 控制生成多样性
    max_tokens=512    # 最大生成 token 数
)

# 打印结果
print(response.choices[0].message.content)
```









sentence-transformers 或 vllm 部署 编码模型 和 rerank模型





> 学习率和动态学习率 可以都设置吗?

可以，**动态学习率 如何调整学习率的策略**



>  老师 pydantic 是类型控制吗？需要重点关注吗

需要，应用的程序：数据结构 + 逻辑控制 / 业务逻辑



>  能不能示范一下，怎么操作，示范一下完整操作过程

这个要几个小时。。。



> 老师推荐使用哪个 AI 工具来做 vibe coding 开发呢

claude code / codex / dsh



> 为什么解码器适合文本生成、编码器适合文本分类、编码器解码器适合机器翻译？

解码器（Decoder-Only）适合生成，因为它必须依赖“过去”预测“未来”，这与人类说话的逻辑完全一致；编码器（Encoder-Only）适合分类，因为它能同时看到“左右”上下文，提取的语义特征最全面；编码器-解码器（Encoder-Decoder）适合翻译，因为它用“独立编码器”彻底理解源语言，再用“交叉注意力”动态对齐目标语言，实现了信息的强解耦。



>  老师如果只有decoder的话，那么编码器的输入现在变成了什么呢？

参考自回归的输入



> 感觉glm很强大啊，尤其5.1   5.2，为啥enc-dnc架构会被逐步替代。

ppt 列举是 早期的glm ，和现在的glm 不同，现在glm 也是decoder only



> 模型后面框出来的数字是什么意思



> 编码器和解码器本质区分是什么呢，没有概念，只知道有个编解码划分





> Bert如果是判别式的，那为什么适合编码呢

gpt 也适合 文本编码， bert 速度更快



> seq2seq是哪种模型架构

**Seq2Seq（Sequence-to-Sequence，序列到序列）不是“某一种”具体的模型架构，而是一类“任务范式”或“框架”的总称。** 在 Transformer 出现之前，它特指基于 RNN/LSTM 的 Encoder-Decoder 架构；在 Transformer 时代，**Encoder-Decoder 是 Seq2Seq 最正宗、最直接的实现**，但**Decoder-Only 在特定条件下也能承担 Seq2Seq 任务**。



>  老师像手机相册归纳分类是使用 bert 模型吗

bert 模型适用于文本分类，你说的是图像分类，图像分类的模型，cnn / vit



>  bge是适合用来语义检索吗

是的



> 解码器-only 为什么能不通过编码器就能预测下一个token?原理是什么?

**Decoder-Only**：编码是**隐式的**（利用深度自注意力，在每一层逐步提炼历史信息）。它不需要把整个句子压缩成一个固定向量，而是**保留所有历史 Token 的动态表示**，在生成每一步时动态调用。



> 问 老师，编码器是不是就是语义关系特征

可以理解为提取特征



>  双胎模型的训练和推理的最后输出为什么是两种不同的结构？



>  意图识别用的哪种bert

bert 微调 分类，不是sbert



> 文本编码，是文本做向量化吗？

yes



>  实际工作过程中，BM25和sentence-bert，哪个使用场景多呢？
>
> sentence-bert和BM25分别在哪个场景使用啊，举例了解一下吧

可以一起使用



> Cross Encoder是不是就是语义排序模型？

yes



>  cross-bert 文本对分类是那个根据上文判断下个句子是不是上个句子续写的任务吗

next sentence  pridiction

<句子 1， 句子2> -》 句子对列别



>  老师，数据库中有 100 个文本，可以提前做编码 100 × 256 ， 为什么是 256呀





>  为什么是一次矩阵乘法而不是100次呢，计算余弦相似度不是得一对一对的算吗



> sentence Bert就没有注意力机制来？

Bert 自己是有注意力机制，sbert肯定有

sbert 没有修改模型结构，但在训练过程和使用上有修改； 是Bert的使用方式



> 为什么cross模型把100个拼接起来,是有分隔符的嘛?然后计算不同句子的关联关系?

数据库有100个文本， t1， t2 。。。 t100

待查询文本 query



<t1, query> -> bert -> 匹配的打分

<t2, query> -> bert -> 匹配的打分

<t3, query> -> bert -> 匹配的打分

<t100, query> -> bert -> 匹配的打分



>  Bi-Encoder是用来做召回的吗？Cross-Encoder做精排（还是重排来着）？

yes



> bi-encoder把每次查询文本的相似度也存到原始文本中，会不会增加之后查询的准确率

业务逻辑



> GPT掩码自注意力是在训练时用到的吧，推理的时候应该不用吧？



BERT中的[CLS]和[SEP] 也是特殊token吗？

yes



> 视频生成也是基于decoder吗？



> 老师 直观上感觉编解码器 比只编码或 解码应该更好，现在实际上都不用了，可以解释下原理吗？

decoder only 不是 没有encoder； gpt 还是会对文本进行有效的语言编码；



> 能讲下池化技术吗

![A improved pooling method for convolutional neural networks | Scientific  Reports](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-024-51258-6/MediaObjects/41598_2024_51258_Fig1_HTML.png)



![Illustration of Max Pooling and Average Pooling Figure 2 above shows an...  | Download Scientific Diagram](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ8TU5JVqqG1cXMbzaHr4x3wX37JujdFN1hFBEKavvbGptqQoDi80B90Ara&s=10)

![Pooling (CNN) — EpyNN 1.0 documentation](https://epynn.net/_images/pool-01.svg)

> 编码器为什么是文本维度降低？我之前理解的是升维，升维后才能体现更多含义

**比如一句话有10个词经过编码器后，通常会被压缩成1个或几个固定的向量**



> 老师能总结一下各个大模型分别适合哪方面的vibe coding开发吗

都适合



>  老师有没有过多终端 vibecoding 协同开发工具推荐？

claude code / codex / dsh



> 那我可以理解为编码器解码器本质区分就是多头注意力机制中掩码器控制隐藏哪部分文本？

- **编码器（Encoder）**：使用**全可见掩码（无掩码）**。所有输入词可以互相看到对方（双向注意力），目的是“理解整句话”。
- **解码器（Decoder）**：使用**因果掩码（Causal Mask / 下三角掩码）**。每个词只能看到自己和前面的词（单向注意力），目的是“按顺序生成”。



>  SBert训练时候的A,B句子是2个不同的句子还是一个相同的句子训练2次？

<s1, s2> 是否相似



>  预训练是模型厂商做，后训练是我们这些后端开发去做？

大模型应用开发 / agent 开发，可以直接用大模型，或者微调（后训练）



>  后训练和提示词工程、harness工程的差别是什么？



>  为什么说指令微调之后得到“初步的大模型”，初步的大模型和base modole的区别是什么？



> 奖励模型和重排序有什么不同吗

重排序 是 从相似度角度拍序



> 强化学习和训练奖励函数的区别是什么



> 归一化是为了干什么呢？

稳定性 和 效果



>  老师 你觉得我有必要去学学高数，真感觉后学是数学的较量

基础的线性代数 和矩阵运算



> 强化学习在大模型中也有用到吗，比如claude code

pretrain + postrain（sft、rl） -》 大模型

claude code 是 vibe coding的工具，不是模型



> 打分模型可以自己给自己打分嘛？
>
> 意识是训练的过程中损失中加入强化学习打分就是带强化练习的训练吗





> 老师 之前旧的模型 如 RNN LSTM GRU 这些都不用学了吗

第1节课 第2节课



> 服务器Linux上怎么本地部署

https://autodl.com/console/instance/list

带 gpu 的 linux 服务器



> 老师远程部署的话推荐两个都需要会吗

vllm / sglang 掌握其中一个即可



> 老师个人学习模型部署 有什么好的服务器推荐
>
> 老师这种服务器级别的gpu最低的算力要求要几张gpu，能说下吗，对算力没什么概念

autodl ，4090 先试试



> 一般企业里是用哪种方式调用模型？调用云端的模型还是本地？一般是考虑到哪些情况做选择？

云端模型，计费，没有部署成本； 速度会慢，隐私安全；

本地模型，只有部署成本；速度会快，效果会差；



> 刚才调用3B模型时，模型名右面少了个引号



curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "帮我介绍机器学习"}
        ]
    }'



ollama / llama.cpp 非gpu 场景多

sglang / vllm 在gpu 场景多 -》 对模型的性能优化做的好



> docqa，用的是什么算法，在文档中抽取相关段落的？

可以用bert 实现



> 没有后端基础 我应该怎么知道使用什么最好呢



> ORM用的不多，复杂SQL比较难实现，后期维护排查也不容易。我这边还是Mybatis用的比较多



> VLLM是怎么部署的，在本机测试只能通过租用服务器部署吗

https://docs.vllm.ai/en/latest/getting_started/installation/gpu/

vllm / sglang 推荐 在 linux 和 带有gpu 的机器上部署，不推荐在本地笔记本上使用



> python在企业中一般都使用什么现场的底座呢？比如java中的若依，芋道等封装好的架子呢？

python的生态开放，没有大而全的框架



>  bert是用来文本分类，sbert是用来进行句子相似度计算的吗？还有其他的吗

是的



向量/到排序/tokenizer/embedding是什么关系？



写项目的话是用这几周讲的项目吗，rag和agent没学怎么写一个更好的项目，后续的课程有更好的项目吗

下节课



es一般存什么？文本类吗？会需要把传统数据库转es吗

文档、结构化



RAG项目是建议使用langchain来连接各个模块还是像这里自己用脚本组合？

都可以



老师，之前了解一些专用于存储向量的向量数据库，那到时候做项目，是用ES还是其它向量数据库？

miluvs 后序也会讲解，擅长向量检索

es 擅长文档检索，也可以支持向量检索



SentenceTransformers加载模型和BertForSequence...有区别吗？

本质没有区别， SentenceTransformers 使用更加方便



老师，预训练通过反向传播计算权重，后训练在预训练基础上如何改变权重呢

post training 也有标注，也有训练过程



Transformers的模型加载和ollama本地模型部署的区别是什么

Transformers 是一个训练框架，加载模型进行推理 / 训练 /微调

ollama / vllm / sglang / llama.cpp 单纯是一个推理框架



huggingface 下载的模型是权重吗？为什么下载后，使用vllm就可以部署了，不用模型的源码吗

vllm 包括常见模型的代码，vllm serve 模型路径



比如简历写了bert ，写es -》 es 面试题，bert面试题



背面试题过了，但是实际能力不行，实际工作中还是写不了项目怎么办呢



SentenceTransformers和Transformers，做模型微调哪种用的多？有区别吗？

SentenceTransformers: 文本检索、文本匹配、文本匹配的场景

Transformers： 不限制场景



>  vibecoding可以写出代码，但是代码结构怎么样，技术栈怎么样，也不会优化，需要怎么提升

提升自己的品味，只要什么是好代码？推荐用的框架和中间件是什么？

