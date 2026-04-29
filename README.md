# 多领域 RAG（场景路由 + 知识隔离）

## 简介
这是一个多场景/多领域 RAG 项目。  
如果将所有 PDF 直接混合进一个向量库，容易出现跨场景知识干扰（例如用户问“医疗报销流程”，却引用“IT 设备申请指南”），从而降低回答准确性和可信度。

本项目通过**场景路由 + 知识隔离**的方式，先识别用户问题所属场景，再只在对应知识库中检索与生成答案。

## 功能特点
- 支持多个领域：IT、金融、HR
- 使用 ChromaDB 作为向量数据库，便于本地开发
- 基于 LlamaIndex 框架，支持多索引、路由、PDF 加载
- 场景分类器采用“规则 + LLM 零样本分类”策略，无需训练，快速上线

## 一、核心思路：场景路由 + 知识隔离
不要把所有知识混在一起，而是：
1. 先判断用户问题属于哪个场景
2. 只检索该场景对应的知识库
3. 用该场景知识生成答案

## 二、架构：分层 RAG + 动态路由
系统分为三层：

1. **路由层（Classifier）**  
   - 先走关键词规则匹配  
   - 匹配不到时，调用 LLM 进行零样本分类兜底

2. **检索层（Scene Index）**  
   - 每个场景（`hr` / `it` / `finance`）维护独立向量索引  
   - 使用 ChromaDB 持久化存储，避免场景间向量混淆

3. **生成层（RAG Answering）**  
   - 根据路由结果选择对应 Query Engine  
   - 仅基于该场景召回内容回答问题

### 执行流程
`用户问题 -> 场景分类 -> 对应场景索引检索 -> LLM 生成答案`

## 三、技术栈（结构清晰）
- **语言**：Python
- **RAG 框架**：LlamaIndex
- **向量数据库**：ChromaDB（本地持久化）
- **大模型/Embedding**：DashScope（Qwen + Embedding）
- **分类策略**：规则关键词 + OpenAI 兼容接口零样本分类
- **配置管理**：`.env` + `python-dotenv`

## 四、项目结构
```text
.
├─ app.py              # 交互入口（CLI）
├─ rag_engine.py       # 多场景索引初始化、检索与问答
├─ classifier.py       # 场景分类（规则 + LLM）
├─ config.py           # 场景定义与数据路径
├─ data/
│  ├─ hr/              # HR 领域文档（PDF）
│  ├─ it/              # IT 领域文档（PDF）
│  └─ finance/         # 财务领域文档（PDF）
└─ storage/            # 各场景索引与 Chroma 持久化目录
```

## 五、快速开始
### 1) 安装依赖
```bash
pip install llama-index chromadb openai python-dotenv
pip install llama-index-llms-dashscope llama-index-embeddings-dashscope
```

### 2) 配置环境变量
在项目根目录创建 `.env`：

```env
DASHSCOPE_API_KEY=your_api_key_here
```

### 3) 准备知识库文档
将各领域 PDF 分别放入：
- `data/hr`
- `data/it`
- `data/finance`

### 4) 运行
```bash
python app.py
```

首次运行会自动构建各场景索引并持久化到 `storage/`；后续会优先加载已有索引。

## 六、示例
- 输入：`公司报销流程是怎样的？`
- 路由：`finance`
- 输出：基于财务知识库生成答案

## 七、可扩展方向
- 新增场景：在 `config.py` 添加场景定义并准备对应数据目录
- 优化路由：引入更细粒度规则或分类置信度策略
- 强化检索：增加 rerank、召回融合、多路检索
- 增强可观测性：记录路由命中、召回片段、答案来源
