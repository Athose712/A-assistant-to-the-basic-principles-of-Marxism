import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_dashscope.embeddings import DashScopeEmbeddings

os.environ["DASHSCOPE_API_KEY"] = "sk-xxx"

#文件的加载
print("正在从 'mayuan_raw_data' 文件夹加载文档...")
loader = PyPDFDirectoryLoader("./mayuan_raw_data/")
documents = loader.load()
print(f"成功加载 {len(documents)} 份文档。")

#文档的分割
print("正在分割文档...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 每个块的最大字符数
    chunk_overlap=100   # 块之间的重叠字符数，有助于保持上下文连续性
)
#clunk_size和chunk_overlap2个参数可以调整
split_docs = text_splitter.split_documents(documents)
print(f"文档被分割成 {len(split_docs)} 个小块。")

#文本的embedding操作
print("正在初始化文本嵌入模型...")
embeddings = DashScopeEmbeddings(model="text-embedding-v2")

#数据库的创建
print("正在创建并保存向量数据库 (采用分批处理模式)...")

# 定义一个安全的批处理大小
batch_size = 20

# 使用第一批文档来初始化FAISS数据库
# 这会创建数据库的初始结构
print(f"正在处理第一批 {min(batch_size, len(split_docs))} 个文档块...")
vectorstore = FAISS.from_documents(split_docs[:batch_size], embeddings)

# 循环处理剩余的文档块
for i in range(batch_size, len(split_docs), batch_size):
    # 获取当前批次的文档块
    end_index = min(i + batch_size, len(split_docs))
    batch_docs = split_docs[i:end_index]

    print(f"正在处理第 {i + 1} 到 {end_index} 个文档块...")

    # 将当前批次的文档块添加到已存在的数据库中
    vectorstore.add_documents(batch_docs)

    # 在两次API请求之间短暂暂停，避免因请求过于频繁而被服务器限流
    #time.sleep(0.5)

#数据库的储存
vectorstore.save_local("database_agent_mayuan")
print("知识库构建完成，并已保存到本地 'database_agent_mayuan' 文件夹。")