# 1 Preparation

import GstoreConnector
# 1.1 Prepare for LLM

# Only For Azure OpenAI
# import os
# import json
# import openai
# from langchain.llms import AzureOpenAI
# from langchain.embeddings import OpenAIEmbeddings
# from llama_index import LangchainEmbedding
# from llama_index import (
#     VectorStoreIndex,
#     SimpleDirectoryReader,
#     KnowledgeGraphIndex,
#     LLMPredictor,
#     ServiceContext
# )
#
# from llama_index.storage.storage_context import StorageContext
# from llama_index.graph_stores import NebulaGraphStore
#
# import logging
# import sys
#
# from IPython.display import Markdown, display
#
# logging.basicConfig(stream=sys.stdout, level=logging.INFO) # logging.DEBUG for more verbose output
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
#
# openai.api_type = "azure"
# openai.api_base = "INSERT AZURE API BASE"
# openai.api_version = "2022-12-01"
# os.environ["OPENAI_API_KEY"] = "INSERT OPENAI KEY"
# openai.api_key = os.getenv("OPENAI_API_KEY")
#
# # define LLM
# llm = AzureOpenAI(
#     deployment_name="INSERT DEPLOYMENT NAME",
#     temperature=0,
#     openai_api_version=openai.api_version,
#     model_kwargs={
#         "api_key": openai.api_key,
#         "api_base": openai.api_base,
#         "api_type": openai.api_type,
#         "api_version": openai.api_version,
#     }
# )
# llm_predictor = LLMPredictor(llm=llm)
#
# # You need to deploy your own embedding model as well as your own chat completion model
# embedding_llm = LangchainEmbedding(
#     OpenAIEmbeddings(
#         model="text-embedding-ada-002",
#         deployment="INSERT DEPLOYMENT NAME",
#         openai_api_key=openai.api_key,
#         openai_api_base=openai.api_base,
#         openai_api_type=openai.api_type,
#         openai_api_version=openai.api_version,
#     ),
#     embed_batch_size=1,
# )
#
# service_context = ServiceContext.from_defaults(
#     llm_predictor=llm_predictor,
#     embed_model=embedding_llm,
# )

# Only For OpenAI

import os
import time


# 定义llm
# 准备一个graph_store用来存放gstore
# 需要一个RDF三元组数据集
# 把RDF三元组数据集存入gstore中
# 构建storage_context
# 利用KnowledgeGraphQueryEngine（storage_context,service_context,llm=llm）进行查询

os.environ['OPENAI_API_KEY'] = "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"

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO) # logging.DEBUG for more verbose output

from llama_index import (
    KnowledgeGraphIndex,
    LLMPredictor,
    ServiceContext,
    SimpleDirectoryReader,
)

from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore


from langchain import OpenAI
from IPython.display import Markdown, display

# define LLM
llm = OpenAI(temperature=0, model_name="text-davinci-002")
llm_predictor = LLMPredictor(llm)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=512)

# 利用 LLM，几行代码构建知识图谱

gc = GstoreConnector.GstoreConnector("127.0.0.1", 9000, "ghttp", "root", "123456")
# # 参数含义：[服务器IP]，[服务器上http端口]，[http服务类型]，[用户名]，[密码]
# 功能：初始化

res = gc.build("demo", "C:/Users/64367/Desktop/llm_kg_llama_index/Example1 N-Triples.nt")
# 功能：通过RDF文件新建一个数据库
# 参数含义：[数据库名称]，[.nt文件路径]，[请求类型"GET"和"post",如果请求类型为“GET”，则可以省略]


# 1.2  Prepare for NebulaGraph as Graph Store  # 准备 GraphStore

storage_context = StorageContext.from_defaults(graph_store=gc)


if __name__ == '__main__':

    # Now we have a Knowledge Graph built on top of Wikipedia. With NebulaGraph LLM tooling, we could query the KG in Natural language(NL2Cypher).

    # First, let's use Llama Index:
    from llama_index.query_engine import KnowledgeGraphQueryEngine

    from llama_index.storage.storage_context import StorageContext
    from llama_index.graph_stores import NebulaGraphStore

    nl2kg_query_engine = KnowledgeGraphQueryEngine(
        storage_context=storage_context,
        service_context=service_context,
        llm=llm,
        verbose=True,
    )

    # We could see KnowledgeGraphQueryEngine could be used to Generate Graph Query and do query for us
    #     and fianlly LLM could help with the answer synthesis in one go!

    response = nl2kg_query_engine.query(
        "Tell me about the career of 罗纳尔多·路易斯·纳萨里奥·德·利马",
    )

    display(Markdown(f"<b>{response}</b>"))
    #
    # # Apart from the e2e KGQA, we could ask for only NL2Cypher like this with generate_query.
    #
    # graph_query = nl2kg_query_engine.generate_query(
    #     "Tell me about Peter Quill?",
    # )
    # graph_query = graph_query.replace("WHERE", "\n  WHERE").replace("RETURN", "\nRETURN")
    #
    # display(
    #     Markdown(
    #         f"""
    # ```cypher
    # {graph_query}
    # ```
    # """
    #     )
