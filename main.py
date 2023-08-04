# 1 Preparation
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
# 1.2  Prepare for NebulaGraph as Graph Store  # 准备 GraphStore
# os.environ['NEBULA_USER'] = "root"
# os.environ['NEBULA_PASSWORD'] = "nebula" # default password
# os.environ['NEBULA_ADDRESS'] = "127.0.0.1:9669" # assumed we have NebulaGraph installed locally
#
# space_name = "guardians"
# edge_types, rel_prop_names = ["relationship"], ["relationship"] # default, could be omit if create from an empty kg
# tags = ["entity"] # default, could be omit if create from an empty kg

# graph_store = NebulaGraphStore(space_name=space_name, edge_types=edge_types, rel_prop_names=rel_prop_names, tags=tags)
#
# storage_context = StorageContext.from_defaults(graph_store=graph_store)
#
# # 从维基百科下载、预处理数据
# from llama_index import download_loader
#
# WikipediaReader = download_loader("WikipediaReader")
#
# loader = WikipediaReader()
#
# documents = loader.load_data(pages=['Guardians of the Galaxy Vol. 3'], auto_suggest=False)
#
# # print(documents) 'text'
# # time.sleep(100)
#
# # 利用 LLM 从文档中抽取知识三元组，并存储到 GraphStore（NebulaGraph）
# kg_index = KnowledgeGraphIndex.from_documents(
#     documents,
#     storage_context=storage_context,
#     max_triplets_per_chunk=10,
#     service_context=service_context,
#     space_name=space_name,
#     edge_types=edge_types,
#     rel_prop_names=rel_prop_names,
#     tags=tags,
#     include_embeddings=True,
# )

if __name__ == '__main__':
    # from langchain.chat_models import ChatOpenAI
    # from langchain.chains import NebulaGraphQAChain
    # from langchain.graphs import NebulaGraph

    from langchain.chat_models import ChatOpenAI
    from langchain.chains import GraphSparqlQAChain
    from langchain.graphs import RdfGraph

    graph = RdfGraph(
        # source_file="http://www.w3.org/People/Berners-Lee/card",
        # source_file="C:/Users/64367/Desktop/Demo_llm2kg/Example1 N-Triples.nt",
        # source_file="C:/Users/64367/Desktop/Demo_llm2kg/disambiguations_en.nt",
        source_file="C:/Users/64367/Desktop/Demo_llm2kg/Example1 N-Triples.nt",
        standard="rdf",
        local_copy="test.ttl",
    )

    # graph = NebulaGraph(
    #     space=space_name,
    #     username="root",
    #     password="nebula",
    #     address="127.0.0.1",
    #     port=9669,
    #     session_pool_size=30,
    # )

    chain = GraphSparqlQAChain.from_llm(
        llm, graph=graph, verbose=True
    )

    # chain = NebulaGraphQAChain.from_llm(
    #     llm, graph=graph, verbose=True
    # )

    # chain.run(
    #     "Tell me about Peter Quill?",
    # )

    # chain.run(
    #     "Save that the person with the name 'Timothy Berners-Lee' has a work homepage at 'http://www.w3.org/foo/bar/'"
    # )

    chain.run("Tell me about the career of 罗纳尔多·路易斯·纳萨里奥·德·利马")

    # query = (
    #     """PREFIX foaf: <http://xmlns.com/foaf/0.1/>\n"""
    #     """SELECT ?hp\n"""
    #     """WHERE {\n"""
    #     """    ?person foaf:name "Timothy Berners-Lee" . \n"""
    #     """    ?person foaf:workplaceHomepage ?hp .\n"""
    #     """}"""
    # )
    # print(graph.query(query))


    # Graph RAG
    # Apart from the NL2Cypher fashion of exploiting KG in QA, especially for complex tasks,
    #     we could also do it in the Retrieval Arguments Generation way.

    # from llama_index import load_index_from_storage
    #
    # storage_context_graph = StorageContext.from_defaults(persist_dir='./storage_graph', graph_store=graph_store)
    # kg_index_new = load_index_from_storage(
    #     storage_context=storage_context_graph,
    #     service_context=service_context,
    #     max_triplets_per_chunk=10,
    #     space_name=space_name,
    #     edge_types=edge_types,
    #     rel_prop_names=rel_prop_names,
    #     tags=tags,
    #     include_embeddings=True,
    # )
    #
    # kg_rag_query_engine = kg_index_new.as_query_engine(
    #     include_text=False,
    #     retriever_mode='keyword',
    #     response_mode="tree_summarize",
    # )
    #
    # response = kg_rag_query_engine.query(
    #     "Tell me about Peter Quill?"
    # )
    # display(Markdown(f"<b>{response}</b>"))
