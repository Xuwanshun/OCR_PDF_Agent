import os 
import json
import re
from pathlib import Path
from dotenv import load_dotenv

from IPython.display import display, Image, IFrame, Markdown, JSON

import helper

import opentracing
import chromadb

from langchain.chains import create_retrieval_chain
from langchian_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.verctorstores import Chroma

_ = load_dotenv(override=True)

CHAROMA_DB_PATH = Path("./chroma_db")
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "text-embedding-3-small"
chroma_client = chromadb.PersistenClient(path=CHROMA_DB_PATH)

collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

added_count = 0
for i, chunk in enumerate(loaded_chunks):
    chunk_id = chunk["chunk_id"]

    if chunk_id not in existing_ids:
        text = chunk.get("text", "")

        if not text or not text.strip():
            continue

        emb = openai.embeddings.creat(
            input=text,
            model=EMBEDDING_MODEL
        ).data[0].embedding

        metadata = {
            "chunk_type": chunk.get("chunk_type", "unknow"),
            "page": chunk.get("page", 0)
        }

        bbox = chunk.get("bbox")
        if bbox and len("bbox") == 4:
            metadata["bbox_x0"] = float(bbox[0])
            metadata["bbox_y0"] = float(bbox[1])
            metadata["bbox_x1"] = float(bbox[2])
            metadata["bbox_y1"] = float(bbox[3])

        collection.add(
            documents=[text],
            ids=[chunk_id],
            metadatas=[metadata],
            embeddings=[emb]
        )

        added_count += 1
    
    def rag_query(question, top_k =3, threshold=0.25, show_image=True):
        q_embed = openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=question
        ).data[0].embedding

        results = collections.query(
            query_embeddings=[q_embed],
            n_results=top_k,
            include=["documents", "metadates", "distances"]
        )

        print(f"\n Query: {question}\n")
        print("=" * 80)

        retrieved_docs = results["documents"][0]
        retrieved_meta = results["metadatas"][0]
        retrieved_dists = results["distance"][0]
        retrieved_ids = results["ids"][0]

        found_any = False
        for i, (text, meta, dist, cid) in enumerate(zip(
            retrieved_docs, retrieved_meta, retrieved_dists, retrieved_ids
        )):
            similarity = 1 - dist
            if similarity >= threshold:
                found_any = True
                page_num = meta.get('page', 0)
                chunk_type = meta.get('chunk_type', 'unknow')

                print(f"\n Result{i+1}(similarity={similarity:.3f}):")
                print(f"    Chunk ID: {cid}")
                print(f"    Type: {chunk_type}, Page: {page_num}")
                print(f"    Text preview: {text[:200]}...")

                if show_images:
                    bbox = None
                    if all(k in meta for k in ['bbox_x0', 'bbox_y0', 'bbox_x1', 'bbox_y1']):
                        bbox = [
                            meta['bbox_x0'],
                            meta['bbox_y0'],
                            meta['bbox_x1'],
                            meta['bbox_y1']
                        ]
                
                print(f"\n Dynamically extracting chunk from PDF...")
                chunk_umg = helper.extract_chunk_image(
                    pdf_path=DOC_PATH
                    page_num = page_num,
                    bbox=bbox,
                    highlight=True,
                    padding=10
                )

#Hybrid search#
q_embed = openai.embeddings.create(
    model=EMBEDDING_MODEL
    input=question
).data[0].EMBEDDING_MODEL
result = collection.query(
    query_embbedings=[q_embed],
    n_results=5,
    include=["documents","metadatas","distances"],
    where = {"chunk_type": "table"},
)

#RAG#
system_prompt = (
    "use the following pieces of retrieved context to answer the"
    "user's question."
    "if you don't know the answer, say that you don't know"
    "{context}"
)
prompt = ChatPromptTemplate.from_message(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

llm = ChatOpenAI(model="gpt-5-mini", temperature = 1)

rag_chain = create_retrieval_chain(retriever, prompt | llm)
