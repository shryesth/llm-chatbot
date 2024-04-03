import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (
    HuggingFaceEmbeddings,
)

# from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()


class AIAgent:
    def __init__(self):
        model_id = "gg-hf/gemma-2b-it"
        hf_token = os.getenv("GEMMA_TOKEN")
        dtype = torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            device_map="cpu",
            torch_dtype=dtype,
        )

    def create_prompt(self, query, context):
        prompt = f"""In the following, you receive a prompt. 
        Answer it based on the given content. Provide only the response, don't say 'Answer:'.
        Question: {query}
        Context: {context}
        Answer:
        """
        return prompt

    def generate(self, query, retrieved_info, max_new_tokens=200):
        prompt = self.create_prompt(query, retrieved_info)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        answer = self.model.generate(
            input_ids, max_new_tokens=max_new_tokens
        )
        answer = self.tokenizer.decode(answer[0], skip_special_tokens=True)
        return answer


class RAGSystem:
    def __init__(
        self,
        ai_agent,
        chroma_db_directory="data/vectordb/processed/chroma",
        num_retrieved_docs=3,
    ):
        self.num_docs = num_retrieved_docs
        self.ai_agent = ai_agent
        self.template = (
            "\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\nContext:\n{context}"
        )
        self.vector_db = Chroma(
            persist_directory=chroma_db_directory,
            embedding_function=HuggingFaceEmbeddings(
                model_name="BAAI/bge-large-en-v1.5"
            ),
        )
        self.retriever = self.vector_db.as_retriever()

    def retrieve(self, query):
        docs = self.retriever.get_relevant_documents(query)
        print("query: " + query)
        # docs = self.retriever.similarity_search(query)
        return docs

    def query(self, query):
        context = self.retrieve(query)
        print(context)
        answer = self.ai_agent.generate(query, context)
        print("answer:" + answer)
        return self.template.format(question=query, answer=answer, context=context)
