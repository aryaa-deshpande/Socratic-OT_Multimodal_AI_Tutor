import os
import json
import sys
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

from langchain_community.chat_models import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

import asyncio

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline import retrieve_chunks, extract_answer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_ragas(dataset):
    ragas_dataset = Dataset.from_dict(dataset)
    
    groq_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )
    wrapped_llm = LangchainLLMWrapper(groq_llm)
    
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)   
    
    print("\nRunning RAGAS evaluation...")
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    results = evaluate(
        ragas_dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=wrapped_llm,
        embeddings=embeddings
    )
    return results

def load_test_set():
    path = os.path.join(BASE_DIR, "data/test_set.json")
    with open(path, "r") as f:
        return json.load(f)
    
def build_dataset(test_set):
    dataset = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    for i, item in enumerate(test_set):
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        print(f"Processing question {i+1}/{len(test_set)}: {question[:50]}")
        
        chunks = retrieve_chunks(question)
        answer = extract_answer(question, chunks)
        
        dataset["question"].append(question)
        dataset["answer"].append(answer)
        dataset["contexts"].append(chunks)
        dataset["ground_truth"].append(ground_truth)
    
    return dataset

if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions\n")
    
    print("Running questions through pipeline...")
    dataset = build_dataset(test_set)
    
    results = run_ragas(dataset)
    print("\nResults:")
    print(results)