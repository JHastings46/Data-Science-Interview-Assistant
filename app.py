import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset, concatenate_datasets
import torch

st.set_page_config(page_title="Data Science Interview Assistant", layout="wide")
st.title("Data Science Interview Assistant")
st.write("Ask any data science interview question and get an AI-powered answer.")

@st.cache_resource
def load_models():
    # Load retriever
    retriever = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Load PEFT model from HuggingFace
    base = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-base",
        torch_dtype=torch.float32
    )
    model = PeftModel.from_pretrained(
        base,
        "joelhastings-ds/ds-interview-assistant"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "joelhastings-ds/ds-interview-assistant"
    )
    return retriever, model, tokenizer

@st.cache_data
def load_qa_pairs():
    ds1 = load_dataset("mjphayes/machine_learning_questions")
    ds2 = load_dataset("UdayG01/DataScienceInterviewQuestions")
    ds2 = ds2.rename_columns({"Question": "question", "Answer": "answer"})
    
    combined = concatenate_datasets([
        ds1["train"].select_columns(["question", "answer"]),
        ds1["test"].select_columns(["question", "answer"]),
        ds2["train"].select_columns(["question", "answer"]),
    ])
    return combined

@st.cache_data
def encode_questions(_retriever, _dataset):
    questions = _dataset["question"]
    embeddings = _retriever.encode(questions, convert_to_tensor=True)
    return embeddings

# Load everything
with st.spinner("Loading models..."):
    retriever, model, tokenizer = load_models()
    dataset = load_qa_pairs()
    question_embeddings = encode_questions(retriever, dataset)

# User input
question = st.text_input("Enter your data science interview question:")

if question:
    with st.spinner("Generating answer..."):
        
        # Step 1: Retrieve top 3 similar Q&A pairs
        query_embedding = retriever.encode(question, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, question_embeddings)[0]
        top_indices = scores.topk(3).indices.tolist()
        
        retrieved = []
        for idx in top_indices:
            retrieved.append({
                "question": dataset[idx]["question"],
                "answer":   dataset[idx]["answer"],
                "score":    scores[idx].item()
            })
        
        # Step 2: Build RAG prompt
        context = ""
        for i, r in enumerate(retrieved):
            context += f"Q{i+1}: {r['question']}\nA{i+1}: {r['answer']}\n\n"
        
        prompt = f"""Use the following Q&A pairs as context to answer the question clearly and concisely.

{context}Now answer this question:
{question}

Answer:"""
        
        # Step 3: Generate answer
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=GenerationConfig(
                    max_new_tokens=200,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            )
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Display answer
        st.subheader("Answer")
        st.write(answer)
        
        # Display retrieved context
        st.subheader("Retrieved Context Used")
        for i, r in enumerate(retrieved):
            with st.expander(f"Source {i+1} — similarity: {r['score']:.2f}"):
                st.write(f"**Question:** {r['question']}")
                st.write(f"**Answer:** {r['answer']}")
