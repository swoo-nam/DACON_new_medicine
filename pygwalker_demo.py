import streamlit as st
import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import jaccard_score

DATA_PATH = "C:\python-code\project_final\data/"
SEED = 42
device = 'cuda' if torch.cuda.is_available() else 'cpu'

loaded_model = AutoModelForCausalLM.from_pretrained(f"{DATA_PATH}0908_gpt_freeze_all").to(device)
tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2",
                                          bos_token='</s>',
                                          eos_token='</s>',
                                          unk_token='<unk>',
                                          pad_token='<pad>',
                                          mask_token='<mask>',
                                          max_len=1024)
bert_model = SentenceTransformer('jhgan/ko-sroberta-multitask')

def evaluate_similarity(input_text, generated_text, alpha=0.9): # cosine 가중치 조절
    input_embedding = bert_model.encode(input_text)
    generated_embedding = bert_model.encode(generated_text)

    # 코사인 유사도
    cosine_sim = 1 - pairwise_distances([input_embedding], [generated_embedding], metric='cosine')[0][0]

    # 자카드 유사도
    input_tokens = set(input_text.split())
    generated_tokens = set(generated_text.split())
    jaccard_sim = len(input_tokens.intersection(generated_tokens)) / len(input_tokens.union(generated_tokens))
    # jaccard_sim = jaccard_score(input_tokens, generated_tokens)

    # 가중 평균 내보기
    weighted_sim = alpha*cosine_sim + (1 - alpha)*jaccard_sim

    return weighted_sim

# Streamlit 페이지의 너비 조정
st.set_page_config(
    page_title="Chatbot with Evaluation",
    layout="wide"
)

# 제목 추가
st.title("Chatbot with Evaluation")

# 사용자 입력받기
user_input = st.text_input("Input: ")

# 추론 및 평가 실행 버튼
if st.button("Generate Response"):
    if user_input:
        st.write("User Input:", user_input)
        text = "<q>" + user_input + "</s><a>"
        x = tokenizer.encode(text, return_tensors="pt").to(device)
        q_len = len(text) + 1

        best_generated_text = None
        best_similarity_score = -1.0
        generated_texts = []  # 답변 후보군을 담을 리스트
        for i in range(10):  # num_samples=10으로 수정
            result_ids = loaded_model.generate(x,
                                        max_length=128,  # 수정: max_len 대신 직접 지정
                                        repetition_penalty=2.0,
                                        num_beams=2,
                                        num_return_sequences=1,
                                        no_repeat_ngram_size=4,
                                        use_cache=True,
                                        do_sample=True,
                                        temperature=0.8,
                                        top_k=50)
            generated_text = tokenizer.decode(result_ids[0])
            generated_text = generated_text[q_len:-4]
            generated_text = re.sub(r'[^가-힣]', ' ', generated_text)

            similarity_score = evaluate_similarity(text, generated_text)
            generated_texts.append((similarity_score, generated_text))

            if similarity_score > best_similarity_score:
                best_similarity_score = similarity_score
                best_generated_text = generated_text

        st.write("Best Reply:", best_generated_text)
        st.write("Best Similarity Score:", best_similarity_score)