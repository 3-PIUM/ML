import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from difflib import SequenceMatcher
import logging
import os
import re

#  COOS 리스트 불러오기 (정답)

def load_coos_from_csv(file_path):
    df = pd.read_csv(file_path)
    coos_list = df['korean_name'].dropna().astype(str).tolist()
    return coos_list

#  유사도 함수
def combined_similarity(input_text, target_text, emb_sim):
    edit_sim = SequenceMatcher(None, input_text, target_text).ratio()
    return 0.6 * emb_sim + 0.4 * edit_sim

#  매핑 함수
def map_to_best_match(user_word):
    input_embedding = embedding_model.encode(user_word, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(input_embedding, coos_embeddings)[0]

    best_score = -1
    best_match = None

    for idx, coos_word in enumerate(coos_list):
        score = combined_similarity(user_word, coos_word, cosine_scores[idx].item())
        if score > best_score:
            best_score = score
            best_match = coos_word

    return best_match if best_score >= 0.75 else f"{user_word}(매칭 실패)"


# 개별 성분 매핑 후 다시 합치기
def correct_ingredient_list(ingredient_str):
    # 1. 문자열을 ',' 기준으로 나누고 양쪽 공백 제거
    ingredients = [i.strip() for i in ingredient_str.split(',') if i.strip()]
    
    # 2. 괄호 안의 '숫자 + ppm' 패턴 제거
    cleaned_ingredients = [
        re.sub(r'\(\s*\d+(?:\.\d+)?\s*(?:ppm|%)\s*\)', '', ing).strip() for ing in ingredients
    ]
    
    # 3. 교정 함수 적용
    corrected = [map_to_best_match(ing) for ing in cleaned_ingredients]
    
    return ', '.join(corrected)

#COOS 리스트 임베딩

coos_list = None
embedding_model = None
coos_embeddings = None



if __name__ == "__main__":
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 경로 설정
    coos_path = "D:/PIUM/성분/COOS_성분_통합.csv"
    input_csv_path = "D:/PIUM/추천/data/마스크팩/[최종]마스크팩_바코드_상품_전처리.csv"
    output_csv_path = "D:/PIUM/성분/상품_성분모델/마스크팩_상품_COOS성분명처리.csv"
    embedding_file = 'D:/PIUM/성분/상품_성분모델/coos_embeddings.pt'

    # Step 1: COOS 성분 리스트 로딩
    logging.info("Step 1: COOS 리스트 로딩 시작")
    coos_list = load_coos_from_csv(coos_path)
    logging.info("Step 1 완료")

    # Step 2: 모델 로딩
    logging.info("Step 2: 모델 로딩 중")
    embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    logging.info("Step 2 완료")

    # Step 3: 임베딩 로딩 또는 계산
    if os.path.exists(embedding_file):
        logging.info("Step 3: 저장된 임베딩 파일 불러오는 중")
        coos_embeddings = torch.load(embedding_file)
        logging.info("Step 3 완료 - 임베딩 불러오기 완료")
    else:
        logging.info("Step 3: COOS 임베딩 계산 중")
        coos_embeddings = embedding_model.encode(coos_list, convert_to_tensor=True)
        torch.save(coos_embeddings, embedding_file)
        logging.info("Step 3 완료 - 임베딩 저장 완료")

    # Step 4: 입력 CSV 로딩 및 성분 교정
    df = pd.read_csv(input_csv_path)
    ingredient_col = '성분'

    logging.info("Step 4: 성분 교정 시작")
    df['교정된_성분'] = df[ingredient_col].astype(str).apply(correct_ingredient_list)
    logging.info("Step 4 완료 - 교정된 성분 처리 완료")

    # Step 5: 결과 저장
    df.to_csv(output_csv_path, index=False)
    print(f"✔️ 교정된 CSV 저장 완료: {output_csv_path}",index=False, encoding='utf-8-sig')