import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def split_text(text, chunk_size=100):
    """
    주어진 텍스트를 단어 단위로 분할하여 지정된 크기의 청크로 나눕니다.

    :param text: 전체 텍스트 문자열
    :param chunk_size: 각 청크의 단어 수
    :return: 분할된 텍스트 청크의 리스트
    """
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def create_faiss_index(embeddings):
    """
    주어진 임베딩을 사용하여 Faiss 인덱스를 생성합니다.

    :param embeddings: 텍스트 임베딩의 numpy 배열
    :return: 생성된 Faiss 인덱스
    """
    d = embeddings.shape[1]  # 벡터의 차원
    index = faiss.IndexFlatL2(d)  # L2 거리 기반의 플랫 인덱스 생성
    index.add(embeddings)  # 벡터를 인덱스에 추가
    return index

def save_faiss_index(index, file_name='vector_index.faiss'):
    """
    Faiss 인덱스를 파일로 저장합니다.

    :param index: 저장할 Faiss 인덱스
    :param file_name: 저장할 파일 이름
    """
    faiss.write_index(index, file_name)

def load_faiss_index(file_name='vector_index.faiss'):
    """
    파일에서 Faiss 인덱스를 로드합니다.

    :param file_name: 로드할 파일 이름
    :return: 로드된 Faiss 인덱스
    """
    return faiss.read_index(file_name)

def search_similar_texts(index, query_vector, top_k=5):
    """
    주어진 쿼리 벡터와 유사한 텍스트를 검색합니다.

    :param index: 검색에 사용할 Faiss 인덱스
    :param query_vector: 검색할 쿼리의 벡터
    :param top_k: 상위 몇 개의 유사한 텍스트를 반환할지 설정
    :return: 유사한 텍스트의 인덱스와 거리
    """
    distances, indices = index.search(query_vector, top_k)
    return distances, indices

if __name__ == '__main__':
    # 1. 데이터 준비
    file_path = './sample_data/banking_law.txt'  # 실제 텍스트 파일의 경로로 변경하세요.

    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    chunks = split_text(text, chunk_size=50)    # 임의 청크로 split

    # 2. 모델 및 인덱스 생성
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, convert_to_numpy=True)

    index = create_faiss_index(embeddings)
    print(f"인덱스에 추가된 벡터 수: {index.ntotal}")

    # 3. 인덱싱
    save_faiss_index(index, 'vector_index.faiss')
    loaded_index = load_faiss_index('vector_index.faiss')

    # 4. 검색
    query_text = "자기자본"  # 실제 검색할 텍스트로 변경하세요.
    query_vector = model.encode([query_text], convert_to_numpy=True)

    distances, indices = search_similar_texts(loaded_index, query_vector, top_k=5)

    for i, idx in enumerate(indices[0]):
        print(f"순위 {i+1}:")
        print(f"텍스트: {chunks[idx]}")
        print(f"거리: {distances[0][i]}")
        print()
