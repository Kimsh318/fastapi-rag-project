# Faiss 관련
from sentence_transformers import SentenceTransformer
import faiss

class FaissClient:
    def __init__(self, index_file_path, metadata_file_path, model_path, search_k=30):
        self.embedding_model = self._load_embedding_model(model_path)
        self.index = self._load_index(index_file_path)
        if metadata_file_path:
            self.metadata = self._load_metadata(metadata_file_path)
        self.search_k = search_k  # Semantic Search 결과 top_k와는 별개. top_k < search_k

    def _load_embedding_model(self, model_path):
        # CPU 사용
        return SentenceTransformer(model_path)
            
    def _load_index(self, index_file_path):
        # CPU에서 인덱스 로드
        return faiss.read_index(index_file_path)

    def _load_metadata(self, metadata_file_path):
        with open(metadata_file_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return metadata

    def _encode_query(self, query):
        # CPU에서 쿼리 인코딩
        return self.embedding_model.encode(query).astype('float32').reshape(1, -1)

    def query(self, query_text):
        query_vector = self._encode_query(query_text)
        distances, indices = self.index.search(query_vector, self.search_k)
        return distances, indices



# FaissClient 객체를 생성하고 테스트하는 코드
if __name__ == "__main__":
    index_file_path = '/app/tmp_test/faiss_test/vector_index.faiss'
    metadata_file_path = None  # 메타데이터 파일 경로가 필요하면 여기에 추가하세요.
    model_path = "all-MiniLM-L6-v2"  # 모델 경로를 여기에 추가하세요.

    # FaissClient 객체 생성
    faiss_client = FaissClient(index_file_path, metadata_file_path, model_path)

    # 테스트 쿼리
    test_query = "여기에 테스트 쿼리를 입력하세요."
    distances, indices = faiss_client.query(test_query)

    # 결과 출력
    print("Distances:", distances)
    print("Indices:", indices)
