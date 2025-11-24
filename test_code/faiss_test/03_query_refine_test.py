# Faiss 관련
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import re
import torch
import time
#==================================FAISS Config on Development Environment============================================
# fastapi app config를 그대로 가져옴
AI_MODEL_PATH: str = '/workspace/ai_model/'
EMBEDDING_MODEL_NM: str = 'bge-m3'
EMBEDDING_MODEL:str = AI_MODEL_PATH+EMBEDDING_MODEL_NM

REFINE_FAISS_FAISS_INDEX_FILE_PATH: str = "/workspace/dev/dev1_p/13_langchain_fastapi_backend_dev/v17/fastapi_backend/app/db_faiss/faiss_useless_terms.index"
REFINE_FAISS_METADATA_FILE_PATH: str = "/workspace/dev/dev1_p/13_langchain_fastapi_backend_dev/v17/fastapi_backend/app/db_faiss/faiss_useless_terms_meta.json"
REFINE_EMBEDDING_MODEL_PATH: str = EMBEDDING_MODEL
REFINE_FAISS_MODEL_DEVICE_ID: str = 'cuda:1'
REFINE_FAISS_INDEX_DEVICE_ID: str = 'cuda:1'
REFINE_QUERY_REFINEMENT_THRESHOLD:float = 0.8

#==============================================================================

class FaissClient:
    def __init__(self, index_file_path, metadata_file_path, model_path, model_device_id, index_device_id, high_priority_threshold, search_k=30):
        self.model = self._load_embedding_model(model_path, model_device_id)
        self.index = self._load_index(index_file_path, index_device_id)
        if metadata_file_path:
            self.metadata = self._load_metadata(metadata_file_path)
        self.search_k = search_k  # Semantic Search 결과 top_k와는 별개. top_k < search_k
        self.high_priority_threshold = high_priority_threshold

    def _load_embedding_model(self, model_path, model_device_id):
        if model_device_id:
            # GPU 사용
            gpu_before_filter = float(torch.cuda.memory_allocated(model_device_id)/1024**2)
            print(f"Sentence Transformer is loading on GPU {model_device_id}")
            model = SentenceTransformer(model_path, device=model_device_id)
            gpu_after_filter = float(torch.cuda.memory_allocated(model_device_id)/1024**2)
            print(f'Sentence Transformer memory usage : {(gpu_after_filter - gpu_before_filter):.2f} MB')
            
            return model
        # CPU 사용
        return SentenceTransformer(model_path)
            
    def _load_index(self, index_file_path, index_device_id=None):
        cpu_index = faiss.read_index(index_file_path)
        print(f"[FAISS INDEX CONFIG]\nindex_file_path : {index_file_path}\nindex_device_id : {index_device_id}\n===============")
        if index_device_id:
            gpu_before_filter = float(torch.cuda.memory_allocated(index_device_id)/1024**2)
            
            print(f"FAISS Index is loading on GPU {index_device_id}")
            # GPU 사용
            res = faiss.StandardGpuResources() # Faiss GPU 자원 설정
            device_num = int(re.search(r'cuda[:\s]*(\d+)', index_device_id).group(1))
            print(f'index_device_id {index_device_id} -> {device_num}')
            gpu_index = faiss.index_cpu_to_gpu(res, device_num, cpu_index)
            
            gpu_after_filter = float(torch.cuda.memory_allocated(index_device_id)/1024**2)
            print(f'FAISS Index memory usage : {(gpu_after_filter - gpu_before_filter):.2f} MB')
            
            return gpu_index
        return cpu_index

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

    # ================Refinement====================================

    def _generate_ngrams(self, text, n=4):
        words = text.split()
        ngrams = [{"phrase": " ".join(words[i:i+j]), "start": i, "end": i+j}
                  for j in range(1, n+1) for i in range(len(words)-j+1)]
        return ngrams

    def filter_query(self, query):
        ngrams = self._generate_ngrams(query)
        
        # 기존 코드 : Numpy array로 변환
        embeddings = self.model.encode(ngrams, convert_to_tensor=False, show_progress_bar=False)
        embeddings = np.array(embeddings).astype('float32')

        # embeddings = self.model.encode(ngrams, convert_to_tensor=True, show_progress_bar=False)
        # embeddings = embeddings.numpy()
        # FAISS 검색: 각 n-gram에 대해 가장 가까운 메타데이터와 유사도를 계산
        
        gpu_before_filter = float(torch.cuda.memory_allocated(device_id)/1024**2)
        distances, indices = self.index.search(embeddings, 1)
        gpu_after_filter = float(torch.cuda.memory_allocated(device_id)/1024**2)
        print(f'\t index search memory usage : {(gpu_after_filter - gpu_before_filter):.2f} MB')
        
        remove_candidates = []
        
        for idx, (ngram, distance) in enumerate(zip(ngrams, distances[:, 0])):
            ngram_length = len(ngram['phrase'].split())
            if distance <= (1 - self.high_priority_threshold):
                remove_candidates.append({"word_count": len(ngram['phrase'].split()), "score":1-distance, "phrase" : ngram['phrase'], "start": ngram['start'], "end": ngram['end']})
        
        remove_candidates = remove_candidates[::-1]
        
        remove_intervals = []
        is_vaild = 0

        for item in remove_candidates:
            if not remove_intervals:
                remove_intervals.append({"phrase": item['phrase'], "start" : item['start'], "end" : item['end']})
            else:
                is_valid = 1
                for i in range(len(remove_intervals)):
                    if item['start'] >= remove_intervals[i]['end'] or item['end'] <= remove_intervals[i]['start']:
                        is_vaild = is_valid * 1
                    else:
                        is_valid = is_valid * 0
                if is_valid == 1:
                    remove_intervals.append({"phrase": item['phrase'], "start" : item['start'], "end" : item['end']})
        diff = 0
        query_tokens = query.split()

        remove_intervals = sorted(remove_intervals, key=lambda x:x["start"])

        for interval in remove_intervals:
            for i in range(interval['start'], interval['end']):
                query_tokens[i] = "[REMOVED]"

        return query_tokens

    # ==================================================================


q_list = [  '차입신청서에는 어떤 항목이 필요해',
            '대부업자에게 대출 할수 있어',
            '임대사업자 앞 대출 지원시 유의사항 알려줘',
            '신규 운영자금 대출시 유의사항 알려줘',
            '여신 취급시, 거래처의 리스 내역도 조사해야돼',
            '거래처에 며칠전까지 기한연장 승인여부를 알려줘야돼',
            '시설자금은 어떤 용도로 사용가능한지 요약해줘',
            'M&A 목적으로 시설자금을 사용가능한지 알려줘',
            '사무실 임차 목적일때, 시설자금으로 대출이 가능한가',
            '시설자금 용도별로 기술조사 특례 항목 요약해줘',
            '시설자금의 대출기간은 20년을 초과할수 있어',
            '시설자금 거치기간 산정 기준 알려줘',
            '운영자금은 어떤 용도로 사용가능한지 정리해줘',
            '이자상환도 운영자금으로 취급가능한지 알려줘',
            '기업분석이 뭐야',
            '신용조사서는 언제 작성해야돼 ',
            '소요자금 사정이란',
            '기술조사는 언제 왜하는지',
            '계획시설이 완성된 경우에는 어떤 조사를 하는지',
            '기술진단과 기술평가는 어떻게 다른지 비교해줘',
            '사전한도 승인 이후 개별 승인시 작성해야 하는 서류는',
            '준공에 따른 조건변경 시 승인신청서에서 생략할 수 있는 항목 알려줘',
            '약식심사에서 우량담보 제공여신의 종류를 알려줘',
            '신규거래처는 약식심사 적용 가능해',
            '여신유의업종 거래처는 약식심사 적용 가능해',
            '시스템심사가 뭐야',
            '시스템심사로 지출검토 할수 있어',
            '시스템심사 대상 알려줘',
            '시스템 심사 방법 알려줘',
            '중소기업 심사 시 주의사항이 뭐야',
            '신규거래처 승인시 유의사항이 뭐야',
            '완제거래처가 다시 거래하는 경우 신규거래처야',
            '여신유의산업 기업 승인시 유의사항이 뭐야',
            '관계회사 분석해야되는 대상은 뭐야',
            '관계회사의 정의가 뭐야',
            '현장조사를 반드시 해야되는 경우는 언제야',
            '심사시 현장조사를 생략할 수 있는 경우에는 어떤것들이 있어',
            '전결권 계산할때, 후취 담보가액 포함이야',
            '금리산정 시 후취담보 포함이야',
            '신용여신의 정의가 뭐야',
            '순신용금액은 어떻게 산출해',
            '순신용금액 산정 기준일자는 언제야',
            '보증서 담보 운영자금 대출인 경우 별도의 심사제도가 있어',
            '보증부 운영자금대출 심사 특례를 받을 수 있는 대상은',
            '보증부 운영자금대출 심사 절차는',
            '중소기업 소액여신 심사 적용 대상은',
            '중소기업 소액여신 심사 시 신용조사서 작성해야돼',
            '시설자금 대출도 중소기업 소액여신 심사 특례 적용할 수 있어',
            '운영자금 한도에 포함되지 않는 운영자금 종류를 알려줘',
            '전자외담대도 운영자금 한도에 포함돼',
            '운영자금 대출한도를 산출하는 방법을 알려줘',
            '재해 발생한 경우 운영자금 한도이상으로 지원할 수 있어',
            '운영자금 기한연장시에도 한도검토를 해야돼',
            '제2금융권 운영자금 상환목적으로 운영자금 대출을 검토할때, 운영자금 대출한도 검토를 제외할 수 있어',
            '운영자금 대출한도 계산 방법을 바꿀 수 있어',
            '스타트업은 운영자금 한도 계산 시 특례가 있어',
            '운영자금 용도외 유용이 뭐야',
            '용도외유용 점검 시 증빙서류 예시 알려줘',
            '용도외유용 사후관리 예외대상 알려줘',
            '용도외유용 면책기준 알려줘',
            '용도외유용 제재조치 대상업체와 계속 거래하려면 누구의 승인을 얻어야해',
            '운영자금 자동연장 옵션 알려줘',
            '다모아 기준금리 뭐야',
            '다모아플러스대출 기여금액 어떻게 계산해',
            '단기운영자금에는 어떤 종류가 있는지 알려줘',
            '단기한도대출은 용도외유용 점검 해야돼',
            '단기한도대출의 기한전상환수수료율 얼마야',
            '단기한도대출 기준금리 뭐야',
            '단기한도대출 계정과목뭐야',
            '단기한도대출 어떻게 기표할 수 있어',
            '단기한도대출을 다모아로 바꿀수 있어',
            '단기한도대출이 뭐야',
            '단기한도대출과 다모아플러스대출을 비교해줘',
            '제작금융대출이 뭐야',
            '제작금융대출의 기준금리는',
            '일시당좌대월은 기한연장할수 있어',
            '당좌대월의 최대 대출기간은',
            '당좌대월 이자 연체시 어떻게 처리해야해',
            '어음할인 대상 어음은',
            '전자어음 상환방법은',
            '전자외담대가 뭐야',
            '전자외담대 최대 대출 금액은',
            '전자외담대 이자 수취방법은',
            '전자외담대 업무 절차를 알려줘',
            '전자외담대에서 판매업체의 외상매출채권 담보 취등방식을 비교해줘',
            '계열RM이 있는 계열은 어떻게 선정돼',
            '계열RM은 어떻게 선정돼',
            '계열 Credit Line 적용 대상은',
            '계열 Credit Line 한도 초과되면 어떻게 돼',
            '계열 Credit Line 승인여부는 누가 결정해',
            '사전한도제 적용 대상은 뭐야',
            '사전한도제 승인 후 개별 승인은 어떻게 해야돼',
            '사전한도제 범위 내에서 승인할때, 전결권은 뭐야',
            '사전한도제에서 개별 승인 시 작성하는 서류는 뭐야',
            'R&D투자 및 보완투자용 시설자금을 사전한도제로 승인하려면 어떻게 해야돼',
            '사전한도제 적용 R&D투자 및 보완투자용 시설자금 약정체결 특별조건 알려줘',
            '사전한도제 한도 조정 검토 대상 알려줘',
            '환율이 올라서 사전한도를 초과한 경우 어떻게 해야돼',
            '포괄여신승인제가 뭐야',
            '포괄여신승인제를 쓸수있는 기업은',
            '포괄여신승인제의 매년 여신조건은 누가 결정해',
            '동산담보는 어느 화면에서 등록해',
            '유형자산의 정의가 뭐야',
            '담보가 될수 있는 유형자산에는 어떤것들이 있어',
            '유형자산 담보대출의 한도는 얼마야',
            '재고자산 담보대출의 대출한도는 어떻게 계산해',
            '매출채권이 담보로 인정받으려면 어떤 조건을 갖추어야해',
            '동산 담보 여부를 은행끼리 공유하는 시스템이 뭐야',
            '동산 채권 담보 대출 실행 시 담보 취득 방법은',
            '동산채권담보 대출 부실 시 어떻게 회수해',
            '당행 예금 담보시 예금담보대출로만 취급해야돼',
            '예금담보대출 금리랑 수수료 알려줘',
            '타행 예금 담보시에도 예금담보대출로 취급해야돼',
            '외화대출 기표시 적용 환율 뭐야',
            '외화대출 원리금 회수 시 적용 환율 뭐야',
            '외화대출 실수요 증빙서류 예시 알려줘',
            '외화대출 시 위험고지 의무를 이행해야 하는 대상은',
            '해외 직투로 취급할 수 있는 경우는',
            '연장조건부 단기시설자금 대출 약정서에 특별조건 알려줘',
            '연장조건부 단기시설자금 대출 내입 후 기한연장 전결권 뭐야',
            '연장조건부 단기시설자금 대출 내입 후 기한연장 시 신속심사 적용할 수 있어',
            '연장조건부 단기시설자금 대출의 상환방식을 분할상환으로 선택할 수 있어',
            '신용보강한도대출이 뭐야',
            '실행금리가 15%를 초과할 수 있어',
            '실행금리는 어떻게 결정돼',
            'RM기준금리 조정항목에는 어떤것들이 있어',
            '유동성리스크프리미엄을 RM이 변경할 수 있어',
            '유동성리스크프리미엄 적용 기준 기간은 어떻게 계산해',
            '신보출연료율은 얼마야',
            '조정항목은 언제 바뀌어',
            '고정금리 일때, 기준금리 대출기간은 어떻게 계산해',
            '대출기간이 3개월 이하인 경우, 기준금리는 뭐야',
            'RM스프레드는 어떻게 구성되어있어',
            '영업마진을 0보다 작게 할수 있어',
            '조정금리가 뭐야',
            '업무원가를 어떻게 산출해',
            '업무원가 계산시, 한도여신의 경우 한도액기준이야 아니면 잔액 기준이야',
            '대출기간 중 기준금리 종류 바꿀수 있어',
            '정책금리를 중복으로 적용할 수 있어',
            'RM스프레드를 여신기간 중 변경할 수 있어',
            '금리인하요구권이 뭐야',
            '예금담보대출도 금리인하 요구권 적용 대상이야',
            '금리인하요구권 심사 주체는 누구야',
            '금리인하요구권 전결권은 누구야',
            '금리인하요구권은 어느 항목에 의해 조정돼',
            '금리인하요구권에 따른 금리인하 시, 영업마진도 조정할 수 있어',
            '금리인하요구를 받은 뒤 몇일 안에 결과를 통보해줘야돼',
            '금리인상 전결권은 어떻게 돼',
            '만기연장과 여신기간중 연장의 차이가 뭐야',
            '만기연장과 여신기간중 연장의 재책정된 금리 적용시점 차이를 비교해줘',
            '여신 조건변경의 종류에 대해 알려줘',
            '일반 운영자금에 대해 기간중 증액할 수 있어',
            '연체이율은 어떻게 계산돼',
            '채무자 사망시 연체이율은 어떻게 돼',
            '연체이율이 17%가 될 수 있어',
            '가지급금에 대한 연체시에는 연체이율이 얼마야',
            '금리가 어떻게 산출되었는지 차주앞에 제공하는 서류',
            '실행보증료율 상한이 얼마야',
            '보증료율은 어떻게 결정돼',
            '융자약정수수료 면제하려면 어떻게 해야돼',
            '융자약정수수료는 어떻게 계산해',
            '합좌 목적으로(승인시기 일치를 위한 대환) 기한전상환시 기한전수수료 면제할 수 있어',
            '특별약정을 이행하기 위해 상환하는 경우, 기한전상환수수료 면제 가능해',
            '운영자금 대환하는 경우에도 기한전상환수수료 징수해야돼',
            '지급보증서 발급 수수료는 얼마야',
            '예외적으로 제3자에게 연대보증 요구할 수 있는 경우 알려줘',
            '법인 연대입보 할때, 연대보증인이 2명일 수 있어',
            '연대보증 면제특약 보증서 담보 대출은 연대보증을 요구할 수 없어',
            '대표이사가 담보를 제공한경우, 대표이사를 연대입보할 수 있어',
            '연대보증 한도는 어떻게 계산해',
            '법인을 연대입보하려면 어떻게 해야돼',
            '연대입보시 한정근으로 처리할 수 있어',
            '기한연장할때 연대보증한도 조정해야돼',
            '연체 발생 시 연대보증인한테 통보 해야돼',
            '약정체결할때, 청산가치담보비율이 100%미만인 경우 특별조건 알려줘',
            '한도대출 약정서에서 채무자의 신용상태 악화 예시는',
            '약정체결 시기는',
            '약정체결시 징구해야하는 서류 알려줘',
            '권리관계서류의 금액 표시 수정할 수 있어',
            '확정일자를 받아야하는 서류는 뭐가 있어',
            '상환방법을 바꾸려면, 별도의 약정을 체결해야돼',
            '비상장주식을 담보로 취득할 수 있어',
            '원재료도 담보로 취득할 수 있어',
            '토지를 제외한 건물만을 담보로 취득할때 유의사항을 알려줘',
            '신보 보증서 담보 취득 시, 유의사항을 알려줘',
            '예금담보를 한정근으로 취득할 수 있어',
            '용도가 담보제공용이 아닌 감정평가서도 인정이 가능해',
            '제3자가 담보를 제공하는 경우, 어떤 서류를 받아야해',
            '피담보채무를 공란으로 둘수 있어',
            '기설정된 근저당권에 대해 피담보채무를 변경하려면 어떻게 해야돼',
            '대출이 완제된 경우 근저당권은 어떻게 처리해야돼',
            '근저당권의 채권최고액을 외화로 한경우 어떤 환율을 적용해',
            '근저당권의 설정 금액은 어떻게 산출해',
            '근저당권 설정 시 비용은 누가 부담해',
            '기계기구를 담보로 설정하는 방법을 알려줘',
            '담보 해지 요청 시 어떻게 처리해',
            '공장부지구입 시 대출 가능금액은',
            '공장부지구입 시 담보취득 방법은',
            '산업단지 내 분양토지의 채권 보전 방법은',
            '토지분양대금반환금 청구권을 선취담보로 인정받는법',
            '유가증권 담보 취득방법 알려줘',
            '당행 예금 담보 취득방법 알려줘',
            '후취담보 현실화율은',
            '담보물 재감정 가능 사유는',
            '항공기 담보가액 계산 방법을 정리해줘',
            '나대지에 미등기 건물이 있는 경우 지상권을 설정해야 하는지',
            '최우선변제임금채권은 어떻게 계산해',
            '상가건물을 담보로 취득할때 유의사항 알려줘',
            '주택담보 취급시 임차보증금 처리방법 알려줘',
            '보증서 기한 연장 시 당행 대출을 대환처리 할 수 있어',
            '후취담보 관리방법 알려줘',
            '양도담보 물건 부보 하는방법은',
            '보험 면제가능한 담보와 면제시 처리 절차는',
            '보험금을 채무자 앞 지급할 수 있는 경우는',
            '신보 보증서 담보 대출 취급시 사전 지출이 가능해',
            '보험가입 거절로 부보가 불가능한 경우 담보로 인정 가능한가',
            '부보기준액은 어떻게 계산해',
            '보험료 납부는 누가해',
            '준공의 기준은 무엇인가',
            '공사 지연으로 미인출되고 있을때, 어떻게 해야돼',
            '상환계획표 통지 방법 알려줘',
            '외국 설비 구입목적 지출 시 타행 해외송금으로도 대금 지급이 가능한가요',
            '기성고조사 없이 사전 지출 가능한 최대 한도는',
            '기성고 조사 후 시설자금 지출 시 지급 방법은',
            '외화대출 원리금 회수 시 원화로 상환이 가능한지',
            '이자 납입일이 휴일인 경우 언제 납입',
            '사후관리 업무 범위는',
            '동태점검 대상 거래처는',
            '동태점검 시 시스템에서 자동 점검되는 항목은',
            '동태점검 시 당연 론모니터링 대상은',
            '특별약정을 체결해야만 하는 경우는',
            '약정관리 대상은',
            '약정관리 대상업체가 약정을 이행하지 않을 경우 어떻게 해야하나요',
            '사후관리 시스템에서 담보관리 점검대상으로 추출되는 사유는',
            '담보 점검 결과 담보가치 하락이 예상되는 경우 어떻게 조치해야해',
            '론모니터링이 뭐야',
            '론모니터링 대상 기업은',
            '론모니터링 수행 절차는',
            '조기경보기업 선정 절차는',
            '론모니터링 등급별 관리 방법은',
            '조기경보기업으로 선정시 관리방안은',
            '부실 자료를 제출한 거래처의 경우 어떻게 관리해',
            '거래처가 분식회계를 했을때 어떻게 처리해야돼',
            '주채무계열이 뭐야',
            '주채권은행은 주채무계열에 대해 정기평가를 얼마나 자주 실시해야돼',
            '중점관리 계열은 어떻게 선정해',
            '신용유의정보는 어느 기관에서 관리해',
            '신용유의정보가 등록되어있으면 어떤 조치가 이루어져',
            '보증의 종류에는 어떤 것들이 있어',
            '지급보증과 채무보증의 차이는 뭐야',
            '당행의 최대 보증한도는 얼마야',
            '외화 표시 보증은 어떤 환율을 적용해서 처리해',
            '보증료는 선취야 후취야',
            '보증 대지급 청구기간은 언제까지야',
            '개인사업자는 일반금융소비자에 포함돼',
            '위법계약 해지권이 뭐야',
            '위법계약해지권을 사용한 경우, 관련 수수료를 부과할 수 있어',
            '일반금융소비자는 며칠 이내에 대출 계약을 철회할 수 있어',
            '적합성 원칙과 적정성 원칙의 차이가 뭐야',
            '포괄근 담보를 요구하는 경우 어떤 금소법 원칙에 위배돼',
            '구속성 행위의 정의가 뭐야',
            '수시 입출금 통장 개설도 구속성 행위에 포함돼',
            '여신 거래처의 관계회사에 예금 상품 가입을 권유하는 행위는 금지 대상이야',
            '각기 다른 차주에 대한 여신이지만 동일 회사에 대한 주식 담보가 합쳐져서 30%이면 신고 대상이야',
            '20%초과 주식을 담보로 잡고있는 여신이 기한연장 된 경우에도 보고해야돼']

# FaissClient 객체를 생성하고 테스트하는 코드
if __name__ == "__main__":
    device_id = REFINE_FAISS_MODEL_DEVICE_ID
    torch.cuda.set_device(device_id)
    
    index_file_path = REFINE_FAISS_FAISS_INDEX_FILE_PATH
    metadata_file_path = REFINE_FAISS_METADATA_FILE_PATH 
    model_path = EMBEDDING_MODEL  # 모델 경로를 여기에 추가하세요.

    # FaissClient 객체 생성
    print(f'Before FAISS Client : {torch.cuda.memory_allocated(device_id)/1024**2:.2f} MB')
    faiss_client = FaissClient(index_file_path=index_file_path, 
                               metadata_file_path=metadata_file_path, 
                               model_path=model_path,
                               model_device_id=device_id,
                               index_device_id=device_id,
                               high_priority_threshold=REFINE_QUERY_REFINEMENT_THRESHOLD)
    print(f'After FAISS Client : {torch.cuda.memory_allocated(device_id)/1024**2:.2f} MB')

    # 테스트 쿼리
    for itr, query in enumerate(q_list):
        gpu_before_filter = float(torch.cuda.memory_allocated(device_id)/1024**2)
        #print(f'Before filter_query : {torch.cuda.memory_allocated(device_id)/1024**2:.2f} MB')
        query_tokens = faiss_client.filter_query(query)
        #print(f'After filter_query : {torch.cuda.memory_allocated(device_id)/1024**2:.2f} MB')
        gpu_after_filter = float(torch.cuda.memory_allocated(device_id)/1024**2)
        print(f"[{itr}] {query}")
        print(f"\t tokens : {query_tokens}")
        print(f'\t memory usage : {(gpu_after_filter - gpu_before_filter):.2f} MB')
        
    # 결과 출력
    #print(query_tokens)

    while True:
        time.sleep(10)
        
