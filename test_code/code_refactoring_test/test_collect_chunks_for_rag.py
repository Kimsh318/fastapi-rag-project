import unittest
from transformers import AutoTokenizer

MAX_INPUT_LEN = 200

#==========================원본 코드==============================
# LLM 전달용 Context Collect 함수
def ori_collect_chunks_for_rag_editted(outputs: list, tokenizer: object, prompt_without_data: str) -> list: 

    temp_chunk_texts = ""
    sum_of_len_input_ids = 0
    merged_chunk_texts = ""
    used_doc_ids_and_scores = {}
    # num_results = 0

    if not outputs:
        # return ["@No data@", num_results, used_doc_ids_and_scores]
        return ["@No data@", used_doc_ids_and_scores]
    
    input_ids_of_prompt_template = tokenizer(prompt_without_data, return_tensors = 'pt')
    #len_prompt_template = input_ids_of_prompt_template['input_ids'].size()[1]
    len_prompt_template = len(input_ids_of_prompt_template['input_ids'])
    MAX_PROMPT_LEN = MAX_INPUT_LEN - len_prompt_template - 16
    
    valid_chunk_ids_texts_lengths_scores = {}
    for i in range(len(outputs)): 
        input_ids_of_chunk_text = tokenizer(outputs[i]["chunk_context"], return_tensors = 'pt')
        len_input_ids = len(input_ids_of_chunk_text['input_ids'])
        
        #outputs의 길이가 기준 길이를 초과할 경우, child_chunk를 찾아서 Swap 처리한다.
        if len_input_ids >= MAX_PROMPT_LEN:     
            child_chunk_info = search_child_document_using_parent_id(outputs[i]["chunk_id"]) #먼저 Child에 해당하는 Chunk의 정보를 추출
            child_chunk_text = child_chunk_info['doc_type']+ '-' + child_chunk_info['chunk_src'] + '\n'+ child_chunk_info['chunk_context']
            input_ids_of_child_chunk_text = tokenizer(child_chunk_text, return_tensors = 'pt')
            
            len_input_ids = len(input_ids_of_child_chunk_text['input_ids'])
            
            outputs[i]["parent_chunk_id"] = outputs[i]["chunk_id"] # child_chunk 기준의 parent_chunk_id에 기존 outputs[i][0]을 할당 
            outputs[i]["chunk_id"] = child_chunk_info['chunk_id'] #탐색된 Child의 chunk_id로 변경
            outputs[i]["chunk_context"] = child_chunk_text #탐색된 Child의 chunk_context로 기존 chunk_context 변경
            # outputs[i]["score"] 즉, score는 유지

        if (len_input_ids < MAX_PROMPT_LEN) and (outputs[i]["chunk_id"] not in valid_chunk_ids_texts_lengths_scores.keys()):
            valid_chunk_ids_texts_lengths_scores[outputs[i]["chunk_id"]] = [outputs[i]["chunk_context"], len_input_ids, outputs[i]["score"]]
    
    chunk_texts_and_lengths_scores = list(valid_chunk_ids_texts_lengths_scores.values())
    chunk_ids_for_texts = list(valid_chunk_ids_texts_lengths_scores.keys())
    
    if not chunk_ids_for_texts:
        # return ["@No data@", num_results, used_doc_ids_and_scores]
        return ["@No data@", used_doc_ids_and_scores]
    
    num_results = 0
    for i in range(len(chunk_texts_and_lengths_scores)): #MAX_PROMPT_LEN에 도달할 때까지, Chunk를 채워넣음.
        sum_of_len_input_ids = sum_of_len_input_ids + chunk_texts_and_lengths_scores[i][1]     
        if sum_of_len_input_ids > MAX_PROMPT_LEN:
            merged_chunk_texts = temp_chunk_texts
            break
        
        num_results += 1
        temp_chunk_texts = temp_chunk_texts + f"\n참고문서[{num_results}]\n{chunk_texts_and_lengths_scores[i][0]}"
        if chunk_ids_for_texts[i] not in used_doc_ids_and_scores.keys():
            used_doc_ids_and_scores[chunk_ids_for_texts[i]] = chunk_texts_and_lengths_scores[i][2] #key: doc_id, val: score

        if i == len(chunk_texts_and_lengths_scores) - 1:
            merged_chunk_texts = temp_chunk_texts
        
    # return [merged_chunk_texts, num_results, used_doc_ids_and_scores]
    return [merged_chunk_texts, used_doc_ids_and_scores]

#====================분할된 코드=================================
import torch

# 전역 변수 설정 (실제 환경에 맞게 조정 필요)
MAX_INPUT_LEN = 2048
LLM_CUDA = 'cuda' if torch.cuda.is_available() else 'cpu'

def new_collect_chunks_for_rag_editted(outputs: list, tokenizer: object, prompt_without_data: str) -> list:
    """
    메인 함수: LLM 전달용 Context Collect
    
    :param outputs: 검색 엔진에서 반환된 관련 문서 청크 리스트
    :param tokenizer: 텍스트를 토큰화하는 데 사용되는 토크나이저 객체
    :param prompt_without_data: 데이터가 포함되지 않은 기본 프롬프트 템플릿
    :return: 병합된 청크 텍스트와 사용된 문서 ID 및 점수를 포함하는 리스트
    """
    if not outputs:
        return handle_no_data()
    
    MAX_PROMPT_LEN = calculate_max_prompt_length(tokenizer, prompt_without_data)
    
    valid_chunk_ids_texts_lengths_scores = process_chunks(outputs, tokenizer, MAX_PROMPT_LEN)
    
    chunk_texts_and_lengths_scores = list(valid_chunk_ids_texts_lengths_scores.values())
    chunk_ids_for_texts = list(valid_chunk_ids_texts_lengths_scores.keys())
    
    if not chunk_ids_for_texts:
        return handle_no_data()
    
    merged_chunk_texts, used_doc_ids_and_scores = merge_chunks(chunk_texts_and_lengths_scores, chunk_ids_for_texts, MAX_PROMPT_LEN)
    
    return [merged_chunk_texts, used_doc_ids_and_scores]

def handle_no_data():
    """
    데이터가 없는 경우 처리
    
    :return: 데이터 없음을 나타내는 메시지와 빈 딕셔너리
    """
    return ["@No data@", {}]

def calculate_max_prompt_length(tokenizer, prompt_without_data):
    """
    최대 프롬프트 길이 계산
    
    :param tokenizer: 텍스트를 토큰화하는 데 사용되는 토크나이저 객체
    :param prompt_without_data: 데이터가 포함되지 않은 기본 프롬프트 템플릿
    :return: 계산된 최대 프롬프트 길이
    """
    # TODO : 반입 후, 아래 주석 원상복귀 필요
    # input_ids_of_prompt_template = tokenizer(prompt_without_data, return_tensors='pt').to(LLM_CUDA)
    input_ids_of_prompt_template = tokenizer(prompt_without_data, return_tensors='pt')
    # TODO : 반입 후, 아래 주석 원상복귀 필요
    # len_prompt_template = input_ids_of_prompt_template['input_ids'].size()[1]
    len_prompt_template = len(input_ids_of_prompt_template['input_ids'])
    return MAX_INPUT_LEN - len_prompt_template - 16  # 16은 안전 마진

def process_chunks(outputs, tokenizer, MAX_PROMPT_LEN):
    """
    각 청크 처리 및 유효한 청크 정보 저장
    
    :param outputs: 검색 엔진에서 반환된 관련 문서 청크 리스트
    :param tokenizer: 텍스트를 토큰화하는 데 사용되는 토크나이저 객체
    :param MAX_PROMPT_LEN: 계산된 최대 프롬프트 길이
    :return: 유효한 청크 ID와 해당 정보(텍스트, 길이, 점수)를 포함하는 딕셔너리
    """
    valid_chunk_ids_texts_lengths_scores = {}
    for output in outputs:
        chunk_text, chunk_id, chunk_score = process_single_chunk(output, tokenizer, MAX_PROMPT_LEN)
        if chunk_text and chunk_id not in valid_chunk_ids_texts_lengths_scores:
            # TODO : 반입 후, 아래 주석 원상복귀 필요
            # chunk_length = len(tokenizer(chunk_text, return_tensors='pt').to(LLM_CUDA)['input_ids'][0])
            chunk_length = len(tokenizer(chunk_text, return_tensors='pt')['input_ids'][0])
            if chunk_length < MAX_PROMPT_LEN:
                valid_chunk_ids_texts_lengths_scores[chunk_id] = [chunk_text, chunk_length, chunk_score]
    return valid_chunk_ids_texts_lengths_scores

def process_single_chunk(output, tokenizer, MAX_PROMPT_LEN):
    """
    단일 청크 처리
    
    :param output: 단일 청크 정보를 포함하는 딕셔너리
    :param tokenizer: 텍스트를 토큰화하는 데 사용되는 토크나이저 객체
    :param MAX_PROMPT_LEN: 계산된 최대 프롬프트 길이
    :return: 처리된 청크 텍스트, 청크 ID, 청크 점수
    """
    chunk_text = output["chunk_context"]
    chunk_id = output["chunk_id"]
    chunk_score = output["score"]
    
    # TODO : 반입 후, 아래 주석 원상복귀 필요
    # input_ids_of_chunk_text = tokenizer(chunk_text, return_tensors='pt').to(LLM_CUDA)
    input_ids_of_chunk_text = tokenizer(chunk_text, return_tensors='pt')
    # TODO : 반입 후, 아래 주석 원상복귀 필요
    # len_input_ids = input_ids_of_chunk_text['input_ids'].size()[1]
    len_input_ids = len(input_ids_of_chunk_text['input_ids'])
    
    # 청크 길이가 최대 길이를 초과하는 경우 하위 청크로 대체
    if len_input_ids >= MAX_PROMPT_LEN:
        child_chunk_info = search_child_document_using_parent_id(chunk_id)
        chunk_text = f"{child_chunk_info['doc_type']}-{child_chunk_info['chunk_src']}\n{child_chunk_info['chunk_context']}"
        chunk_id = child_chunk_info['chunk_id']
    
    return chunk_text, chunk_id, chunk_score

def search_child_document_using_parent_id(parent_id):
    """
    부모 문서 ID를 사용하여 자식 문서를 검색합니다.
    이 함수의 실제 구현은 사용 중인 데이터베이스나 검색 시스템에 따라 달라질 수 있습니다.
    
    :param parent_id: 부모 문서의 ID
    :return: 자식 문서 정보를 포함하는 딕셔너리
    """
    # 이 부분은 실제 구현에 맞게 수정해야 합니다.
    # 예시 구현:
    return {
        'chunk_id': f'child_{parent_id}',
        'doc_type': 'child_document',
        'chunk_src': 'child_source',
        'chunk_context': f'This is a child document of {parent_id}'
    }

def merge_chunks(chunk_texts_and_lengths_scores, chunk_ids_for_texts, MAX_PROMPT_LEN):
    """
    청크 텍스트 병합
    
    :param chunk_texts_and_lengths_scores: 청크 텍스트, 길이, 점수를 포함하는 리스트
    :param chunk_ids_for_texts: 청크 ID 리스트
    :param MAX_PROMPT_LEN: 계산된 최대 프롬프트 길이
    :return: 병합된 청크 텍스트와 사용된 문서 ID 및 점수를 포함하는 튜플
    """
    merged_chunk_texts = ""
    used_doc_ids_and_scores = {}
    sum_of_len_input_ids = 0
    num_results = 0
    
    for i, (chunk_text, chunk_length, chunk_score) in enumerate(chunk_texts_and_lengths_scores):
        if sum_of_len_input_ids + chunk_length > MAX_PROMPT_LEN:
            break
        
        num_results += 1
        merged_chunk_texts += f"\n참고문서[{num_results}]\n{chunk_text}"
        sum_of_len_input_ids += chunk_length
        
        chunk_id = chunk_ids_for_texts[i]
        if chunk_id not in used_doc_ids_and_scores:
            used_doc_ids_and_scores[chunk_id] = chunk_score
    
    return merged_chunk_texts, used_doc_ids_and_scores


#====================테스트 코드=============================
# 테스트를 위한 가짜 토크나이저 클래스
class FakeTokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size

    def __call__(self, text, return_tensors='pt'):
        return {'input_ids': [list(range(len(text.split()))), ]}

    def to(self, device):
        return self

# 테스트 클래스
class TestCollectChunksForRAG(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.prompt_without_data = "이것은 기본 프롬프트입니다."

    def test_functions_equality(self):
        test_cases = [
            # 테스트 케이스 1: 일반적인 경우
            {
                "outputs": [
                    {"chunk_id": "1", "chunk_context": "이것은 첫 번째 청크입니다.", "score": 0.9},
                    {"chunk_id": "2", "chunk_context": "이것은 두 번째 청크입니다.", "score": 0.8},
                ],
            },
            # 테스트 케이스 2: 빈 출력
            {
                "outputs": [],
            },
            # 테스트 케이스 3: 긴 청크가 포함된 경우
            {
                "outputs": [
                    {"chunk_id": "3", "chunk_context": "이것은 매우 긴 청크입니다. " * 50, "score": 0.95},
                    {"chunk_id": "4", "chunk_context": "이것은 짧은 청크입니다.", "score": 0.7},
                ],
            },
            # 테스트 케이스 4: 여러 청크가 포함된 경우
            {
                "outputs": [
                    {"chunk_id": "5", "chunk_context": "첫 번째 청크", "score": 0.9},
                    {"chunk_id": "6", "chunk_context": "두 번째 청크", "score": 0.8},
                    {"chunk_id": "7", "chunk_context": "세 번째 청크", "score": 0.7},
                    {"chunk_id": "8", "chunk_context": "네 번째 청크", "score": 0.6},
                    {"chunk_id": "9", "chunk_context": "다섯 번째 청크", "score": 0.5},
                ],
            },
            # 테스트 케이스 5: 중복된 청크 ID가 포함된 경우
            {
                "outputs": [
                    {"chunk_id": "10", "chunk_context": "중복된 ID의 청크", "score": 0.9},
                    {"chunk_id": "10", "chunk_context": "또 다른 중복된 ID의 청크", "score": 0.8},
                    {"chunk_id": "11", "chunk_context": "유일한 ID의 청크", "score": 0.7},
                ],
            },
        ]

        for i, test_case in enumerate(test_cases, 1):
            with self.subTest(f"Test case {i}"):
                original_result = ori_collect_chunks_for_rag_editted(test_case["outputs"], self.tokenizer, self.prompt_without_data)
                split_result = new_collect_chunks_for_rag_editted(test_case["outputs"], self.tokenizer, self.prompt_without_data)
                self.assertEqual(original_result, split_result)

if __name__ == '__main__':
    unittest.main()