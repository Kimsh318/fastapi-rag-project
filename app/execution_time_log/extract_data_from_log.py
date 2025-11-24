import re
import os
from datetime import datetime, timedelta, timezone
import time
from collections import defaultdict
from string import Template
import traceback
import shutil
import threading

import matplotlib.pyplot as plt
from elasticsearch import Elasticsearch, helpers

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from app.core.config import settings


def search_documents_in_elasticsearch(doc_ids):
    # Elasticsearch에서 doc_id들을 기준으로 문서 검색
    documents = []

    for doc_id in doc_ids:
        try:
            # ES id 검색 설정
            body = {
                "query": {
                    "bool": {
                        "filter": {
                            "term": {
                                "_id": doc_id
                            }
                        }
                    }
                }
            }
            # 인덱스 이름과 doc_id를 사용하여 문서 검색
            response = es_client.search(index=settings.ES_INDEX_NAME, body=body)
            documents.append(response)
        except Exception as e:
            print(f"Error {doc_id}: {e}")
    return documents

def merge_chunks(list_docs):
    # 프롬프팅에 사용될 문서 텍스트들을 병합 - 반드시 fastapi 앱 구현내용과 같이 업데이트 되어야 함
    merged_chunk_texts = ""
    num_of_doc = 1
    for itr, output in enumerate(list_docs):
        chunk_text = f"""{output["doc_type"]}-{output["doc_src"]}\n{output["doc_text"]}"""
        chunk_seperator = f"\n#####\n참고문서[{num_of_doc}]\n#####\n" 
        merged_chunk_texts += f"{chunk_seperator+chunk_text}"
        num_of_doc += 1
    return merged_chunk_texts
    
def get_rag_prompt(query, merged_text):
    # 프롬프팅 템플릿에 문서 텍스트들 추가 - 반드시 fastapi 앱 구현내용과 같이 업데이트 되어야 함
    prompt_template_rag = """You are an AI chatbot that provides detailed ANSWER to a user's QUESTION. The ANSWER must be written using only the SEARCH_RESULTS provided to you. SEARCH_RESULTS typically consist of multiple documents, each separated by a delimiter "\n#####\n참고문서[NUMBER]\n#####\n". If SEARCH_RESULTS do not exist (e.g., SEARCH_RESULTS : @No data@), you should respond with '질문과 관련된 정보를 찾을 수 없습니다. 질문을 다시 작성해 보세요.' After providing the ANSWER, please provide the SOURCES used to write the ANSWER. The SOURCES should be only the 참고문서[NUMBER] and its DOCUMENT_TITLE (e.g., 참고문서[1] (내규/지침-여신지침(3.심사및승인))), not the content of the SEARCH_RESULTS. If multiple SOURCES have been used, please distinguish them through a separator of ','. Typically, DOCUMENT_TITLE exists the first or second line in each of SEARCH_RESULTS. All ANSWER must be in Korean.\n"""
    query_template_rag = Template("QUESTION : $query\n")
    search_result_template_rag = Template("SEARCH RESULTS : $merged_chunk_texts\n")
    answer_template_rag = "\nContinue to answer the QUESTION by using ONLY the SEARCH_RESULTS.\nANSWER : "
    template = Template(prompt_template_rag + query_template_rag.template + search_result_template_rag.template + answer_template_rag)

    prompt = template.safe_substitute(query=query, merged_chunk_texts=merged_text)
    return prompt

# 1. 로그 파일에서 데이터 추출
def extract_log_data(log_file):
    elasped_time_pattern = r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .+ - INFO - \[(?P<user_id>.+?)\] \[(?P<request_id>.+?)\] (?P<function_name>[\w.]+) completed in (?P<elapsed_time>\d+\.\d+)s"

    request_data_pattern =  r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .+ - INFO - \[(?P<request_id>.+?)\] (?P<function_name>[\w.]+) request data : (?P<data_string>.+)"
    
    response_data_pattern =  r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .+ - INFO - \[(?P<request_id>.+?)\] (?P<function_name>[\w.]+) response data : (?P<data_string>.+)"
    
    elasped_time_data, request_data, response_data = [], [], []

    with open(log_file, "r") as file:
        lines = file.readlines()
    dst = log_file.split("/")
    # dst.insert(-1,"log_backup")
    # dst[-1] = str(time.time())+dst[-1]
    # dst = '/'.join(dst)

    backup_time = time.time()
    backup_dtime = datetime.fromtimestamp(backup_time)
    new_fname = backup_dtime.strftime("%y%m%d-%H:%M:%S")+'_'+dst[-1]
    dst = os.path.join(backup_file, new_fname)
    
    #shutil.move(log_file, dst)
    shutil.copy(log_file, dst)
    
    print(f"{time.time()} log file read : {len(lines)} lines")
    for line in lines:
        match = re.search(elasped_time_pattern, line)
        if match:
            # print(f"({match} : {line})")
            timestamp = datetime.strptime(match.group("timestamp"), "%Y-%m-%d %H:%M:%S,%f")
            function_name = match.group("function_name")
            request_id = match.group("request_id")
            elapsed_time = float(match.group("elapsed_time"))
            user_id = match.group("user_id")
            elasped_time_data.append([timestamp, request_id, function_name, elapsed_time, user_id])
            

        match = re.search(request_data_pattern, line)
        if match:
            # print(f"({match} : {line})")
            timestamp = datetime.strptime(match.group("timestamp"), "%Y-%m-%d %H:%M:%S,%f")
            function_name = match.group("function_name")
            request_id = match.group("request_id")
            data_string = match.group("data_string")
            request_data.append([timestamp, request_id, function_name, data_string])

        match = re.search(response_data_pattern, line)
        if match:
            # print(f"({match} : {line})")
            timestamp = datetime.strptime(match.group("timestamp"), "%Y-%m-%d %H:%M:%S,%f")
            function_name = match.group("function_name")
            request_id = match.group("request_id")
            data_string = match.group("data_string")
            response_data.append([timestamp, request_id, function_name, data_string])

    return elasped_time_data, request_data, response_data

def merge_data_by_reqeust_id(elapsed_time_data, request_data, response_data):
    # 각 데이터를 dictionary로 변환
    elasped_time_dict = {entry[1]: entry for entry in elapsed_time_data if 'middleware' not in entry[2]} # middleware는 제외
    request_data_dict = {entry[1]: entry for entry in request_data}
    response_data_dict = {entry[1]: entry for entry in response_data}

    merged_data = []

    # 공통된 request_id를 가진 원소들을 합침
    for request_id in elasped_time_dict.keys() & request_data_dict.keys() & response_data_dict.keys():
        elased_entry = elasped_time_dict[request_id]
        request_entry = request_data_dict[request_id]
        response_entry = response_data_dict[request_id]

        timestamp = elased_entry[0]
        function_name = elased_entry[2]
        elapsed_time = elased_entry[3]
        user_id =elased_entry[-1]
        input_data = request_entry[3]
        output_data = response_entry[3]

        # func_name에 따라 다르게 파싱
        if "search" in function_name:
            # 검색기능과 관련된 함수인 경우
            # 1) 문자열에서 doc_id들을 파싱
            doc_ids = output_data.split(", ")
            # 2) ES에서 doc_id 기준으로 문서 검색
            es_response = search_documents_in_elasticsearch(doc_ids)
            es_response = [res['hits']['hits'][0] for res in es_response]
            es_response = [{'chunk_context': res['_source']['chunk_context'],
                           'chunk_src': res['_source']['chunk_src'],
                           'doc_type': res['_source']['doc_type'],
                           'chunk_id': res['_id']} for res in es_response]
            output_data = {'output_dict':es_response}
        elif "format_query" in function_name:
            # LLM 프롬프팅과 관련된 함수인 경우
            # 1) 문자열에서 doc_id들을 파싱
            doc_ids = output_data.split(", ")
            # 2) ES에서 doc_id 기준으로 문서 검색
            es_response = search_documents_in_elasticsearch(doc_ids)
            es_response = [res['hits']['hits'][0] for res in es_response]
            list_docs = [{'doc_text': res['_source']['chunk_context'],
                           'doc_src': res['_source']['chunk_src'],
                           'doc_type': res['_source']['doc_type'],
                           'chunk_id': res['_id']} for res in es_response]
            # 3) 문서 내용 병합
            merged_text = merge_chunks(list_docs)
            # 4) 프롬프트 생성
            rag_prompt = get_rag_prompt(input_data, merged_text)
            
            output_data = {'output_dict':{'doc_ids': output_data,
                                         'prompt': rag_prompt}}
        else:
            # 일반적인 경우
            output_data = {'output_str': output_data}
        
        # 필요한 데이터를 합치기
        merged_entry = (
            request_id,
            timestamp, 
            function_name, 
            elapsed_time, 
            input_data, 
            output_data,
            user_id,
        )
        merged_data.append(merged_entry)

    return merged_data

# 2. 데이터를 1분 간격으로 그룹화
def group_data_by_minute(data):
    grouped_data = defaultdict(lambda: defaultdict(list))
    start_time = min(entry[0] for entry in data)
    end_time = max(entry[0] for entry in data)
    current_time = start_time

    while current_time <= end_time:
        next_minute = current_time + timedelta(minutes=1)
        for timestamp, _, function_name, elapsed_time, _ in data:
            if current_time <= timestamp < next_minute:
                grouped_data[current_time][function_name].append(elapsed_time)
        current_time = next_minute

    return grouped_data

# 3. 각 그룹의 평균값 계산
def calculate_average_per_minute(grouped_data):
    average_data = defaultdict(dict)
    for minute, functions in grouped_data.items():
        for function_name, times in functions.items():
            average_data[minute][function_name] = sum(times) / len(times) if times else 0
    return average_data

# 4. 그래프 시각화
def plot_average_data(average_data, output_file="./app/execution_time_log/average_elapsed_time.png"):
    plt.figure(figsize=(12, 6))
    functions = set(fn for minute_data in average_data.values() for fn in minute_data.keys())

    for function_name in functions:
        times = []
        averages = []
        for minute, functions in average_data.items():
            times.append(minute)
            averages.append(functions.get(function_name, 0))

        plt.plot(times, averages, label=function_name)

    plt.xlabel("Time (minute)")
    plt.ylabel("Average elapsed time (seconds)")
    plt.title("Average Elapsed Time per Minute")
    plt.legend()
    plt.grid()

    # 그래프를 파일로 저장
    plt.savefig(output_file)
    print(f"Graph saved as {os.getcwd()}/{output_file}")
    plt.close()

def index_merged_data(es_client, index_name, merged_data):
    # 모든 request_id를 추출
    request_ids = [entry[0] for entry in merged_data]
    # =============================================================
    # 이미 존재하는 request_id가 있는지 순차적으로 조회.
    # request_ids를 한 번에 조회하면, search 메소드의 size 제한 때문에 일부 존재하는 request_id가 누락되어 response되기 때문
    existing_request_ids = set()
    for r_id in request_ids:
        search_body = {
            "query": {
                "terms": {
                    "request_id": [r_id]
                }
            }
        }
        response = es_client.search(index=index_name, body=search_body, size=100)
        existing_request_ids.update({hit['_source']['request_id'] for hit in response['hits']['hits']})

    # =============================================================

    # # Elasticsearch에서 이미 존재하는 request_id를 한 번에 조회
    # search_body = {
    #     "query": {
    #         "terms": {
    #             "request_id": request_ids
    #         }
    #     }
    # }
    # response = es_client.search(index=index_name, body=search_body, size=len(request_ids))
   
    # # 이미 존재하는 request_id를 집합으로 저장
    # existing_request_ids = {hit['_source']['request_id'] for hit in response['hits']['hits']}
   
    # 중복되지 않은 데이터만 actions에 추가
    actions = [
        {
            "_index": index_name,
            "_source": {
                'server_ip': settings.SERVER_IP,
                "request_id": entry[0],
                "timestamp": entry[1].astimezone(timezone.utc).isoformat(),
                "func_name": entry[2],
                "elapsed_time": entry[3],
                "input_data": entry[4],
                "output_data": entry[5],
                "user_id": entry[6],
            }
        }
        for entry in merged_data if entry[0] not in existing_request_ids
    ]

    
    s_time = time.time()
    # 중복되지 않은 데이터만 bulk로 저장
    if actions:
       helpers.bulk(es_client, actions)
    print(f"index_merged_data (#{len(actions)}): {time.time()-s_time}s")
    
def index_elasped_time(es_client, index_name, elapsed_time_data):
    # 모든 request_id를 추출
    request_ids = [entry[1] for entry in elapsed_time_data]

    # =============================================================
    # 이미 존재하는 request_id가 있는지 순차적으로 조회.
    # request_ids를 한 번에 조회하면, search 메소드의 size 제한 때문에 일부 존재하는 request_id가 누락되어 response되기 때문
    existing_request_ids = set()
    for r_id in request_ids:
        search_body = {
            "query": {
                "terms": {
                    "request_id": [r_id]
                }
            }
        }
        response = es_client.search(index=index_name, body=search_body, size=100)
        existing_request_ids.update({hit['_source']['request_id'] for hit in response['hits']['hits']})

    # =============================================================
    
    # # Elasticsearch에서 이미 존재하는 request_id를 한 번에 조회
    # search_body = {
    #     "query": {
    #         "terms": {
    #             "request_id": request_ids
    #         }
    #     }
    # }
    # response = es_client.search(index=index_name, body=search_body, size=len(request_ids))

    # # 이미 존재하는 request_id를 집합으로 저장
    # existing_request_ids = {hit['_source']['request_id'] for hit in response['hits']['hits']}
    
    actions = [
        {
            "_index": index_name,
            "_source":{
                'server_ip': settings.SERVER_IP,
                "request_id" : entry[1],
                "timestamp" : entry[0].astimezone(timezone.utc).isoformat(), # datetime.now()
                "func_name" : entry[2],
                "elapsed_time": entry[3],
                "user_id": entry[4],
            }
        }
        for entry in elapsed_time_data if entry[1] not in existing_request_ids
    ]

    import time
    s_time = time.time()
    # 중복되지 않은 데이터만 bulk로 저장
    if actions:
        helpers.bulk(es_client, actions)
    print(f"index_elasped_time (#{len(actions)}) : {time.time()-s_time}s")


# def index_merged_data(es_client, index_name, merged_data):
#     actions = [
#         {
#             "_index": index_name,
#             "_source":{
#                 'server_ip': settings.SERVER_IP,
#                 "request_id" : entry[0],
#                 "timestamp" : entry[1].astimezone(timezone.utc).isoformat(), # datetime.now()
#                 "func_name" : entry[2],
#                 "elapsed_time": entry[3],
#                 "input_data": entry[4],
#                 "output_data": entry[5],
#                 "user_id": entry[6],
#             }
#         }
#         for entry in merged_data
#     ]
#     import time
#     s_time = time.time()
#     helpers.bulk(es_client, actions)
#     print(f"index_merged_data : {time.time()-s_time}s")


# def index_elasped_time(es_client, index_name, elapsed_time_data):
#     actions = [
#         {
#             "_index": index_name,
#             "_source":{
#                 'server_ip': settings.SERVER_IP,
#                 "request_id" : entry[1],
#                 "timestamp" : entry[0].astimezone(timezone.utc).isoformat(), # datetime.now()
#                 "func_name" : entry[2],
#                 "elapsed_time": entry[3],
#                 "user_id": entry[4],
#             }
#         }
#         for entry in elapsed_time_data
#     ]

#     import time
#     s_time = time.time()
#     helpers.bulk(es_client, actions)
#     print(f"index_elasped_time : {time.time()-s_time}s")


def process_log_file(log_file):
    func_usage_index_name = 'func-usage-logs'
    elapsed_time_index_name = "elapsed-time-logs"
    # 로그 데이터 추출
    elapsed_time_data, request_data, response_data = extract_log_data(log_file)

    # 로그 데이터 병합 : ES에 저장하기 위함
    merged_data = merge_data_by_reqeust_id(elapsed_time_data, request_data, response_data)

    try:
        index_merged_data(es_client, index_name=func_usage_index_name, merged_data=merged_data)
        es_client.indices.refresh(index=func_usage_index_name)
        
        index_elasped_time(es_client, index_name=elapsed_time_index_name, elapsed_time_data=elapsed_time_data)
        es_client.indices.refresh(index=elapsed_time_index_name)
        
        # elasped_time을 1분 간격으로 시각화
        grouped_data = group_data_by_minute(elapsed_time_data)
        average_data = calculate_average_per_minute(grouped_data)
        plot_average_data(average_data)
    
    except Exception as e:
        traceback.print_exc()
        print(f"Got Error : {e}")

# log 파일 모니터링
class LogHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory: return
        if event.src_path.endswith(".log.1"):
            # 파일크기 기준 log 백업파일 감지
            print(f"[{time.time()}]New log file detected: {event.src_path}")
            threading.Thread(target=process_log_file, args=(event.src_path,)).start()

        if re.search(r"\.log\.\d{4}-\d{2}-\d{2}_\d{2}-\d{2}$", event.src_path):
            # 시간 기준 log 백업 감지
            print(f"[{time.time()}]New log file detected: {event.src_path}")
            threading.Thread(target=process_log_file, args=(event.src_path,)).start()
    
    def on_moved(self, event):
        if event.is_directory: return
        if event.dest_path.endswith(".log.1"):
            # 파일크기 기준 log 백업파일 감지
            print(f"[{time.time()}]Log file moded detected: {event.dest_path}")
            threading.Thread(target=process_log_file, args=(event.dest_path,)).start()
        
        if re.search(r"\.log\.\d{4}-\d{2}-\d{2}_\d{2}-\d{2}$", event.dest_path):
            # 시간 기준 log 백업 감지
            print(f"[{time.time()}] Log file moved detected: {event.dest_path}")
            threading.Thread(target=process_log_file, args=(event.dest_path,)).start()
            
# 5. 실행
if __name__ == "__main__":
    log_file = './app/execution_time_log/log_files'
    backup_file = './app/execution_time_log/log_backup'

    # Elasticsearch에 저장
    if settings.APP_ENVIRONMENT == 'production':
        es_client = Elasticsearch("http://127.0.0.1:9200", http_auth=("kdb", "kdbAi1234!")) # development 환경
    else:
        es_client = Elasticsearch("http://127.0.0.1:9200") # development 환경
    
    event_handler = LogHandler()
    observer = Observer()
    observer.schedule(event_handler, log_file, recursive=False)
    observer.start()

    print(f'{settings.APP_SERVER} starting')
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Program interrupted by user")
    finally:
        observer.stop()
        observer.join()
        print("Observer stopped and resources cleaned up.")