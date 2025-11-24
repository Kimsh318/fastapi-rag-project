import re
import os
from datetime import datetime, timedelta
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from elasticsearch import Elasticsearch, helpers

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 1. 로그 파일에서 데이터 추출
def extract_log_data(log_file):
    elasped_time_pattern = r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .+ - INFO - \[(?P<request_id>.+?)\] (?P<function_name>\w+) completed in (?P<elapsed_time>\d+\.\d+)s"

    request_data_pattern =  r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .+ - INFO - \[(?P<request_id>.+?)\] (?P<function_name>\w+) request data : (?P<data_string>.+)"
    
    response_data_pattern =  r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .+ - INFO - \[(?P<request_id>.+?)\] (?P<function_name>\w+) response data : (?P<data_string>.+)"
    
    elasped_time_data, request_data, response_data = [], [], []

    with open(log_file, "r") as file:
        for line in file:
            print(line)
            match = re.search(elasped_time_pattern, line)
            if match:
                print(f"({match} : {line})")
                timestamp = datetime.strptime(match.group("timestamp"), "%Y-%m-%d %H:%M:%S,%f")
                function_name = match.group("function_name")
                request_id = match.group("request_id")
                elapsed_time = float(match.group("elapsed_time"))
                elasped_time_data.append([timestamp, request_id, function_name, elapsed_time])

            match = re.search(request_data_pattern, line)
            if match:
                print(f"({match} : {line})")
                timestamp = datetime.strptime(match.group("timestamp"), "%Y-%m-%d %H:%M:%S,%f")
                function_name = match.group("function_name")
                request_id = match.group("request_id")
                data_string = match.group("data_string")
                request_data.append([timestamp, request_id, function_name, data_string])

            match = re.search(response_data_pattern, line)
            if match:
                print(f"({match} : {line})")
                timestamp = datetime.strptime(match.group("timestamp"), "%Y-%m-%d %H:%M:%S,%f")
                function_name = match.group("function_name")
                request_id = match.group("request_id")
                data_string = match.group("data_string")
                response_data.append([timestamp, request_id, function_name, data_string])

    return elasped_time_data, request_data, response_data

def merge_data_by_reqeust_id(elapsed_time_data, request_data, response_data):
    # 각 데이터를 dictionary로 변환
    elasped_time_dict = {entry[1]: entry for entry in elapsed_time_data}
    request_data_dict = {entry[1]: entry for entry in request_data}
    response_data_dict = {entry[1]: entry for entry in response_data}

    merged_data = []

    # 공통된 request_id를 가진 원소들을 합침
    for request_id in elasped_time_dict.keys() & request_data_dict.keys() & response_data_dict.keys():
        elased_entry = elasped_time_dict[request_id]
        request_entry = request_data_dict[request_id]
        response_entry = response_data_dict[request_id]

        # 필요한 데이터를 합치기
        merged_entry = (
            request_id,
            elased_entry[0], # timestamp
            elased_entry[2], # function_name
            elased_entry[3], # elapsed_time
            request_entry[3], # request data
            response_entry[3], # response data
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
        for timestamp, _, function_name, elapsed_time in data:
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
def plot_average_data(average_data, output_file="average_elapsed_time.png"):
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
    print(f"Graph saved as {output_file}")
    plt.close()

def index_merged_data(es_client, index_name, merged_data):
    actions = [
        {
            "_index": index_name,
            "_source":{
                "request_id" : entry[0],
                "timestamp" : entry[1],
                "endpoint" : entry[2],
                "elapsed_time": entry[3],
                "input_data": entry[4],
                "output_data": entry[5],
                "status": "samle_status_completed",
            }
        }
        for entry in merged_data
    ]
    helpers.bulk(es_client, actions)


def process_log_file(log_file):
    index_name = 'func-usage-logs'
    
    # 로그 데이터 추출
    elasped_time_data, request_data, response_data = extract_log_data(log_file)

    # 로그 데이터 병합 : ES에 저장하기 위함
    merged_data = merge_data_by_reqeust_id(elasped_time_data, request_data, response_data)

    # elasped_time을 1분 간격으로 시각화
    grouped_data = group_data_by_minute(elasped_time_data)
    average_data = calculate_average_per_minute(grouped_data)
    plot_average_data(average_data)

    # Elasticsearch에 저장
    es_client = Elasticsearch("http://127.0.0.1:9200") # development 환경
    index_merged_data(es_client, index_name=index_name, merged_data=merged_data)


# log 파일 모니터링
class LogHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory: return
        if event.src_path.endswith(".log.1"):
            print(f"[{time.time()}]New log file detected: {event.src_path}")
            process_log_file(event.src_path)
    
        
# 5. 실행
if __name__ == "__main__":
    log_file = './'
    event_handler = LogHandler()
    observer = Observer()
    observer.schedule(event_handler, log_file, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Program interrupted by user")
    finally:
        observer.stop()
        observer.join()
        print("Observer stopped and resources cleaned up.")