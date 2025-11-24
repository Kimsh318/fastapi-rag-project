from elasticsearch import Elasticsearch, helpers
import requests
from prometheus_client.parser import text_string_to_metric_families
import time
from datetime import datetime, timezone
import psutil
from collections import defaultdict
from copy import deepcopy
# Elasticsearch 클라이언트 초기화
# es_client = Elasticsearch("http://172.18.0.5:9200") # prototype 환경
# es_client = Elasticsearch("http://127.0.0.1:9200") # development 환경
es_client = Elasticsearch("http://127.0.0.1:9200", http_auth=("kdb", "kdbAi1234!")) # production 환경

server_ip = 'http://10.6.40.78:32117' # serving 1
index_name = "app-metric-logs"


# 이전 http_total_requests_total 값을 저장할 변수
previous_total_requests = None
previous_success_requests = None
previous_failure_requests = None
previous_api_requests = None

total_memory = psutil.virtual_memory().total
print(f"total memory {total_memory}")

def diff_dict(dict1, dict2):
    diff = defaultdict(dict)
    # for key in dict1:
    #     for nested_key in dict1[key]:
    #         if nested_key in dict2[key]:
    #             diff[key][nested_key] = dict1[key][nested_key] - dict2[key][nested_key]

    for key in dict1:
        if 'metrics' in key: continue
        for nested_key in dict1[key]:
            if nested_key in dict2[key]:
                diff[key][nested_key] = dict1[key][nested_key] - dict2[key][nested_key]
    # 비율 계산
    for key in diff:
        completed, success, failure = diff[key]['api_completed_requests_total'], diff[key]['api_success_requests_total'], diff[key]['api_failed_requests_total']
        if success+failure:
            diff[key]['api_success_requests_rate'] = (success / (success +failure)) * 100
            diff[key]['api_failed_requests_rate'] = 100 - diff[key]['api_success_requests_rate']
        
    
    # # 비율 계산
    # for key in diff:
    #     # api request가 모두 성공이거나, 실패이면, key 값이 없을수도 있음
    #     # 두 key 모두 검증
    #     if 'api_success_requests_total' in diff[key] and  'api_failed_requests_total' in diff[key]:
    #         total, success, failure = diff[key]['api_total_requests_total'], diff[key]['api_success_requests_total'], diff[key]['api_failed_requests_total']
    #         concurrnet = total - (success + failure)
    #     elif 'api_success_requests_total' in diff[key]:
    #         total, success, failure = diff[key]['api_total_requests_total'], diff[key]['api_success_requests_total'], 0
    #         concurrnet = total - (success + failure)
    #     elif 'api_failed_requests_total' in diff[key]:
    #         total, success, failure = diff[key]['api_total_requests_total'], 0, diff[key]['api_failed_requests_total']
    #         concurrnet = total - (success + failure)

    #     diff[key]['api_success_requests_rate'] = (success / (success +failure)) * 100
    #     diff[key]['api_failed_requests_rate'] = 100 - diff[key]['api_success_requests_rate']
    #     diff[key]['concurrent'] = concurrnet
    
        # if 'api_success_requests_total' in diff[key] and diff[key]['api_total_requests_total']:
        #     total, success, failure = diff[key]['api_total_requests_total'], diff[key]['api_success_requests_total'], diff[key]['api_failed_requests_total']
        #     concurrnet = total - (success + failure)
        #     diff[key]['api_success_requests_rate'] = (success / (success +failure)) * 100
        #     diff[key]['api_failed_requests_rate'] = 1 - diff[key]['api_success_requests_rate']
        #     #diff[key]['api_success_requests_rate'] = diff[key]['api_success_requests_total'] / diff[key]['api_total_requests_total'] * 100        
        #     print(f"diff[{key}]['api_success_requests_rate'] = {diff[key]['api_success_requests_total']} / {diff[key]['api_total_requests_total']} * 100 ")
        # elif 'api_failed_requests_total' in diff[key] and diff[key]['api_total_requests_total']:
        #     diff[key]['api_failed_requests_rate'] = diff[key]['api_failed_requests_total'] / diff[key]['api_total_requests_total'] * 100
        #     print(f"diff[{key}]['api_failed_requests_rate'] = {diff[key]['api_failed_requests_total']} / {diff[key]['api_total_requests_total']} * 100 ")
        
    return diff

def parse_and_store_metrics():
    global previous_total_requests
    global previous_success_requests
    global previous_failure_requests
    global previous_api_requests

    # 메트릭 데이터 수집
    response = requests.get(server_ip+'/metrics')
    metrics_data = response.text

    # 필요한 메트릭 이름
    required_metrics = {
        'http_total_requests_total',  # Counter 타입은 _total 접미사가 붙음
        'http_success_requests_total',
        'http_failed_requests_total',
        
        'api_total_requests_total',
        'api_success_requests_total',
        'api_failed_requests_total',
        
        #'worker_cpu_usage',  # CPU 사용량
        #'worker_memory_usage_percent',  # 메모리 사용량
        #'worker_gpu_memory_usage_percent',
        #'worker_gpu_utilization',
    }

    # 메트릭 데이터 파싱
    parsed_metrics = defaultdict(list)

    for family in text_string_to_metric_families(metrics_data):
        for sample in family.samples:
            # 각 메트릭의 이름과 값을 저장
            metric_name = sample.name
            metric_value = sample.value
            metric_labels = sample.labels

            # 필요한 메트릭만 저장
            if metric_name in required_metrics:
                # if metric_name not in parsed_metrics:
                #     parsed_metrics[metric_name] = []

                # 1. 전체, worker 구분하여 metric 저장
                if metric_labels: #'worker' in metric_name:
                    # worker별 metric 저장 : labels 필요
                    parsed_metrics[metric_name].append({
                        'value': metric_value,
                        'labels': metric_labels
                    })
                else:
                    # 전체 metric 저장
                    parsed_metrics[metric_name].append({
                        'value': metric_value,
                    })


    # endpoint 기준 : 전체, 성공, 실패 request 수
    requests_per_api = defaultdict(dict)
    for metric_name in ['api_total_requests_total', 'api_success_requests_total', 'api_failed_requests_total']:
        for data in  parsed_metrics[metric_name]:
            endpoint = data['labels']['endpoint']
            value = data['value']
            requests_per_api[endpoint].update({
                metric_name: value
            })
            
    # endpoint 기준 : concurrent 계산
    for endpoint, request_info in requests_per_api.items():
        if 'api_success_requests_total' not in request_info: request_info['api_success_requests_total'] = 0
        if 'api_failed_requests_total' not in request_info: request_info['api_failed_requests_total'] = 0
        total, success, failure = request_info['api_total_requests_total'], request_info['api_success_requests_total'], request_info['api_failed_requests_total']
        completed = success + failure
        concurrent = total - completed
        
        request_info.update({
            'api_completed_requests_total': completed,
            'api_concurrent_requests_total': concurrent
        })
    
    # 1분 동안 수행된 요청 수 계산
    if 'http_total_requests_total' in parsed_metrics:
        current_total_requests = sum(sample['value'] for sample in parsed_metrics['http_total_requests_total'])
        current_success_requests = sum(sample['value'] for sample in parsed_metrics['http_success_requests_total'])
        current_failure_requests = sum(sample['value'] for sample in parsed_metrics['http_failed_requests_total'])

        current_api_reuqests = deepcopy(requests_per_api)

        succes_rate_in_last_minutef = None
        
        if previous_total_requests is not None:
            requests_in_last_minute = current_total_requests - previous_total_requests
            success_in_last_minute = current_success_requests - previous_success_requests
            failure_in_last_minute = current_failure_requests - previous_failure_requests

            # Metric 수집 중 fastapi app 재실행되면 0이하 값 가질 수 있음
            if requests_in_last_minute <0: return None
            
            if success_in_last_minute + failure_in_last_minute != 0:
                succes_rate_in_last_minute = round((success_in_last_minute/(success_in_last_minute+failure_in_last_minute))*100,2)
                failure_rate_in_last_minute = 100 - succes_rate_in_last_minute
            
            concurrent_requests = current_total_requests - (current_success_requests+current_failure_requests)
            
            diff_requests_in_last_minute = diff_dict(current_api_reuqests, previous_api_requests)
           
            print(f"Requests in the last minute: {requests_in_last_minute}")


            # Request 관련 metric 생성
            if succes_rate_in_last_minute:
                request_metric_data = {
                    'server_ip': server_ip,
                    'metric_type': 'requests',
                    'metric_data': {
                        'total':{
                            'total_requests': parsed_metrics['http_total_requests_total'][0]['value'],
                            'concurrent_requests': concurrent_requests,
                            'requests_per_minutes': requests_in_last_minute,
                            'success_requests_per_minutes': success_in_last_minute,
                            'failure_requests_per_minutes': failure_in_last_minute,
                            'success_rate_per_minutes': succes_rate_in_last_minute,
                            'failure_rate_per_minutes': failure_rate_in_last_minute,
                        },
                        'api': diff_requests_in_last_minute
                    },
                    'timestamp': datetime.now(timezone.utc).isoformat(),#.isoformat()+'Z',
                }
            else:
                request_metric_data = {
                    'server_ip': server_ip,
                    'metric_type': 'requests',
                    'metric_data': {
                        'total':{
                            'total_requests': parsed_metrics['http_total_requests_total'][0]['value'],
                            'concurrent_requests': concurrent_requests,
                            'requests_per_minutes': requests_in_last_minute,
                            'success_requests_per_minutes': success_in_last_minute,
                            'failure_requests_per_minutes': failure_in_last_minute,
                        },
                        'api': diff_requests_in_last_minute
                    },
                    'timestamp': datetime.now(timezone.utc).isoformat(),#.isoformat()+'Z',
                }
            # Elasticsearch에 인덱싱
            es_client.index(index=index_name, body=request_metric_data)

            # # 자원사용량 관련 metric 생성
            # resource_metric_data = {
            #     'server_ip': server_ip,
            #     'metric_type': 'resource_usage',
            #     'metric_data': {
            #         'worker_cpu_usage': parsed_metrics['worker_cpu_usage'],
            #         'worker_memory_usage_percent': parsed_metrics['worker_memory_usage_percent'],
            #     },
            #     'timestamp': datetime.now(timezone.utc).isoformat(),#.isoformat()+'Z',
                
            # }
            # # Elasticsearch에 인덱싱
            # es_client.index(index=index_name, body=resource_metric_data)

        # 이전 값을 현재 값으로 업데이트
        previous_total_requests = current_total_requests
        previous_success_requests = current_success_requests
        previous_failure_requests = current_failure_requests
        previous_api_requests = current_api_reuqests

    # 파싱결과 출력
    # for metric_name, samples in parsed_metrics.items():
    #     print(f"Metric: {metric_name}")
    #     for sample in samples:
    #         print(sample)
    #         if 'labels' in sample:#and 'pid' in sample['labels']:
    #             print(f"  Value: {sample['value']}, Labels: {sample['labels']}")
    #         else:
    #             print(f"  Value: {sample['value']}")
    print('==============================')
# 1분마다 파싱 및 저장 작업 수행
while True:
    try:
        parse_and_store_metrics()
        time.sleep(60)
    except Exception as e:
        print(e)
        time.sleep(60)
        
 