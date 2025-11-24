from elasticsearch import Elasticsearch, NotFoundError
from datetime import datetime, timezone, timedelta

import time

# Elasticsearch 클라이언트 초기화
es_client = Elasticsearch("http://127.0.0.1:9200") # ("http://172.18.0.5:9200")
def format_elapsed_time(time_in_s):
    """경과 시간을 보기 좋게 포맷팅"""
    if time_in_s is None:
        return "N/A"
    return f"{time_in_s:.3f}s"

def create_table_row(columns, widths):
    """테이블 행 생성"""
    row = "│"
    for col, width in zip(columns, widths):
        row += f" {str(col):<{width}} │"
    return row

def print_horizontal_line(widths, symbol="─"):
    """수평선 생성"""
    line = "├" if symbol == "─" else "┌" if symbol == "┄" else "└"
    for width in widths:
        line += symbol * (width + 2) + "┤" if symbol == "─" else symbol * (width + 2) + "┐" if symbol == "┄" else symbol * (width + 2) + "┘"
    return line



def print_middleware_results(response):
    """집계 결과를 테이블 형식으로 출력"""
    if not response or "aggregations" not in response:
        print("No aggregation results found")
        return

    # 헤더 정의
    headers = ["Endpoint", "Average Time",  "Average Logging Time","Request Count"]
    
    # 데이터 준비
    table_data = []
    buckets = response["aggregations"]["endpoint_stats"]["buckets"]
    
    for bucket in buckets:
        endpoint = bucket["key"]
        avg_time = bucket["average_elapsed_time"]["value"]
        avg_logging_time = bucket.get("average_logging_elapsed_time", {}).get("value")
        doc_count = f"{bucket['doc_count']:,}"
        table_data.append([endpoint, format_elapsed_time(avg_time), format_elapsed_time(avg_logging_time), doc_count])

    # 각 컬럼의 최대 너비 계산
    widths = []
    for i in range(len(headers)):
        column_values = [str(row[i]) for row in table_data] + [headers[i]]
        widths.append(max(len(str(value)) for value in column_values))

    # 결과 출력
    print("\n===HTTP Middleware Performance Analysis ===")
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 테이블 출력
    print(print_horizontal_line(widths, "┄"))
    print(create_table_row(headers, widths))
    print(print_horizontal_line(widths))
    
    for row in table_data:
        print(create_table_row(row, widths))
    
    print(print_horizontal_line(widths, "═"))

    # 요약 통계 출력
    total_requests = sum(bucket["doc_count"] for bucket in buckets)
    print(f"\nTotal Endpoints: {len(table_data)}")
    print(f"Total Requests: {total_requests:,}")
    
def print_aggregation_results(response):
    """집계 결과를 테이블 형식으로 출력"""
    if not response or "aggregations" not in response:
        print("No aggregation results found")
        return

    # 헤더 정의
    headers = ["Endpoint", "Average Time",  "Average Logging Time","Request Count"]
    
    # 데이터 준비
    table_data = []
    buckets = response["aggregations"]["endpoint_stats"]["buckets"]
    
    for bucket in buckets:
        endpoint = bucket["key"]
        avg_time = bucket["average_elapsed_time"]["value"]
        avg_logging_time = bucket.get("average_logging_elapsed_time", {}).get("value")
        doc_count = f"{bucket['doc_count']:,}"
        table_data.append([endpoint, format_elapsed_time(avg_time), format_elapsed_time(avg_logging_time), doc_count])

    # 각 컬럼의 최대 너비 계산
    widths = []
    for i in range(len(headers)):
        column_values = [str(row[i]) for row in table_data] + [headers[i]]
        widths.append(max(len(str(value)) for value in column_values))

    # 결과 출력
    print("\n=== Endpoint Performance Analysis ===")
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 테이블 출력
    print(print_horizontal_line(widths, "┄"))
    print(create_table_row(headers, widths))
    print(print_horizontal_line(widths))
    
    for row in table_data:
        print(create_table_row(row, widths))
    
    print(print_horizontal_line(widths, "═"))

    # 요약 통계 출력
    total_requests = sum(bucket["doc_count"] for bucket in buckets)
    print(f"\nTotal Endpoints: {len(table_data)}")
    print(f"Total Requests: {total_requests:,}")

def print_service_aggregation_results(response):
    """서비스별 집계 결과를 테이블 형식으로 출력"""
    if not response or "aggregations" not in response:
        print("No aggregation results found")
        return

    # 헤더 정의
    headers = ["Service Name", "Average Time",  "Average Logging Time","Request Count"]
    
    # 데이터 준비
    table_data = []
    buckets = response["aggregations"]["service_stats"]["buckets"]
    
    for bucket in buckets:
        service_name = bucket["key"]
        avg_time = bucket.get("average_elapsed_time", {}).get("value")
        avg_logging_time = bucket.get("average_logging_elapsed_time", {}).get("value")
        doc_count = f"{bucket['doc_count']:,}"
        table_data.append([service_name, format_elapsed_time(avg_time), format_elapsed_time(avg_logging_time), doc_count])

    # 각 컬럼의 최대 너비 계산
    widths = []
    for i in range(len(headers)):
        column_values = [str(row[i]) for row in table_data] + [headers[i]]
        widths.append(max(len(str(value)) for value in column_values))

    # 결과 출력
    print("\n=== Service Performance Analysis ===")
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 테이블 출력
    print(print_horizontal_line(widths, "┄"))
    print(create_table_row(headers, widths))
    print(print_horizontal_line(widths))
    
    for row in table_data:
        print(create_table_row(row, widths))
    
    print(print_horizontal_line(widths, "═"))

    # 요약 통계 출력
    total_requests = sum(bucket["doc_count"] for bucket in buckets)
    total_avg_time = sum(bucket.get("average_elapsed_time", {}).get("value", 0) * bucket["doc_count"] for bucket in buckets) / total_requests if total_requests > 0 else 0
    
    print(f"\nTotal Services: {len(table_data)}")
    print(f"Total Requests: {total_requests:,}")
    print(f"Overall Average Time: {format_elapsed_time(total_avg_time)}")


def print_processor_aggregation_results(response):
    """프로세서별 집계 결과를 테이블 형식으로 출력"""
    if not response or "aggregations" not in response:
        print("No aggregation results found")
        return

    # 헤더 정의
    headers = ["Processor Name", "Average Func Time", "Average Logging Time", "Request Count"]
    
    # 데이터 준비
    table_data = []
    buckets = response["aggregations"]["processor_stats"]["buckets"]
    
    print("\n=== Processor buckets ===")
    print(buckets)
    for bucket in buckets:
        service_name = bucket["key"]
        avg_time = bucket.get("average_elapsed_time", {}).get("value")
        avg_logging_time = bucket.get("average_logging_elapsed_time", {}).get("value")
        doc_count = f"{bucket['doc_count']:,}"
        table_data.append([service_name, format_elapsed_time(avg_time), format_elapsed_time(avg_logging_time), doc_count])

    # 각 컬럼의 최대 너비 계산
    widths = []
    for i in range(len(headers)):
        column_values = [str(row[i]) for row in table_data] + [headers[i]]
        widths.append(max(len(str(value)) for value in column_values))

    # 결과 출력
    print("\n=== Processor Performance Analysis ===")
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 테이블 출력
    print(print_horizontal_line(widths, "┄"))
    print(create_table_row(headers, widths))
    print(print_horizontal_line(widths))
    
    for row in table_data:
        print(create_table_row(row, widths))
    
    print(print_horizontal_line(widths, "═"))

    # 요약 통계 출력
    total_requests = sum(bucket["doc_count"] for bucket in buckets)
    total_avg_time = sum(bucket.get("average_elapsed_time", {}).get("value", 0) * bucket["doc_count"] for bucket in buckets) / total_requests if total_requests > 0 else 0
    
    print(f"\nTotal Processors: {len(table_data)}")
    print(f"Total Requests: {total_requests:,}")
    print(f"Overall Average Time: {format_elapsed_time(total_avg_time)}")



def print_etc_func_results(response):
    """집계 결과를 테이블 형식으로 출력"""
    if not response or "aggregations" not in response:
        print("No aggregation results found")
        return

    # 헤더 정의
    headers = ["Endpoint", "Average Time",  "Average Logging Time","Request Count"]
    
    # 데이터 준비
    table_data = []
    buckets = response["aggregations"]["endpoint_stats"]["buckets"]
    
    for bucket in buckets:
        endpoint = bucket["key"]
        avg_time = bucket["average_elapsed_time"]["value"]
        avg_logging_time = bucket.get("average_logging_elapsed_time", {}).get("value")
        doc_count = f"{bucket['doc_count']:,}"
        table_data.append([endpoint, format_elapsed_time(avg_time), format_elapsed_time(avg_logging_time), doc_count])

    # 각 컬럼의 최대 너비 계산
    widths = []
    for i in range(len(headers)):
        column_values = [str(row[i]) for row in table_data] + [headers[i]]
        widths.append(max(len(str(value)) for value in column_values))

    # 결과 출력
    print("\n===Etc Function Performance Analysis ===")
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 테이블 출력
    print(print_horizontal_line(widths, "┄"))
    print(create_table_row(headers, widths))
    print(print_horizontal_line(widths))
    
    for row in table_data:
        print(create_table_row(row, widths))
    
    print(print_horizontal_line(widths, "═"))

    # 요약 통계 출력
    total_requests = sum(bucket["doc_count"] for bucket in buckets)
    print(f"\nTotal Endpoints: {len(table_data)}")
    print(f"Total Requests: {total_requests:,}")

def get_query(index_name):
    num_data = 1260
    if index_name == "http-middleware-logs":
        return {
            "size": 0,
            "sort": [
                {"timestamp": {"order":"desc"}}  
            ],
            "aggs": {
                "endpoint_stats": {
                    "terms": {
                        "field": "url",
                        "size": num_data, # 10000
                    },
                    "aggs": {
                        "average_elapsed_time": {
                            "avg": {
                                "field": "elapsed_time"
                                }
                            },
                        "average_logging_elapsed_time":{
                            "avg": {
                                "field": "logging_elapsed_time"  # elapsed_time 필드가 존재한다고 가정
                            }
                        }
                    }
                }
            }
        }
    elif index_name == "api-usage-logs":
        return {
            "size": 0,
            "sort": [
                {"timestamp": {"order":"desc"}}  
            ],
            "aggs": {
                "endpoint_stats": {
                "terms": {
                    "field": "endpoint",
                    "size": num_data, # 10000
                },
                "aggs": {
                    "average_elapsed_time": {
                        "avg": {
                            "field": "elapsed_time"
                            }
                        },
                    "average_logging_elapsed_time":{
                        "avg": {
                            "field": "logging_elapsed_time"  # elapsed_time 필드가 존재한다고 가정
                            }
                        }
                    }
                }
            }
        }
    elif index_name == "service-usage-logs":
        return {
            "size": 0,
            "sort": [
                {"timestamp": {"order":"desc"}}  
            ],
            "aggs": {
                "service_stats": {
                    "terms": {
                        "field": "service_name",
                        "size": num_data, # 10000
                    },
                    "aggs": {
                        "average_elapsed_time": {
                            "avg": {
                                "field": "elapsed_time"  # elapsed_time 필드가 존재한다고 가정
                            }
                        },
                        "average_logging_elapsed_time":{
                            "avg": {
                                "field": "logging_elapsed_time"  # elapsed_time 필드가 존재한다고 가정
                            }
                        }
                    }
                }
            }
        }
    elif index_name == "processor-usage-logs":
        return {
            "size": 0,
            "sort": [
                {"timestamp": {"order":"desc"}}  
            ],
            "aggs": {
                "processor_stats": {
                    "terms": {
                        "field": "processor_name",
                        "size": num_data, # 10000
                    },
                    "aggs": {
                        "average_elapsed_time": {
                            "avg": {
                                "field": "elapsed_time"
                            }
                        },
                        "average_logging_elapsed_time":{
                            "avg": {
                                "field": "logging_elapsed_time"  # elapsed_time 필드가 존재한다고 가정
                            }
                        }
                    }
                }
            }
        }
    elif index_name == "etc-func-logs":
        return {
            "size": 0,
            "sort": [
                {"timestamp": {"order":"desc"}}  
            ],
            "aggs": {
                "endpoint_stats": {
                    "terms": {
                        "field": "func_name",
                        "size": num_data, # 10000
                    },
                    "aggs": {
                        "average_elapsed_time": {
                            "avg": {
                                "field": "elapsed_time"
                                }
                            },
                        "average_logging_elapsed_time":{
                            "avg": {
                                "field": "logging_elapsed_time"  # elapsed_time 필드가 존재한다고 가정
                            }
                        }
                    }
                }
            }
        }
    return None

def read_all_data(index_name):
    """
    인덱스의 모든 데이터를 조회하고, 각 endpoint에 대한 평균 elapsed_time을 계산하는 함수
    """
   # 집계 쿼리 활성화 및 수정
    agg_query = get_query(index_name)
    # Elasticsearch에서 데이터 검색 및 집계
    response = es_client.search(index=index_name, body=agg_query)

    if index_name == "http-middleware-logs":
        print_middleware_results(response)
    elif index_name == "api-usage-logs":
        print_aggregation_results(response)
    elif index_name == "service-usage-logs":
        print_service_aggregation_results(response)
    elif index_name == "processor-usage-logs":
        #print_processor_aggregation_results(response)
        print_processor_aggregation_results(response)
    elif index_name == "etc-func-logs":
        print_etc_func_results(response)
    # Elasticsearch에서 데이터 검색 및 집계
    # response = es_client.search(index=index_name, body=agg_query)
    # print(response)
    # # 결과 출력
    # print("각 endpoint별 평균 경과 시간:")
    # for bucket in response["aggregations"]["avg_elapsed_time_by_endpoint"]["buckets"]:
    #     endpoint = bucket["key"]
    #     avg_time = bucket["avg_elapsed_time"]["value"]
    #     print(f"{endpoint}: {avg_time}ms")

if __name__ == '__main__':
    # index 내 모든 데이터 확인
    print('======== http-middleware-logs ==========')
    read_all_data("http-middleware-logs")
    
    print('======== api-usage-data ==========')
    read_all_data("api-usage-logs")

    print('======== service-usage-data ==========')
    read_all_data("service-usage-logs")

    print('======== processor-usage-data ==========')
    read_all_data("processor-usage-logs")

    print('======== etc-func-data ==========')
    read_all_data("etc-func-logs")