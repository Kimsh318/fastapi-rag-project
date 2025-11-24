# ========================================
# request metric 수집(method 구분 x)
# 엔듶포인트별 구분
# ========================================
from prometheus_client import Counter, Gauge, CollectorRegistry, multiprocess, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import os
import psutil
import time
import subprocess
import re
from multiprocessing import Process

# Prometheus 메트릭 선언
counters = {}

def get_or_create_counter(metric_name, description, labels):
    """동적으로 Counter를 생성하여 확장성을 확보"""
    if metric_name not in counters:
        counters[metric_name] = Counter(metric_name, description, labels)
    return counters[metric_name]

# 요청 관련 메트릭 (동적 생성)
HTTP_TOTAL_REQUESTS = Counter('http_total_requests', 'Total HTTP requests count')
HTTP_SUCCESS_REQUESTS = Counter('http_success_requests', 'Total HTTP successful requests count')
HTTP_FAILED_REQUESTS = Counter('http_failed_requests', 'Total HTTP failed requests count')

API_TOTAL_REQUESTS = get_or_create_counter('api_total_requests', 'Total HTTP requests count', ['endpoint'])
API_SUCCESS_REQUESTS = get_or_create_counter('api_success_requests', 'Total HTTP successful requests count', ['endpoint'])
API_FAILED_REQUESTS = get_or_create_counter('api_failed_requests', 'Total HTTP failed requests count', ['endpoint'])

# 워커별 리소스 메트릭
CPU_USAGE = Gauge('worker_cpu_usage', 'CPU usage of the application in percentage', ['worker'])
MEMORY_USAGE_PERCENT = Gauge('worker_memory_usage_percent', 'Memory usage of the application in percentage', ['worker'])
GPU_MEMORY_USAGE_PERCENT = Gauge('worker_gpu_memory_usage_percent', 'GPU memory usage of the application in percentage', ['worker'])
GPU_UTILIZATION = Gauge('worker_gpu_utilization', 'GPU utilization of the application in percentage', ['worker', 'gpu'])

class PrometheusMiddleware(BaseHTTPMiddleware):
    """요청을 가로채어 Prometheus 메트릭을 업데이트하는 미들웨어"""
    async def dispatch(self, request: Request, call_next):
        endpoint = request.url.path

        # 요청 카운트 증가
        HTTP_TOTAL_REQUESTS.inc()
        
        API_TOTAL_REQUESTS.labels(endpoint=endpoint).inc()

        try:
            response = await call_next(request)
            status_code = response.status_code

            if 200 <= status_code < 300:
                HTTP_SUCCESS_REQUESTS.inc()
                API_SUCCESS_REQUESTS.labels(endpoint=endpoint).inc()
            else:
                HTTP_FAILED_REQUESTS.inc()
                API_FAILED_REQUESTS.labels(endpoint=endpoint).inc()

        except Exception:
            HTTP_FAILED_REQUESTS.inc()
            API_FAILED_REQUESTS.labels(endpoint=endpoint).inc()
            raise

        return response

def get_gpu_info():
    """nvidia-smi 명령을 통해 GPU 정보를 수집"""
    try:
        # nvidia-smi 실행
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")

        # 결과 파싱
        gpu_info = []
        for line in result.stdout.strip().split("\n"):
            total, used, utilization = map(int, re.split(", *", line))
            memory_usage_percent = (used / total) * 100 if total > 0 else 0
            gpu_info.append({
                "memory_usage_percent": memory_usage_percent,
                "utilization": utilization,
            })
        return gpu_info

    except FileNotFoundError:
        return []  # nvidia-smi가 없는 경우 빈 리스트 반환
    except Exception as e:
        raise RuntimeError(f"Failed to get GPU info: {str(e)}")

def record_worker_metrics(worker_id: str):
    """각 워커의 리소스 사용량을 주기적으로 기록"""
    while True:
        # CPU 사용량
        cpu_usage = psutil.cpu_percent()
        CPU_USAGE.labels(worker=worker_id).set(cpu_usage)

        # 메모리 사용량 (퍼센트)
        memory_usage_percent = psutil.virtual_memory().percent
        MEMORY_USAGE_PERCENT.labels(worker=worker_id).set(memory_usage_percent)

        # # GPU 정보 기록
        # gpu_info = get_gpu_info()
        # for idx, gpu in enumerate(gpu_info):
        #     GPU_MEMORY_USAGE_PERCENT.labels(worker=f"{worker_id}_gpu_{idx}").set(gpu['memory_usage_percent'])
        #     GPU_UTILIZATION.labels(worker=worker_id, gpu=f"gpu_{idx}").set(gpu['utilization'])

        time.sleep(30)  # 5초 간격으로 업데이트

def setup_metrics(app):
    """Prometheus 멀티프로세스 환경 설정 및 /metrics 엔드포인트 추가"""
    app.add_middleware(PrometheusMiddleware)

    # @app.on_event("startup")
    # async def start_worker_metrics():
    #     """각 워커별 메트릭 기록"""
    #     import multiprocessing
    #     worker_id = f"worker_{multiprocessing.current_process().pid}"
    #     os.environ["WORKER_ID"] = worker_id
    #     os.getenv("WORKER_ID", f"worker_{multiprocessing.current_process().pid}")
        
    #     process = Process(target=record_worker_metrics, args=(worker_id,), daemon=True)
    #     process.start()

    @app.get('/metrics')
    async def metrics():
        """Prometheus 메트릭 엔드포인트"""
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        
        return Response(content=generate_latest(registry), media_type="text/plain")
 



# ========================================
# request metric 수집(method 구분 x)
# ========================================

# from prometheus_client import Counter, Gauge, CollectorRegistry, multiprocess, generate_latest
# from starlette.middleware.base import BaseHTTPMiddleware
# from starlette.requests import Request
# from starlette.responses import Response
# import os
# import psutil
# import time

# try:
#     import GPUtil
#     gpu_available = True
# except ImportError:
#     gpu_available = False

# # Prometheus 메트릭 선언
# HTTP_TOTAL_REQUESTS = Counter('http_total_requests', 'Total HTTP requests count')
# HTTP_SUCCESS_REQUESTS = Counter('http_success_requests', 'Total HTTP successful requests count')
# HTTP_FAILED_REQUESTS = Counter('http_failed_requests', 'Total HTTP failed requests count')

# CPU_USAGE = Gauge('worker_cpu_usage', 'CPU usage of the application in percentage', ['worker'])
# MEMORY_USAGE = Gauge('worker_memory_usage', 'Memory usage of the application in MB', ['worker'])
# GPU_MEMORY_USAGE = Gauge('worker_gpu_memory_usage', 'GPU memory usage of the application in MB', ['worker'])

# class PrometheusMiddleware(BaseHTTPMiddleware):
#     async def dispatch(self, request: Request, call_next):
#         # 요청 카운트 증가
#         HTTP_TOTAL_REQUESTS.inc()

#         try:
#             response = await call_next(request)
#             status_code = response.status_code

#             if 200 <= status_code < 300:
#                 HTTP_SUCCESS_REQUESTS.inc()
#             else:
#                 HTTP_FAILED_REQUESTS.inc()

#         except Exception:
#             HTTP_FAILED_REQUESTS.inc()
#             raise

#         return response

# def record_worker_metrics(worker_id: str):
#     """각 워커의 리소스 사용량을 기록"""
#     while True:
#         # CPU 사용량
#         cpu_usage = psutil.cpu_percent()
#         CPU_USAGE.labels(worker=worker_id).set(cpu_usage)

#         # 메모리 사용량
#         memory_usage = psutil.virtual_memory().used / (1024 * 1024)
#         MEMORY_USAGE.labels(worker=worker_id).set(memory_usage)

#         # GPU 메모리 사용량 (GPU가 있을 경우)
#         if gpu_available:
#             gpus = GPUtil.getGPUs()
#             for gpu in gpus:
#                 GPU_MEMORY_USAGE.labels(worker=f"{worker_id}_gpu_{gpu.id}").set(gpu.memoryUsed)

#         time.sleep(5)  # 5초 간격으로 업데이트

# def setup_metrics(app):
#     """Prometheus 멀티프로세스 환경 설정 및 /metrics 엔드포인트 추가"""
#     app.add_middleware(PrometheusMiddleware)

#     @app.on_event("startup")
#     async def start_worker_metrics():
#         """각 워커별 메트릭 기록"""
#         import multiprocessing
#         worker_id = os.getenv("WORKER_ID", f"worker_{multiprocessing.current_process().pid}")
#         from threading import Thread
#         thread = Thread(target=record_worker_metrics, args=(worker_id,), daemon=True)
#         thread.start()

#     @app.get('/metrics')
#     async def metrics():
#         # 멀티프로세스 레지스트리 생성
#         registry = CollectorRegistry()
#         multiprocess.MultiProcessCollector(registry)
#         return Response(content=generate_latest(registry), media_type="text/plain")



# # ========================================
# # Method별 request metric 수집
# # ========================================
# from prometheus_client import Counter, CollectorRegistry, multiprocess, generate_latest
# from starlette.middleware.base import BaseHTTPMiddleware
# from starlette.requests import Request
# from starlette.responses import Response
# from time import time
# import os

# # Prometheus 메트릭 선언 (Counter는 멀티프로세스 모드에서 병합 가능)
# HTTP_TOTAL_REQUESTS = Counter('http_total_requests', 'Total HTTP requests count', ['method'])
# HTTP_SUCCESS_REQUESTS = Counter('http_success_requests', 'Total HTTP successful requests count', ['method'])
# HTTP_FAILED_REQUESTS = Counter('http_failed_requests', 'Total HTTP failed requests count', ['method'])

# class PrometheusMiddleware(BaseHTTPMiddleware):
#     async def dispatch(self, request: Request, call_next):
#         start_time = time()

#         # 요청 카운트 증가
#         HTTP_TOTAL_REQUESTS.labels(request.method).inc()

#         try:
#             response = await call_next(request)
#             status_code = response.status_code

#             # 상태 코드에 따라 성공/실패 카운트 증가
#             if 200 <= status_code < 300:
#                 HTTP_SUCCESS_REQUESTS.labels(request.method).inc()
#             else:
#                 HTTP_FAILED_REQUESTS.labels(request.method).inc()

#         except Exception:
#             # 실패한 요청
#             HTTP_FAILED_REQUESTS.labels(request.method).inc()
#             raise

#         finally:
#             process_time = time() - start_time
#             # 요청 처리 시간 등 추가 로직이 필요하면 여기서 처리 가능

#         return response


# def setup_metrics(app):
#     """Prometheus 멀티프로세스 환경 설정 및 /metrics 엔드포인트 추가"""
#     app.add_middleware(PrometheusMiddleware)

#     @app.get('/metrics')
#     async def metrics():
#         # 멀티프로세스 레지스트리 생성
#         registry = CollectorRegistry()
#         multiprocess.MultiProcessCollector(registry)

#         return Response(content=generate_latest(registry), media_type="text/plain")


# ========================================
# 기존 Metric 수집 코드
# ========================================


# # core/metrics.py
# from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest
# from starlette.middleware.base import BaseHTTPMiddleware
# from starlette.requests import Request
# from starlette.responses import Response
# from time import time
# import psutil
# import gc
# import os
# from app.core.config import settings

# # Prometheus 메트릭 선언
# #HTTP_TOTAL_REQUESTS = Counter('http_total_requests', 'Total HTTP requests count', ['method'])
# HTTP_TOTAL_REQUESTS = Counter('http_total_requests', 'Total HTTP requests count')
# HTTP_SUCCESS_COUNT = Counter('http_success_count', 'Overall HTTP success count')
# HTTP_FAILURE_COUNT = Counter('http_failure_count', 'Overall HTTP failure count')
# HTTP_SUCCESS_RATE = Gauge('http_success_rate', 'Overall HTTP success rate as a percentage')
# HTTP_FAILURE_RATE = Gauge('http_failure_rate', 'Overall HTTP failure rate as a percentage')



# API_REQUEST_COUNT = Counter('api_request_count', 'Count of requests per API', ['endpoint', 'method'])
# API_SUCCESS_COUNT = Counter('api_success_count', 'Count of successful requests per API', ['endpoint', 'method'])
# API_FAILURE_COUNT = Counter('api_failure_count', 'Count of failed requests per API', ['endpoint', 'method'])
# API_SUCCESS_RATE = Gauge('api_success_rate', 'Success rate of requests per API as a percentage', ['endpoint', 'method'])
# API_FAILURE_RATE = Gauge('api_failure_rate', 'Failure rate of requests per API as a percentage', ['endpoint', 'method'])

# RESPONSE_TIME = Histogram('response_time', 'API response time', ['method', 'endpoint'])
# CPU_USAGE = Gauge('cpu_usage', 'CPU usage of the application in percentage')
# MEMORY_USAGE = Gauge('memory_usage', 'Memory usage of the application in MB')
# DISK_USAGE = Gauge('disk_usage', 'Disk usage of the application in GB')
# REQUEST_LATENCY = Summary('request_latency', 'Time spent processing requests')
# GC_COLLECTIONS = Counter('gc_collections', 'Number of garbage collections')
# MAX_MEMORY_USAGE = Gauge('max_memory_usage', 'Maximum memory usage observed in MB')
# CONCURRENT_REQUESTS = Gauge('concurrent_requests', 'Concurrent requests in process')

# WORKER_CPU_USAGE = Gauge('worker_cpu_usage', 'CPU usage of the application in percentage', ['pid'])
# WORKER_MEMORY_USAGE = Gauge('worker_memory_usage', 'Memory usage of the application in MB', ['pid'])


# class PrometheusMiddleware(BaseHTTPMiddleware):
#     async def dispatch(self, request: Request, call_next):
#         # 동시 요청 수 증가
#         CONCURRENT_REQUESTS.inc()
#         start_time = time()

#         # 요청 및 API 카운트
#         HTTP_TOTAL_REQUESTS.inc()
#         API_REQUEST_COUNT.labels(request.url.path, request.method).inc()

#         try:
#             response = await call_next(request)
#             status_code = response.status_code

#             if 200 <= status_code < 300:
#                 API_SUCCESS_COUNT.labels(request.url.path, request.method).inc()
#                 HTTP_SUCCESS_COUNT.inc()
#             else:
#                 API_FAILURE_COUNT.labels(request.url.path, request.method).inc()
#         except Exception:
#             API_FAILURE_COUNT.labels(request.url.path, request.method).inc()
#             HTTP_FAILURE_COUNT.inc()
#             raise
#         finally:
#             process_time = time() - start_time
#             RESPONSE_TIME.labels(request.method, request.url.path).observe(process_time)
#             REQUEST_LATENCY.observe(process_time)

#             # 성공/실패 비율 계산
#             total_success = sum([c.samples[0].value if c.samples else 0 for c in API_SUCCESS_COUNT.collect()])
#             total_failure = sum([c.samples[0].value if c.samples else 0 for c in API_FAILURE_COUNT.collect()])
#             total_requests = total_success + total_failure

#             if total_requests > 0:
#                 HTTP_SUCCESS_RATE.set((total_success / total_requests) * 100)
#                 HTTP_FAILURE_RATE.set((total_failure / total_requests) * 100)
#             CONCURRENT_REQUESTS.dec()

#         return response

# def setup_metrics(app):
#     """FastAPI 애플리케이션에 Prometheus 메트릭 및 엔드포인트 추가"""
#     app.add_middleware(PrometheusMiddleware)

#     @app.get('/metrics')
#     async def metrics():

#         # Gunicorn worker들의 pid 읽어오기
#         if hasattr(settings,  'GUNICORN_PID_INFO_PATH') and os.path.exists(settings.GUNICORN_PID_INFO_PATH):
#             with open(settings.GUNICORN_PID_INFO_PATH, 'r') as f:
#                 pids = f.readlines()
#                 print(f"READ pid info from : {settings.GUNICORN_PID_INFO_PATH}\n -> {pids}")

#             for pid in pids:
#                 pid = pid.strip()
#                 if not pid: continue

#                 worker = psutil.Process(int(pid))
#                 try:
#                     cpu_usage = worker.cpu_percent(interval=1)
#                     memeory_usage = worker.memory_info().rss / (1024*1024)
                    
#                     WORKER_CPU_USAGE.labels(pid=worker.pid).set(cpu_usage)
#                     WORKER_MEMORY_USAGE.labels(pid=worker.pid).set(memeory_usage)
                    
#                     print(f"worker {pid} set cpu {cpu_usage}, memory usage {memeory_usage}")

#                 except Exception as e:
#                     print(f"Got Error while monitoring worekr : {e}")
#                     continue
        

        
#         CPU_USAGE.set(psutil.cpu_percent())
#         MEMORY_USAGE.set(psutil.virtual_memory().used / (1024 * 1024))
#         DISK_USAGE.set(psutil.disk_usage('/').used / (1024 ** 3))

#         # Garbage Collection 횟수 업데이트
#         gc.collect()
#         GC_COLLECTIONS.inc(gc.get_count()[0])

#         return Response(content=generate_latest(), media_type="text/plain")
