import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

import os
from functools import wraps
import uuid
from datetime import datetime, timezone, timedelta
from copy import deepcopy
import contextvars
from abc import ABC, abstractmethod
from typing import Optional, Any, AsyncGenerator
import asyncio
import threading
import json
import traceback  
from string import Template
import time

from fastapi.responses import StreamingResponse
from starlette.requests import Request as StarletteRequest

from app.atomic_services.retrieval_parallel_v4.models import Document


# ============================================================================
# Log ID 관리를 위한 Context Variable
# ============================================================================
user_id_var = contextvars.ContextVar("user_id", default=None)
request_id_var = contextvars.ContextVar("request_id", default=None)
service_id_var = contextvars.ContextVar("service_id", default=None)
processor_id_var = contextvars.ContextVar("processor_id", default=None)


class LoggingContext:
    @staticmethod
    def set_user_id(user_id):
        user_id_var.set(user_id)

    @staticmethod
    def get_user_id():
        return user_id_var.get()
    
    @staticmethod
    def set_request_id(request_id):
        request_id_var.set(request_id)

    @staticmethod
    def get_request_id():
        return request_id_var.get()

    @staticmethod
    def set_service_id(service_id):
        service_id_var.set(service_id)

    @staticmethod
    def get_service_id():
        return service_id_var.get()

    @staticmethod
    def set_processor_id(processor_id):
        processor_id_var.set(processor_id)

    @staticmethod
    def get_processor_id():
        return processor_id_var.get()


# ============================================================================
# Logging 설정
# ============================================================================
class CustomFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        now = datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        
    def format(self, record):
        record.asctime = self.formatTime(record)
        return super().format(record)


def setup_logging():
    """로깅 설정을 초기화합니다."""
    log_file = f"./app/execution_time_log/log_files/excution_time_{os.getpid()}.log"
    max_log_size = 1 * 128 * 1024
    backup_count = 2
    time_rotation_when = "m"
    time_rotation_interval = 10
    
    logger = logging.getLogger("excution_time_logger")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()
    
    time_handler = TimedRotatingFileHandler(
        log_file, 
        when=time_rotation_when, 
        interval=time_rotation_interval, 
        backupCount=backup_count
    )
    stream_handler = logging.StreamHandler()

    custom_formatter = CustomFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    time_handler.setFormatter(custom_formatter)
    stream_handler.setFormatter(custom_formatter)
    
    logger.addHandler(time_handler)
    logger.addHandler(stream_handler)

    return logger


# ============================================================================
# Logging Utility Functions
# ============================================================================
def get_kor_timestamp():
    """한국시간 기준 timestamp 반환"""
    timestamp = datetime.now(timezone(timedelta(hours=9)))
    return timestamp.strftime('%Y%m%d%H%M%S.%f')


def calculate_time_difference(timestamp1, timestamp2):
    """Timestamp 시간차 계산하여 반환"""
    format = '%Y%m%d%H%M%S.%f'
    dt1 = datetime.strptime(timestamp1, format)
    dt2 = datetime.strptime(timestamp2, format)
    return abs((dt2 - dt1).total_seconds())


def calculate_iso_time_difference(timestamp1, timestamp2):
    """ISO 형식 타임스탬프 간 시간차 계산"""
    dt1 = datetime.fromisoformat(timestamp1)
    dt2 = datetime.fromisoformat(timestamp2)
    time_difference = abs(dt2 - dt1)
    return time_difference.total_seconds()


def generate_log_id():
    """로그 Unique ID 생성"""
    return f"{uuid.uuid4()}"


async def save_log_to_es(index_name, log_data):
    """ElasticSearch에 로그를 저장합니다."""
    from app.main import app

    try:
        response = await app.state.log_es_client.index(index=index_name, body=log_data)
        log_id = response['_id']
        return log_id
    except Exception as e:
        print(f"Error saving log to {index_name} index ES: {e}")
        return None


def get_user_id(request_args, request_kwargs) -> str:
    """요청에서 user_id를 추출합니다."""
    request_data = request_args[0] if len(request_args) > 0 else None
    user_id = request_data.user_id if (hasattr(request_data, 'user_id') and request_data.user_id) else None 
    if user_id:
        return user_id

    user_id = request_kwargs['request'].user_id if 'request' in request_kwargs else None
    return user_id


# ============================================================================
# Logger Class : Standard, Streaming Logging기능 구현
# ============================================================================
class BaseLogger(ABC):
    """Non-streaming, streaming 서비스 로깅에 사용될 Logger 클래스"""
    
    def __init__(self, func_name: str, index_type: str, endpoint: Optional[str] = ''):
        self.endpoint = endpoint
        self.func_name = func_name
        self.index_type = index_type
        self.api_data = {}
        self.index_name = f"{index_type}-usage-logs"

    def initialize_logging(self, args: tuple, kwargs: dict) -> None:
        self.api_data = initialize_log_data()
        kwargs_copy = {k: v for k, v in kwargs.items() if not isinstance(v, (type(asyncio.Lock()), type(threading.Lock())))}
        kwargs_copy.update({'func_name': self.func_name})

        if self.index_type == "api":
            setup_api_log_data(self.api_data, self.endpoint, kwargs_copy, args)
            LoggingContext.set_request_id(self.api_data['request_id'])
        elif self.index_type == "service":
            setup_service_log_data(self.api_data, kwargs_copy)
            LoggingContext.set_service_id(self.api_data['service_id'])
        elif self.index_type == "processor":
            setup_processor_log_data(self.api_data, kwargs_copy)
            LoggingContext.set_processor_id(self.api_data['processor_id'])

    def serialize_data(self, response: Any) -> Any:
        """응답 데이터를 직렬화합니다."""
        if not response:
            return response
        
        if 'documents' in response and len(response['documents']) >= 1 and isinstance(response['documents'][0], Document):
            response.update({'documents': [doc.dict() for doc in response['documents']]})
            return response

        if ('formatted_query' in response and isinstance(response['formatted_query'], Template)):
            response['formatted_query'] = response['formatted_query'].template
        elif ('template' in response and isinstance(response['template'], Template)):
            response['template'] = response['template'].template
            return response
        return response

    @abstractmethod
    async def log_response(self, response: Any) -> Any:
        pass


class StandardLogger(BaseLogger):
    async def log_response(self, response: Any) -> Any:
        """Non-streaming 서비스 로깅하는 함수"""
        if self.index_type == "api":
            response = self.serialize_data(response)

        finalize_log_data(self.api_data, response)
        await save_log_to_es(index_name=self.index_name, log_data=self.api_data)
        return response

    async def log_streaming_response(self, response: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
        """스트리밍 응답을 로그와 동시에 처리하는 함수"""
        async for chunk in response:
            yield chunk
        
        finalize_log_data(self.api_data, {'status': 'completed'})
        await save_log_to_es(index_name=self.index_name, log_data=self.api_data)


# ============================================================================
# 데코레이터용 로깅 함수
# ============================================================================
def log_api_call(endpoint: Optional[str] = '', index_type: str = None):
    """Non-streaming 서비스 로깅을 위한 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            func_name = func.__qualname__
            logger = StandardLogger(func_name=func_name, index_type=index_type, endpoint=endpoint)
            logger.initialize_logging(args, kwargs)

            start_time = datetime.now(timezone.utc).isoformat()
            response = await func(*args, **kwargs)
            if not response:
                return None
                
            func_elpased_time = calculate_iso_time_difference(datetime.now(timezone.utc).isoformat(), start_time)
            response['func_elapsed_time'] = func_elpased_time
            asyncio.create_task(logger.log_response(response))
            return response
        return wrapper
    return decorator


def log_streaming_api_call(endpoint: Optional[str] = None, index_type: str = None):
    """Streaming 서비스 로깅을 위한 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            func_name = func.__qualname__
            logger = StandardLogger(func_name=func_name, index_type=index_type, endpoint=endpoint)
            logger.initialize_logging(args, kwargs)

            async_generator = await func(*args, **kwargs)
            return StreamingResponse(logger.log_streaming_response(async_generator), media_type="text/event-stream")
        return wrapper
    return decorator


def log_execution_time_async(func):
    """함수 실행시간을 로깅하는 비동기 데코레이터"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not LoggingContext.get_request_id():
            LoggingContext.set_request_id(str(uuid.uuid4()))
        if not LoggingContext.get_user_id():
            LoggingContext.set_user_id(get_user_id(args, kwargs))
        
        request_id = LoggingContext.get_request_id()
        user_id = LoggingContext.get_user_id()
        
        logger = logging.getLogger("excution_time_logger")
        logger.info(f"{func.__qualname__} started")
        start_time = time.time()
        
        result = await func(*args, **kwargs)
        elapsed_time = time.time() - start_time

        if not user_id:
            user_id = LoggingContext.get_user_id()
        
        if len(args) >= 1 and isinstance(args[0], StarletteRequest) and 'metrics' not in str(args[0].url):
            logger.info(f"[{user_id}] [{request_id}] {func.__qualname__} completed in {elapsed_time:.2f}s")
        else:
            logger.info(f"[{user_id}] [{request_id}] {func.__qualname__} completed in {elapsed_time:.2f}s")

        return result
    return wrapper


def log_execution_time(func):
    """함수 실행시간을 로깅하는 동기 데코레이터"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not LoggingContext.get_request_id():
            LoggingContext.set_request_id(str(uuid.uuid4()))
        if not LoggingContext.get_user_id():
            LoggingContext.set_user_id(get_user_id(args, kwargs))
        
        request_id = LoggingContext.get_request_id()
        user_id = LoggingContext.get_user_id()
        
        logger = logging.getLogger("excution_time_logger")
        logger.info(f"{func.__qualname__} started")
        start_time = time.time()
        
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        if not user_id:
            user_id = LoggingContext.get_user_id()
        
        if len(args) >= 1 and isinstance(args[0], StarletteRequest) and 'metrics' not in str(args[0].url):
            logger.info(f"[{user_id}] [{request_id}] {func.__qualname__} completed in {elapsed_time:.2f}s")
        else:
            logger.info(f"[{user_id}] [{request_id}] {func.__qualname__} completed in {elapsed_time:.2f}s")

        return result
    return wrapper


# ============================================================================
# Log Data Initialization and Finalization
# ============================================================================
def initialize_log_data():
    """로그 데이터를 초기화합니다."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_data": None,
        "output_data": None,
        "status": "processing",
        "elapsed_time": None,
    }


def setup_api_log_data(api_data, endpoint, kwargs, args):
    """API 로그 데이터를 설정합니다."""
    request_id = generate_log_id()
    kwargs.pop('service')
    kwargs.pop('func_name')

    request = dict(kwargs.pop('request', args[0] if args else None))
    api_data.update({
        'request_id': request_id,
        'session_id': request.pop('session_id'),
        'user_id': request.pop('user_id'),
        "endpoint": endpoint,
        "input_data": request,
    })


def setup_service_log_data(api_data, kwargs):
    """서비스 로그 데이터를 설정합니다."""
    service_id = generate_log_id()
    api_data.update({
        'service_id': service_id,
        'request_id': LoggingContext.get_request_id(),
        'service_name': kwargs.pop('func_name'),
        "input_data": kwargs,
    })


def setup_processor_log_data(api_data, kwargs):
    """프로세서 로그 데이터를 설정합니다."""
    processor_id = generate_log_id()
    api_data.update({
        'processor_id': processor_id,
        'service_id': LoggingContext.get_service_id(),
        'processor_name': kwargs.pop('func_name'),
        "input_data": kwargs,
    })


def finalize_log_data(api_data, response):
    """로그 데이터를 마무리합니다."""
    api_data["status"] = "completed"
    api_data["elapsed_time"] = response.pop("func_elapsed_time")
    api_data["output_data"] = response
    api_data["logging_elapsed_time"] = calculate_iso_time_difference(
        datetime.now(timezone.utc).isoformat(), 
        api_data["timestamp"]
    )


# ============================================================================
# Error Log 
# ============================================================================
def extract_function_name_from_traceback():
    """스택 트레이스에서 함수 이름을 추출합니다."""
    tb = traceback.format_exc()
    list_func_name = []
    for line in tb.splitlines():
        if "File" in line and "in " in line:
            list_func_name.append(line.split("in ")[-1].strip())
    if list_func_name: 
        return list_func_name[-1]
    return "Unknown"


async def log_error_to_es(function_name, error_message, input_text):
    """에러를 ElasticSearch에 로깅합니다."""
    from app.main import app
    log_entry = {
        "function": function_name,
        "error_message": error_message,
        "timestamp": datetime.now(),
        "input": input_text,
    }
    await app.state.log_es_client.index(index='api-error-logs', body=log_entry)


async def log_http_middleware_to_es(url, timestamp, elapsed_time):
    """HTTP 미들웨어 로그를 ElasticSearch에 저장합니다."""
    from app.main import app
    log_entry = {
        "url": url,
        "timestamp": timestamp,
        "elapsed_time": elapsed_time,
        "logging_elapsed_time": calculate_iso_time_difference(datetime.now(timezone.utc).isoformat(), timestamp),
    }
    await app.state.log_es_client.index(index="http-middleware-logs", body=log_entry)


async def log_simple_data_to_es(func_name, timestamp, elapsed_time, index_name):
    """간단한 데이터를 ElasticSearch에 로깅합니다."""
    from app.main import app
    
    if 'api' in index_name: 
        log_entry = {
            "endpoint": "api-" + func_name,
            "timestamp": timestamp,
            "elapsed_time": elapsed_time,
            "logging_elapsed_time": calculate_iso_time_difference(datetime.now(timezone.utc).isoformat(), timestamp),
        }
    elif 'service' in index_name: 
        log_entry = {
            "service_name": "service-" + func_name,
            "timestamp": timestamp,
            "elapsed_time": elapsed_time,
            "logging_elapsed_time": calculate_iso_time_difference(datetime.now(timezone.utc).isoformat(), timestamp),
        }
    elif 'processor' in index_name: 
        log_entry = {
            "processor_name": "processor-" + func_name,
            "timestamp": timestamp,
            "elapsed_time": elapsed_time,
            "logging_elapsed_time": calculate_iso_time_difference(datetime.now(timezone.utc).isoformat(), timestamp),
        }
    else:
        log_entry = {
            "func_name": func_name,
            "timestamp": timestamp,
            "elapsed_time": elapsed_time,
            "logging_elapsed_time": calculate_iso_time_difference(datetime.now(timezone.utc).isoformat(), timestamp),
        }
    await app.state.log_es_client.index(index=index_name, body=log_entry)


async def log_api_data(func_name: str, request_data: str, response_data: str):
    """API 데이터를 로깅합니다."""
    if not LoggingContext.get_request_id():
        LoggingContext.set_request_id(str(uuid.uuid4()))
    request_id = LoggingContext.get_request_id()
    
    logger = logging.getLogger("excution_time_logger")
    logger.info(f"[{request_id}] {func_name} request data : {request_data}")
    logger.info(f"[{request_id}] {func_name} response data : {response_data}")
