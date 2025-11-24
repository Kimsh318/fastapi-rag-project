# TODO: CMD Guvicorn만 사용하도록 수정
# OPTIMIZE : supervisord를 사용하여 프로세스 관리(시작, 재시작, 모니터링 등)

# 미리 만들어둔 이미지 사용
# FROM nvcr.io/nvidia/tritonserver:24.03-vllm-python-py3-RAG-FIN-autorag-5
# 베이스 이미지로 Python 3.9를 사용

# 우분투 22.04를 베이스 이미지로 사용
FROM ubuntu:22.04

# 패키지 목록 업데이트 및 Python 3.10 설치
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3-pip && \
    apt-get -y install curl\
                        net-tools\
                        nginx\
                        htop\
                        lsof

RUN pip install litellm==1.44.11 &&\
       pip install gunicorn &&\
       pip install celery &&\
       pip install redis==5.0.8 &&\
       pip install  llama-index-core==0.11.14 &&\
       pip install glances 
     

# 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    nginx \
    supervisor \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Grafana 7.5.4 설치
RUN wget -q -O - https://packages.grafana.com/gpg.key | apt-key add - \
    && echo "deb https://packages.grafana.com/oss/deb stable main" | tee -a /etc/apt/sources.list.d/grafana.list \
    && apt-get update \
    && apt-get install -y grafana=7.5.4 \
    && rm -rf /var/lib/apt/lists/*


# 필요한 패키지 설치
RUN apt-get update && apt-get install -y curl

WORKDIR /app

# Python 패키지 설치를 위한 requirements.txt 파일 복사
COPY requirements.txt .

# Python 패키지 설치 (ELL 포함)
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install -U ell-ai[all]

# 애플리케이션 코드 복사
COPY . .

# Nginx 설정
# COPY nginx.conf /etc/nginx/nginx.conf

# Supervisor 설정
# COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Rerank 모델 사용을 위한 라이브러리
RUN pip install FlagEmbedding==1.2.9


# 포트 설정 (Nginx: 80, Grafana: 3000)
EXPOSE 80 3000

# REMOVE : pycache 생성안되도록함
ENV PYTHONDONTWRITEBYTECODE=1




# 개발단계에 따라 아래 서비스 시작 방법 중 1개를 택
# 1. FastAPI만 실행
# CMD 명령을 수정하여 FastAPI 실행
# 1-1. uvicoron 사용
CMD ["uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]

# 1-2. gunicorn 사용 (w/ uvicorn worker)
#CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"]
#CMD ["gunicorn", "app.main:app", "--reload", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers", "4", "--timeout","300"]

# 2. Supervisor와 FastAPI 모두 실행
## supervisord.conf 파일 작성 필요
# CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]


