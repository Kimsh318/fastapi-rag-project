# ML1 : 8000 => http://10.6.40.76:32050
# ML6 : 5000 => http://http://10.6.40.90:32018

# ==========================================================
# FastAPI 앱 실행
# ==========================================================

# ============= 프로메테우스 멀티프로세스 설정======================
# 앱 실행시마다 설정 필요
export PROMETHEUS_MULTIPROC_DIR=./prometheus/prometheus_multiproc_dir
rm -rf $PROMETHEUS_MULTIPROC_DIR
mkdir -p $PROMETHEUS_MULTIPROC_DIR

# ===============FastAPI APP 실행 설정============================
## Uvicorn 실행 : Prototype, Development 환경에서 주로 사용
# uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --workers 1

## Gunicorn 실행 : Production 환경에서 주로 사용
# 상수 값 정의
WORKERS=8
THREADS=6

# Gunicorn 실행 명령어
gunicorn -k uvicorn.workers.UvicornWorker app.main:app --keep-alive 10 --config ./gunicorn_log/gunicorn.conf.py --access-logfile ./gunicorn_log/access.log --error-logfile ./gunicorn_log/error.log --bind 0.0.0.0:8000 -w $WORKERS --threads $THREADS -t 60 --env WORKER_ID="worker_%(worker_num)s"