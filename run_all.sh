#!/bin/bash

# 로그 파일 경로
LOG_DIR="./monitoring_logs"
mkdir -p $LOG_DIR

FASTAPI_LOG="$LOG_DIR/fastapi_app.log"
BATCHING_LOG="$LOG_DIR/log_batching.log"
METRICS_LOG="$LOG_DIR/metrics_batch.log"

# PID 파일 경로
FASTAPI_PID_FILE="$LOG_DIR/fastapi_app.pid"
BATCHING_PID_FILE="$LOG_DIR/log_batching.pid"
METRICS_PID_FILE="$LOG_DIR/metric_batch.pid"
LOG_ROLLER_PID_FILE="$LOG_DIR/log_roller.pid"

# 로그 파일 초기화
echo "FastAPI app execution started at $(date)" > $FASTAPI_LOG
echo "Log batching execution started at $(date)" > $BATCHING_LOG
echo "Metrics batching execution started at $(date)" > $METRICS_LOG

# run_fastapi_app.sh 실행
bash run_fastapi_app.sh >> $FASTAPI_LOG 2>&1 &
FASTAPI_PID=$!
echo $FASTAPI_PID > "$FASTAPI_PID_FILE"

# run_metric_batching.sh 실행
METRICS_LOG="$METRICS_LOG" bash run_metric_batching.sh &
METRICS_PID=$!
echo $METRICS_PID > "$METRICS_PID_FILE"

# run_log_batching.sh 실행
BATCHING_LOG="$BATCHING_LOG" bash run_log_batching.sh &
BATCHING_PID=$!
echo $BATCHING_PID > "$BATCHING_PID_FILE"

# # run_log_batching.sh 실행
# bash run_log_batching.sh >> $BATCHING_LOG 2>&1 &
# BATCHING_PID=$!
# echo $BATCHING_PID > "$BATCHING_PID_FILE"

# # run_metric_batch.sh 실행
# bash run_metric_batch.sh >> $METRICS_LOG 2>&1 &
# METRICS_PID=$!
# echo $METRICS_PID > "$METRICS_PID_FILE"

# 로그 파일 크기 주기적으로 확인
while true; do
    python3 log_roller.py
    sleep 600  # 10분 대기
done &
LOG_ROLLER_PID=$!
echo $LOG_ROLLER_PID > "$LOG_ROLLER_PID_FILE"

# 모든 프로세스가 종료될 때까지 대기
wait $FASTAPI_PID
if [ $? -ne 0 ]; then
    echo "Error: run_fastapi_app.sh failed" >> $FASTAPI_LOG
fi

wait $BATCHING_PID
if [ $? -ne 0 ]; then
    echo "Error: run_log_batching.sh failed" >> $BATCHING_LOG
fi

wait $METRICS_PID
if [ $? -ne 0 ]; then
    echo "Error: run_metric_batch.sh failed" >> $METRICS_LOG
fi

echo "All scripts executed successfully at $(date)"