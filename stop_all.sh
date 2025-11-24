#!/bin/bash

# 로그 파일 경로
LOG_DIR="./monitoring_logs"

# PID 파일 경로
FASTAPI_PID_FILE="$LOG_DIR/fastapi_app.pid"
BATCHING_PID_FILE="$LOG_DIR/log_batching.pid"
METRICS_PID_FILE="$LOG_DIR/metric_batch.pid"
LOG_ROLLER_PID_FILE="$LOG_DIR/log_roller.pid"

# PID 파일에서 프로세스 ID 읽기
FASTAPI_PID=$(cat "$FASTAPI_PID_FILE")
BATCHING_PID=$(cat "$BATCHING_PID_FILE")
METRICS_PID=$(cat "$METRICS_PID_FILE")
LOG_ROLLER_PID=$(cat "$LOG_ROLLER_PID_FILE")

# 실행 중인 프로세스 종료
kill $FASTAPI_PID
kill $BATCHING_PID
kill $METRICS_PID
kill $LOG_ROLLER_PID

echo "All processes have been terminated."