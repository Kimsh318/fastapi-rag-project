#!/bin/bash

# 현재 작업 디렉토리 경로
project_root=$(pwd)

# app 디렉토리 경로
app_path="$project_root"

# app 디렉토리가 PYTHONPATH에 없으면 추가
echo $PYTHONPATH | grep -q -E "(^|:)$app_path(:|$)" || export PYTHONPATH="$app_path:$PYTHONPATH"

# python 명령어의 출력을 $METRICS_LOG 파일에 저장
python3 -m app.execution_time_log.extract_data_from_log >> "$METRICS_LOG" 2>&1