# Monitoring Logs

`monitoring_logs` 폴더는 시스템의 로그 파일을 관리하고 모니터링하는 데 사용됩니다. 이 폴더에는 `run_all.sh` 스크립트 실행 시 시작되는 각 프로세스의 로그 파일과 이를 관리하는 파이썬 스크립트가 포함되어 있습니다.

## 로그 파일
1. fastapi_app.log: FastAPI 애플리케이션의 실행 로그를 기록
2. log_batching.log: 로그 배칭 프로세스의 실행 로그를 기록(`/app/execution_time_log` 참고)
3. parse_metrics.log: 메트릭 파싱 프로세스의 실행 로그를 기록합니다.(`/prometheus` 참고)


## 파이썬 스크립트
1. monitoring_logs_roller.py
    - 로그 파일의 크기를 모니터링하고, 지정된 크기(100MB)를 초과할 경우 로그 파일을 롤링합니다. 
    - 로그파일들이 큰 용량을 차지하는 경우를 방지합니다.

2. stop_process.py
    - FastAPI 앱 사용중인 포트를 확실하게 Idle 상태로 바꾸기 위해, 해당 포트를 사용하는 프로세스를 찾아 종료합니다.

## 사용 방법
- 직접 파이썬 스크립트들을 직접 실행하기보다는 `run_all.sh`로 실행하는 것을 권장합니다.
- 터미널에서 아래 명령어들을 실행하여, 각 로그 파일들을 실시간으로 모니터링 합니다.

```bash
# fastapi 앱 로그 모니터링
tail -n 20 -f fastapi_app.log

# log batching 모니터링
tail -n 20 -f log_batching.log

# Metric 파싱 모니터링
tail -n 20 -f parse_metrics.log
```

