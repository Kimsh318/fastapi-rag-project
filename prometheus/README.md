# Prometheus 모니터링 설정

이 폴더는 Prometheus를 사용하여 FastAPI 애플리케이션의 성능 및 리소스 사용량을 모니터링하기 위한 설정과 스크립트를 포함하고 있습니다.

## 폴더 구성

- **prometheus-checkpoint.yaml**: Prometheus의 설정 파일로, FastAPI 애플리케이션의 메트릭을 수집하기 위한 설정이 포함되어 있습니다.
  - `scrape_interval`: Prometheus가 데이터를 수집하는 주기 (15초로 설정).
  - `targets`: 모니터링할 FastAPI 서버의 IP와 포트.

## 파일 설명

1. **Prometheus 설정**: `prometheus.yaml` 파일을 사용하여 Prometheus 서버를 설정합니다. FastAPI 서버의 IP와 포트를 올바르게 설정해야 합니다.

2. **모니터링 스크립트 실행**: `parse_metrics.py`를 실행하여 CPU 및 메모리 사용량을 모니터링하고, 메트릭 데이터를 수집하여 Elasticsearch에 저장합니다.
    - `run_metric_batching.sh` 참고

### prometheus.yaml 설정 파일
- Prometheus metric 수집 주기, FastAPI 앱 주소 등을 지정 가능
- 만약, FastAPI 앱의 주소가 바뀐다면, 본 설정 파일에서도 변경해주어야 함

### parse_metrics.py 실행 파일
- python 명령어로 직접 실행해도 되나, `run_all.sh`, `run_metric_batching.sh` 명령어로 실행하는 것을 권장합니다.
- 파일의 핵심기능은 아래와 같습니다.
    1. **메트릭 데이터 수집**: Prometheus에서 FastAPI 서버의 메트릭 데이터를 주기적으로 수집합니다.(1분 등)
    2. **메트릭 데이터 파싱**: 수집된 메트릭 데이터를 파싱하여 필요한 메트릭만 추출합니다. (총 요청 수, 성공/실패 요청 수 등)
    3. **메트릭 계산**: 이전과 현재의 메트릭 데이터를 비교하여, 주기별 요청 수, 성공/실패 비율, 동시 요청 수 등을 계산합니다. 본 코드에서는 1분 주기로 비교합니다.
    4. **Elasticsearch에 저장**: 계산된 메트릭 데이터를 Elasticsearch에 인덱싱하여 저장합니다.
