
# 다기능 서비스 플랫폼: FastAPI를 이용한 확장 가능한 서비스 아키텍처 개발

## 소개

이 프로젝트는 **FastAPI**를 사용하여 **Retrieval Augmented Generation (RAG)** 기능을 제공하는 서버를 구현합니다. 초기에는 RAG 서비스를 제공하지만, 향후에는 사용자의 입력 텍스트를 요약하는 **요약(summary)** 서비스나 문장의 나머지를 완성해주는 **자동 완성(auto-complete)** 서비스 등 다양한 기능들을 추가할 수 있습니다. 이처럼 새로운 서비스를 쉽게 추가할 수 있는 확장 가능한 아키텍처를 갖추고 있습니다.

## 주요 기능

- **확장 가능한 서비스 아키텍처**: 새로운 서비스를 쉽게 추가할 수 있도록 설계되었습니다.
- **라우팅**: FastAPI 애플리케이션 및 다양한 서비스의 API 엔드포인트를 정의합니다.
- **RAG서비스 구현**: 검색과 생성을 결합하여 맥락에 맞는 응답을 생성합니다.(현재 Client에서는 Retrieval Endpoint만 사용 중)
- **모니터링**: Prometheus를 사용하여 애플리케이션 성능을, 로그를 파싱하여 사용패턴을 모니터링합니다.
- **데이터베이스 연동**: 문서 및 임베딩 데이터를 저장하기 위한 데이터베이스 연결을 구현합니다(FAISS, ElasticSearch).
- **API 문서화**: Swagger UI를 통해 자동으로 API 문서를 제공합니다. API의 사용방법을 확인하고, 손쉽게 테스트 해볼 수 있습니다.

## 시스템 아키텍처

### 주요 설계 철학
- **명확한 계층**: 역할이 명확히 구분된 계층 구조 (Controller → Service → Processor → Utils → External)
- **모듈성**: 독립적 개발 및 테스트가 가능한 구조로, 확장성과 유지보수성 향상
- **단일 책임 원칙**: 각 모듈은 하나의 책임만 담당 (`api`: 라우팅, `service`: 비즈니스 로직, `processor`: 세부 로직, `utils`: 공통 유틸)
- **유지보수성**: 명확한 역할 분담으로 변경의 영향을 최소화.
- **테스트 용이성**: 각 계층별로 독립적 테스트 가능
- **확장성 고려 (TBD)**: `atomic_services`로 기능 추가, `aggregated_services`로 기능 조합 (향후 개발 필요)

### 프로젝트 구조

```
project_root/
├── app/
│   ├── main.py                  # FastAPI 애플리케이션 및 라우터 정의
│   ├── core/                    # 설정 및 로깅
│   │   ├── __init__.py
│   │   ├── config.py            # 설정 관리
│   │   ├── logging.py           # 로그 관리
│   │   └── events.py            # 애플리케이션 시작/종료 이벤트 처리
│   │
│   ├── db/                      # 데이터베이스 관련 파일
│   │   │                        (Production 환경에서는 생략될 수 있음)
│   │   ├── __init__.py
│   │   └── database.py          # 데이터베이스 연결 및 모델 관리
│   │
│   ├── atomic_services/         # 개별 단일 서비스 API
│   │   ├── retrieval/           # 검색 API
│   │   │   ├── __init__.py
│   │   │   ├── api.py           # 검색 API 엔드포인트
│   │   │   ├── service.py       # 검색 비지니스 로직 구현
│   │   │   ├── processors.py    # 검색 세부로직 구현
│   │   │   ├── models.py        # 검색 관련 데이터 모델
│   │   │   └── helpers.py       # 검색 관련 유틸리티 함수
│   │   ├── ...                  # 생략된 디렉토리
│   │
│   ├── aggregated_services/     # 결합 서비스 API (TBD)
│   │   └── refine_and_search/   # 정제 + 검색 결합 API
│   │       ├── __init__.py
│   │       └── api.py           # 결합 API 엔드포인트
│   │
│   ├── utils/                   # 공통 유틸리티 함수
│   │   ├── __init__.py
│   │   ├── es_client.py         # Custom Elasticsearch 클라이언트
│   │   ├── logging_utils.py     # 로깅 관련 유틸리티
│   │   ├── streaming.py         # 스트리밍 관련 유틸리티
│   │   ├── text_utils.py        # 텍스트 처리 유틸리티
│   │   └── validation_utils.py  # 검증 관련 유틸리티
│
├── gunicorn_log/                # Gunicorn 관련 로그 및 설정
│   ├── gunicorn.conf.py         # Gunicorn 설정 파일
│   ├── pid_file.txt             # Gunicorn worker들의 PID 파일
│   ├── error.log                # 에러 로그
│   └── access.log               # 접근 로그
│
├── prometheus/                  # Prometheus 관련 설정 및 스크립트
│   ├── prometheus.yml           # Prometheus 설정 파일
│   └── parse_metrics.ipynb      # 메트릭 파싱 스크립트(필요시, 수동실행)
│
├── monitoring_logs/             # 모니터링 로그 및 스크립트
│   ├── fastapi_app.log          # FastAPI 앱 로그
│   ├── log_batching.log         # 로그 배칭 로그
│   ├── parse_metrics.log        # 메트릭 파싱 로그
│   ├── monitoring_logs_roller.py # 로그 롤링 스크립트(로그파일이 일정 크기 이상되면 초기화)
│   └── stop_process.py          # fastapi app 중지 스크립트
│
└── __init__.py
```

### 레이어 구조

```plaintext
[요청 흐름]
Client
  ↓
FastAPI Router (api.py, models.py)
  ↓
Service Layer (service.py)
  ↓
Processor Layer (processors.py)
  ↓
Utils/Helpers (helpers.py)
  ↓
External (DB etc..)
```
- **`api.py`**: FastAPI 엔드포인트를 정의합니다. 클라이언트 요청을 받고, 응답을 반환하며, 요청의 유효성을 검사하고 오류를 처리합니다.

- **`service.py`**: 주요 비즈니스 로직을 구현합니다. API 레이어로부터 받은 요청을 처리하고, 프로세서 레이어의 기능을 호출하여 필요한 작업을 수행합니다.

- **`processors.py`**: 세부 처리 로직을 구현합니다. 데이터 처리, 변환, 외부 API 호출 등의 구체적인 작업이 이곳에서 이루어집니다.

- **`models.py`**: Pydantic 모델을 정의합니다. 요청 및 응답 데이터의 구조를 명확히 하여 데이터 유효성을 보장합니다.

- **`helpers.py`**: 유틸리티 함수들을 포함합니다. 공통적으로 사용되는 함수들을 모아 코드의 재사용성을 높입니다.



## 사용방법
### 앱 실행, 종료
1. 전체 앱 실행 및 종료
   ```
   # 실행
   sh run_all.sh

   # 종료
   sh stop_all.sh
   ```

2. 개별 실행
   ```
   # FastAPI APP 실행 : Gunicorn worker, Thread 수 설정
   sh run_fastapi_app.sh

   # Log 수집 실행 : 로그 파일들을 파싱하여 ElasticSearch에 저장
   sh run_log_batching.sh

   # Metric 수집 실행 : 주기적으로 FastAPI앱의 Metric 수집(10분 주기 등)
   sh run_metric_batching.sh
   ```

3. 실행 결과 모니터링   
   - 프로세스들이 정상적으로 실행 중인지 확인하기 위해, `/monitoring_logs`의 log파일 3개를 실시간 모니터링(fastapi_app.log, log_batching.log, pasre_metrics.log)

   ```bash
   # fastapi 앱 로그 모니터링
   tail -n 20 -f ./monitoring_logs/fastapi_app.log

   # log batching 모니터링
   tail -n 20 -f ./monitoring_logs/log_batching.log

   # Metric 파싱 모니터링
   tail -n 20 -f ./monitoring_logs/parse_metrics.log
   ```


<br>

> ⚠️ **실행, 종료 시 주의사항**
> 
> 실행, 종료 port 번호 일치 필요(default 8000번)
> - 실행 port 번호:  "run_fastapi_app.sh"에서 사용하는 port 번호
> - 종료 port 번호:  "/app/monitoring_logs/stop_process.py"에서 사용하는 port 번호


### API 입출력 형식 확인 및 테스트
- FastAPI Swagger 문서가 자동 생성하여, Client 개발자가 API를 쉽게 이해할 수 있도록 지원
   - 접속 방법 : '{fastapi app 경로}/docs'
   - API 테스트 : 'Try it out' 기능을 통해 직접 API 요청을 테스트 가능


## 개발 가이드

### 서비스 추가
새로운 서비스를 추가하려면 atomic_services 혹은 aggregated_services 디렉토리에 새로운 디렉토리를 생성하고, 그 안에 api.py, models.py, service.py, processors.py, helpers.py, __init__.py 파일을 작성

1. **서비스 파일 생성**
```
sh create_endpoint.sh {endpoint_name}
```
 - atomic_services 하위에 {endpoint_name} 디렉토리 생성
 - 생성된 디렉토리 하위에는 api.py, models.py, service.py, processors.py, helpers.py, __init__.py 파일을 생성됨
 - 각 파일에는 샘플코드가 작성되어 있으며, 이를 참고하여 서비스 개발 작업 가능.

2. **서비스 개발**
 - 엔드포인트 개발 : 서비스 요청, 응답 모델 정의(models.py), 엔드포인트 정의 및 요청, 응답 방법 정의(api.py)
 - 서비스 로직 개발 : 비지니스 로직(service.py)과 그 하위 프로세스들(processors.py)을 구분하여 개발 
 - 설정 파일 업데이트 : 새로운 서비스에 필요한 설정값들을 추가(/core/config.py)

3. **라우터 등록**
   - `app/main.py` 또는 해당하는 곳에서 새로운 라우터를 애플리케이션에 등록합니다.

4. **로그 파싱 테스트**
   - 개발한 서비스의 로그가 정상적으로 파싱되는지 확인 필요합니다.
      1. "/app/execution_time_log/"에 위치한 로그 파일들 중 서비스 관련 로그 수기 추출
      2. "/app/execution_time_log/extract_data_from_log.py"의 "extract_log_data" 함수에 입력
      3. 함수 출력 결과 확인
   - 만약 파싱이 안된다면, "/app/execution_time_log/extract_data_from_log.py" 파일에서 파싱 규칙 추가해야 합니다.

<br>

> ⚠️ **개발 시 주의사항**
> 
> 1. **비지니스 로직과 세부구현의 분리** :  비즈니스 로직과 구현 세부사항을 명확히 분리하여 작성합니다. `service.py`와 같은 파일에서는 비즈니스 로직을 구현하며, `APP_EVN`와 무관하게 동작해야 합니다. `processors.py`와 같은 파일에서는 구체적인 동작을 구현합니다. 특히 `APP_ENV`에 맞는 클래스, 기능들을 구현해야 합니다.  코드의 가독성과 유지보수성을 높입니다.
> 2. **테스트 코드 작성**: 새로운 기능을 추가할 때마다 "/test_code" 하위에 테스트 코드를 작성하여 기능이 올바르게 작동하는지 확인합니다.
> 3. **API 문서화**: Swagger 문서를 통해 API를 직관적으로 이해할 수 있도록 주석, 코드 작성 필요합니다.("Swagger 문서 작성 방법" 참조)
> 4. **에러 처리**: 예상치 못한 상황에 대비하여 적절한 에러 처리를 구현합니다. `api.py`에서는 client의 동작방법을 감안하여 발생한 에러에 따라 적절한 HTTP 에러 코드를 전달해야 합니다. 예를 들어, 잘못된 요청에는 400 Bad Request, 인증 실패에는 401 Unauthorized, 서버 오류에는 500 Internal Server Error 등을 사용합니다.
> 5. **주석 작성**: 복잡한 로직이나 중요한 부분에는 주석을 추가하여 코드 이해 도움을 줍니다.


### 개발 환경에 따른 프로세서 구현

`processors.py` 파일에서는 `prototype`, `development`, `production` 환경에 따라 서로 다른 프로세서를 구현하고 있습니다. 이는 각 개발 단계에서 요구되는 기능과 성능을 독립적으로 관리하고 배포할 수 있도록 하기 위함입니다.
- ~~**Prototype 환경**~~ (Deprecated): 
  - 
  - **목적**: ~~단순 기능 테스트를 통해 빠르게 구현 가능성을 탐색~~
  - **구현 방식**: 
    - ~~간단한 로직과 설정을 사용하여, 복잡한 기능 구현 없이 샘플 입력과 출력을 테스트~~
    - ~~단순 기능 테스트를 통해 아이디어를 빠르게 검증하고, 피드백을 수집~~
    - ~~주로 개발 초기 단계에서 사용되며, 빠른 반복을 통해 기능의 방향성 결정~~

- **Development 환경**: 
  - **목적**: 배포하고자 하는 기능을 테스트 및 디버깅. 배포 준비
  - **구현 방식**: 
    - 연구 및 개발 단계에서 검증된 로직을 서비스 앱으로 이관
    - 비동기 처리를 통해 성능을 최적화하고, 다양한 시나리오에서 기능을 검증
    - 실제 배포 전, 모든 기능이 예상대로 작동하는지 확인

- **Production 환경**: 
  - **목적**: 서버자원(GPU 메모리 등)에 적합한 설정값 세팅하여, 실제 운영 환경에서의 기능을 제공.
  - **구현 방식**: 
    - 운영 환경에 적합한 워커 조정, 서비스간 연동(DB, app 연동 등) 및 최적화 작업(비동기 처리 등) 수행하여, 다양한 요청을 안정적으로 처리
    - 지속적인 모니터링을 통해 시스템의 안정성과 성능을 유지하고 개선

### Swagger 문서 작성 방법
1. **Pydantic 모델 사용**: 요청 및 응답 데이터 모델을 정의할 때 Pydantic 모델을 사용하고, 각 필드에 `description`과 `example`을 추가하여 Swagger 문서에 반영되도록 합니다. 이는 각 필드의 목적과 예제를 명확히 설명합니다.(models.py)

2. **Docstring 사용**: FastAPI의 경로 함수에 docstring을 추가하여 함수의 목적, 입력 매개변수, 반환 값 등에 대한 설명을 제공합니다. 이 설명은 Swagger 문서에 반영되어 API 사용자가 해당 엔드포인트의 기능을 이해하는 데 도움을 줍니다.(api.py)

3. **Request 및 Response 모델 지정**: 
   - **Request 모델**: `Pydantic` 모델을 사용하여 요청 데이터의 구조를 명확히 정의합니다. 각 필드에 `description`과 `example`을 추가하여 Swagger 문서에 반영되도록 합니다. 이는 각 필드의 목적과 예제를 명확히 설명합니다.(models.py, api.py)
   - **Response 모델**: `response_model` 매개변수를 사용하여 응답 데이터의 구조를 명확히 정의합니다. 이는 Swagger 문서에 응답 형식을 명확히 나타냅니다.(models.py, api.py)

4. **상세한 예제 제공**: `example` 매개변수를 사용하여 요청 및 응답의 예제를 제공하면, 사용자가 API를 더 쉽게 이해할 수 있습니다.(models.py)

## 운영 가이드

운영 환경에서 시스템의 안정성과 성능을 유지하기 위해 다양한 모니터링 및 로그 수집 메커니즘을 사용합니다.

### 로그 수집

- **목적**: 시스템의 상태와 성능을 실시간으로 모니터링하고, 문제 발생 시 신속하게 대응하기 위함입니다.
- **구현**:
  - **로그 출력**: 
    - `app/main.py`에서 설정된 로깅 미들웨어를 통해 모든 HTTP 요청과 응답이 로깅됩니다. 
    - 이 미들웨어는 `app/utils/logging_utils.py`의 함수를 사용하여 로그를 포맷하고 출력합니다.
    - **로그 수집**: 
      - **애플리케이션 로그**: 로그는 기본적으로 `monitoring_logs/` 디렉토리에 저장됩니다. 이 디렉토리에는 `fastapi_app.log`, `log_batching.log`, `parse_metrics.log` 로그 파일이 저장됩니다.
        - fastapi_app.log : HTTP 요청 및 응답, 오류 메시지, 경고, 정보성 메시지 등 애플리케이션의 동작과 관련된 정보 제공
        - log_batching.log : 로그 파일을 주기적으로 수집하고, 이를 Elasticsearch와 같은 데이터 저장소에 저장하는 과정에서 발생하는 이벤트를 기록
        - parse_metrices.log : 모니터링 도구(Prometheus)에서 수집한 메트릭 데이터를 파싱한 이벤트를 기록. 
      - **Gunicorn 로그**: Gunicorn과 관련된 로그는 `gunicorn_log/` 디렉토리에 별도로 저장됩니다.
        - access.log : 클라이언트 요청에 대한 정보(요청 메서드, URL, HTTP 상태코드 등)를 기록
        - error.log : Gunicorn이 발생시킨 오류 메시지(GPU OOM 등)를 기록
    - **로그 파싱**: `extract_data_from_log.py` 스크립트를 사용하여 로그 데이터를 파싱하고, 필요한 정보를 추출합니다.
    - **백업 및 보관**: 로그 유실을 방지하기 위해 `log_backup` 디렉토리에 로그를 백업합니다.

### 메트릭 수집 (Prometheus)

- **목적**: 시스템의 성능 지표를 수집하고 분석하여, 운영 상태를 지속적으로 개선합니다.
- **구현**:
  - **메트릭 데이터 수집**: Prometheus는 FastAPI 서버의 메트릭 데이터를 주기적으로 수집합니다.
  - **메트릭 데이터 파싱 및 계산**: 수집된 메트릭 데이터를 파싱하여, 총 요청 수, 성공/실패 요청 수 등을 계산합니다.
  - **Elasticsearch 저장**: 계산된 메트릭 데이터를 Elasticsearch에 저장하여, 장기적인 분석과 모니터링에 활용합니다.

이러한 운영 가이드는 시스템의 안정성과 성능을 유지하는 데 중요한 역할을 하며, 지속적인 모니터링과 피드백을 통해 시스템을 개선할 수 있도록 지원합니다.



## 향후 개선사항 및 추가 작업

### 1) 향후 확장성 고려한, atomic service와 aggregated service의 구분
프로젝트의 서비스는 **atomic_services**와 **aggregated_services** 디렉토리로 구분됩니다. 이 구조는 단일 기능의 API와 복합 기능의 API를 명확히 구분하여 서비스의 재사용성과 관리 효율성을 높입니다.
- **atomic_services**: 각 단일 서비스가 독립적으로 수행할 수 있는 개별 기능을 제공합니다. 예를 들어, `queryRefine`은 질의 정제 작업을, `retrieval`은 검색 작업을 단독으로 수행할 수 있습니다. 이 구조는 개별 서비스가 독립적으로 테스트 및 유지보수가 가능하도록 하며, 재사용성을 극대화합니다.
- **aggregated_services**: 여러 **atomic_services**를 결합하여 복합 작업을 수행하는 API입니다. 예를 들어, `refine_and_search` API는 `queryRefine`과 `retrieval` 서비스를 결합하여 정제된 쿼리를 검색에 활용하는 복합 워크플로우를 제공합니다. 이 구조는 복합적인 기능을 클라이언트에게 일관된 API로 제공할 수 있는 장점이 있습니다.

### 포맷

- **[우선순위]** 작업 제목: 작업의 간단한 설명

### 작업내용

- **[높음]** 코드 리팩토링: retrieval의 api.py 파일 리팩토링
- **[높음]** 코드 리팩토링: aggregated_services와 atomic_services 분리 구현
- **[보통]** Swagger 상세 문서 내용 작성: API 설명, request, response 형식의 이해를 도울 수 있도록 Swagger 문서 내용 개선
- **[보통]** API 버전 관리: API 버전 관리 방안 마련


<br>
<br>


------

<br>

# 개발 환경설정
- 외부망에서 Docker Image를 활용한 개발환경 구축 과정 소개
- IDE(VSCode 등)에서 도커 컨테이너를 연결하여 개발하는 것을 추천함("컨테이너 접속" 참고)

## Docker Image 기반 개발환경 구축
1. **Docker Image 빌드 및 컨테이너 실행**
    
   1-1\) 이미지 빌드
   ```
    docker build -t langchain_fastapi_server_image .
   ```
      - 이미지 빌드 실패하고, 완전히 처음부터 다시 빌드하고 싶다면 "--no-cache" 옵션을 추가하여, 캐시를 사용하지 않고 이미지 빌드
 
   1-2\) 컨테이너 실행
   ```
    ## -p : 8000번 포트를 통해 접속
    ## -d : 백그라운드 실행
    ## -it : 터미널 입출력 환경 제공, 컨테이너 내부에서 명령어 실행 가능
    ## -v : 로컬 프로젝트 경로와 컨테이너 경로 연결. 컨테이너에서 작업한 내역이 로컬에 바로 반영됨.
    docker run -d -it -p 8000:8000 -v ~/CursorAI_Project/langchain_fastapi_server_v4:/app --name langchain_fastapi_container langchain_fastapi_server_image 

   ## docker network(app-network)를 통해 다른 도커 컨테이너와 통신하려면 아래 명령어로 실행
   docker run -d -it --network app-network -p 8000:8000 -v ~/CursorAI_Project/langchain_fastapi_server_v3:/app --name langchain_fastapi_container langchain_fastapi_server_image 
   ```
   1-3\) 컨테이너 목록 확인
   ```
   docker ps
   ```

2. **컨테이너 접속**
   
    2-1\) IDE 활용(Cursor AI 혹은 VScode)
    - Cursor AI 혹은 VScode 환경에서 컨테이너 내부에 접속하여 파일 확인 및 명령어 실행 가능
    - extension으로 remote development 설치 및 사용
    - 좌측 탭에서 'Remote Explorer' 검색 후 클릭
    - 우측 상단 '+' 버튼 클릭 후 'Container' 선택
    - 컨테이너 접속 후 파일 확인 및 명령어 실행 가능

   2-2\) 터미널 활용
      ```bash
      docker exec -it langchain_fastapi_container /bin/bash # 컨테이너 접속
      exit # 컨테이너 탈출
      ```

3. **컨테이너 로그 확인**
   ```
   docker logs --tail 50 <컨테이너 이름> # 마지막 N개 로그만 보기
   docker logs -f <컨테이너 이름> # 실시간 로그 확인
   docker logs --details <컨테이너 이름> # 자세한 로그 확인
   ```

4. (Optional) **Docker 네트워크 구성**
   - 다른 도커 컨테이너를 한 네트워크에서 연결하여 통신해야할 때 사용
   
   4-1\) Docker network 생성
   ```
   docker network create my-network
   ```
   4-2\) Docker 컨테이너들을 하나의 Docker 네트워크에 추가
   ```
   docker network connect my-network my-container
   ```
   4-3\) 네트워크 연결 확인
   ```
   docker network inspect my-network
   ```
   4-4\) 컨테이너별 IP 업데이트
   - request 보낼 IP에 변동이 없는지, 컨테이너마다 도커 네트워크 내 할당된 IP를 확인 및 수정


5. 기타사항 
   ** Ollama 실행 **
   - 외부망에서 개발 시, Ollama 이용해서 LLM 모델 서빙
   ```
   docker pull ollama/ollama   # 도커이미지 가져오기
   docker run -d -it --network app-network -p 11434:11434 --name ollama_container ollama/ollama # Ollama 컨테이너 실행
   docker exec -it ollama_container ollama run gemma2:2b # 모델 다운로드 및 실행
   ```

