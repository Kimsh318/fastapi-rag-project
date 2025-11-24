#!/bin/bash

# LLM 서비스 프로젝트 구조 생성 스크립트

# 메인 프로젝트 디렉토리 이름
PROJECT_DIR="llm_service"

# 프로젝트 디렉토리 존재 여부 확인
if [ -d "$PROJECT_DIR" ]; then
  echo "디렉토리가 이미 존재합니다: $PROJECT_DIR"
  exit 1
fi

# 메인 프로젝트 디렉토리 생성
mkdir -p $PROJECT_DIR

# 프로젝트 디렉토리로 이동
cd $PROJECT_DIR

# app 디렉토리와 파일 생성
mkdir -p app/{core,services/{retrieval,generation,summary,autocomplete},utils}
touch app/main.py
touch app/core/{__init__.py,config.py,events.py}
touch app/utils/{__init__.py,streaming.py}
touch app/services/__init__.py

# 각 서비스별 디렉토리와 파일 생성
for service in retrieval generation summary autocomplete; do
  mkdir -p app/services/$service
  touch app/services/$service/{__init__.py,api.py,models.py,service.py}
done

# tests 디렉토리와 파일 생성
mkdir -p tests/{services,api}/{retrieval,generation,summary,autocomplete}
touch tests/__init__.py
touch tests/services/__init__.py
touch tests/api/__init__.py

# 각 테스트 서비스별 디렉토리와 파일 생성
for test_service in retrieval generation summary autocomplete; do
  touch tests/services/$test_service/{__init__.py,test_service.py}
  touch tests/api/$test_service/{__init__.py,test_api.py}
done

echo "LLM 서비스 프로젝트 구조가 성공적으로 생성되었습니다."
