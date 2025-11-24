#!/bin/bash
# 실행 방법 : sh  ./create_endpoint.sh {endpoint_name}

# 디렉토리, 파일 경로 설정
BASE_PATH="/app/app/atomic_services"
TEMPLATES_PATH="$BASE_PATH/templates"

ENDPOINT_NAME=$1 # 첫 번째 인자로 엔드포인트 디렉토리 이름을 받습니다.
ENDPOINT_PATH=$BASE_PATH/$ENDPOINT_NAME

# 인자가 제공되지 않았을 경우, 오류 메시지 출력
if [ -z "$ENDPOINT_NAME" ]; then
  echo "엔드포인트 디렉토리 이름을 입력하세요."
  exit 1
fi

# 동일한 이름의 디렉토리가 이미 존재하는지 확인
if [ -d "$ENDPOINT_PATH" ]; then
  echo "이미 동일한 이름의 디렉토리가 존재합니다: $ENDPOINT_PATH"
  exit 1
fi

# BASE_PATH 하위에 엔드포인트 디렉토리 생성
mkdir -p "$ENDPOINT_PATH"

# 이미 작성된 template 파일들을 복사
cp "$TEMPLATES_PATH/api.py" "$ENDPOINT_PATH/api.py"
cp "$TEMPLATES_PATH/service.py" "$ENDPOINT_PATH/service.py"
cp "$TEMPLATES_PATH/processors.py" "$ENDPOINT_PATH/processors.py"
cp "$TEMPLATES_PATH/models.py" "$ENDPOINT_PATH/models.py"
cp "$TEMPLATES_PATH/helpers.py" "$ENDPOINT_PATH/helpers.py"
cp "$TEMPLATES_PATH/__init__.py" "$ENDPOINT_PATH/__init__.py"

# 생성 완료 메시지와 경로 출력
echo "폴더와 파일이 성공적으로 생성되었습니다."
echo "생성된 경로: $ENDPOINT_PATH"