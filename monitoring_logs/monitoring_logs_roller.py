import os
import shutil
from datetime import datetime

# 로그 파일 경로
log_files = ['./monitoring_logs/fastapi_app.log', './monitoring_logs/log_batching.log', './monitoring_logs/parse_metrics.log']
max_size = 100 * 1024 * 1024  # 100MB

def roll_log_file(log_file_path):
    # 로그 파일이 존재하는지 확인
    if os.path.exists(log_file_path):
        # 파일 크기 확인
        if os.path.getsize(log_file_path) >= max_size:
            # 롤링할 파일 이름 생성
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            rolled_file_path = f"{log_file_path}.{timestamp}"
            
            # 기존 로그 파일을 롤링
            shutil.move(log_file_path, rolled_file_path)
            print(f"Rolled log file to {rolled_file_path}")

            # 새로운 로그 파일 생성
            open(log_file_path, 'w').close()

def main():
    # 각 로그 파일에 대해 크기 확인 및 롤링
    for log_file in log_files:
        roll_log_file(log_file)

if __name__ == "__main__":
    main()