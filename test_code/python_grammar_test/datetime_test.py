from datetime import datetime, timezone, timedelta
import time

def get_kor_timestamp():
    # 한국시간 기준으로 타임스탬프 생성
    timestamp = datetime.now(timezone(timedelta(hours=9))) # 한국시간 생성
    timestamp = timestamp.strftime('%Y%m%d%H%M%S')
    
    return timestamp

def time_delta_to_integer(time_diff:timedelta):
    total_seconds = delta.total_seconds()
    return total_seconds

def calculate_time_difference(timestamp1, timestamp2):
    # 문자열 타임스탬프를 datetime 객체로 변환
    format = '%Y%m%d%H%M%S'
    dt1 = datetime.strptime(timestamp1, format)
    dt2 = datetime.strptime(timestamp2, format)
    
    # 두 타임스탬프 간 시간차 계산
    time_difference = abs(dt2 - dt1)  # 절대값 사용으로 순서와 관계없이 양수 차이
    return time_difference

def calculate_iso_time_difference(timestamp1, timestamp2):
    # 문자열 타임스탬프를 datetime 객체로 변환
    dt1 = datetime.fromisoformat(timestamp1)
    dt2 = datetime.fromisoformat(timestamp2)
    
    # 두 타임스탬프 간 시간차 계산
    time_difference = abs(dt2 - dt1)  # 절대값 사용으로 순서와 관계없이 양수 차이
    return time_difference.total_seconds()

if __name__ == "__main__":
    # timedelta to 초
    delta = timedelta(days=1, hours=2, minutes=30)
    print(time_delta_to_integer(delta))

    # timestamp 차이 계산
    t1 = get_kor_timestamp()
    time.sleep(1)
    t2 = get_kor_timestamp()
    time_diff = calculate_time_difference(t2, t1)
    print(f"time diff type : {type(time_diff)}")
    print(f"diff : {time_diff}")


    t1 = datetime.now(timezone.utc).isoformat()
    time.sleep(1)
    t2 = datetime.now(timezone.utc).isoformat()
    time_diff = calculate_iso_time_difference(t2, t1)
    print(f"time diff type : {type(time_diff)}")
    print(f"diff : {time_diff}")