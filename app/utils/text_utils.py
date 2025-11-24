def clean_text(text):
    # 텍스트 정리 함수
    return text.strip().lower()

# 사용자의 질의 내 부적합 특수문자, 초성 등을 필터링
def refine_query(query: str):
    bad_pattern_hangul = '[ㄱ-ㅎㅏ-ㅣ]'
    bad_pattern_symbols = r'[!@#\$%\^&\*\(\)\-_=\+\[\]\{\};:\'",<>\./\?\\\|`~]+$' #만약 사용자 질의가 특수문자만으로 구성될 경우, 해당 특수문자를 제거
    query = re.sub(bad_pattern_hangul, '', query)
    if re.match(bad_pattern_symbols, query):
        query = re.sub(bad_pattern_symbols, '',  query)
    return query.strip()