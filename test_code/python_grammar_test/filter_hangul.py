import re

def remove_hangul_jamo(text):
    # 자음 패턴
    consonant_pattern = re.compile(r'[\u1100-\u1112\u11A8-\u11C2\u3131-\u314E]')
    # 모음 패턴
    vowel_pattern = re.compile(r'[\u1161-\u1175\u314F-\u3163]')

    remove_consonant = consonant_pattern.sub('', text)
    remove_vowel = vowel_pattern.sub('',remove_consonant)
    # 정규 표현식을 사용하여 자모음을 빈 문자열로 대체
    return remove_vowel

# 예제 사용
text = "안녕ㅇ 하세요 ㅓㅓㅓ"
cleaned_text = remove_hangul_jamo(text)
print(cleaned_text)  # 출력: 안녕 하세요