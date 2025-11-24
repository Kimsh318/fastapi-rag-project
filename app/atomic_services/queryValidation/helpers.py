from transformers import AutoTokenizer

def get_tokenizer_info(settings:object)->dict:
    return {
        'tokenizer_path': settings.LLM_TOKENIZER_PATH,
        'hf_token': settings.HF_TOKEN,
    }

def get_validation_criteria(settings:dict)->dict:
    return {
        'max_character_threshold': settings.MAX_CHRACTER_LEN,
        'min_character_threshold': settings.MIN_CHRACTER_LEN,
    }    


# Tokenizer 불러오기 유틸리티 함수
def get_tokenizer(tokenizer_path:str, hf_token=''):
    # FIXME : 반입 후, 내부망에 설치된 tokenizer 경로 지정 필요
    # FIXME : 반입 후, huggingface 로그인을 위해 입력한 token 인자 제거 필요
    if hf_token:
        return AutoTokenizer.from_pretrained(tokenizer_path, use_auth_token=hf_token)
    return AutoTokenizer.from_pretrained(tokenizer_path)