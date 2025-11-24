# app/services/queryValidate/processors.py
import re

from app.atomic_services.queryValidation.helpers import get_tokenizer


def get_query_processor(tokenizer_info, validation_criteria, app_env):
    """Query Processor를 선택하는 함수"""
    if app_env == "prototype":
        return PrototypeQueryProcessor(tokenizer_info, validation_criteria)
    elif app_env == "development":
        return DevelopmentQueryProcessor(tokenizer_info, validation_criteria)
    elif app_env == "production":
        return ProductionQueryProcessor(tokenizer_info, validation_criteria)    
    else:
        raise ValueError("지원하지 않는 환경입니다.")


class BaseQueryProcessor:
    """Base Query Processor"""
    
    # 패턴을 클래스 변수로 정의 (모든 인스턴스가 공유)
    BAD_PATTERN_ENGLISH = '^[A-Za-z\s]+$'
    BAD_PATTERN_HANGUL = '[ㄱ-ㅎㅏ-ㅣ\s]+'
    BAD_PATTERN_SYMBOLS = r'[!@#\$%\^&\*\(\)\-_=\+\[\]\{\};:\'",<>\./\?\\\|`~]+$'
    
    def __init__(self, tokenizer_info: dict, validation_criteria: dict):
        self.tokenizer = get_tokenizer(tokenizer_info['tokenizer_path'], tokenizer_info['hf_token'])
        self.max_chars_thr = validation_criteria['max_character_threshold']
        self.min_chars_thr = validation_criteria['min_character_threshold']

    async def calculate_token_count(self, query: str) -> dict:
        """토큰 개수를 계산합니다."""
        input_ids = self.tokenizer(query, return_tensors='pt')
        len_input_ids = input_ids['input_ids'].size()[1]
        return {"num_tokens": len_input_ids}

    async def cleansing_query(self, query: str) -> dict:
        """쿼리를 정제합니다."""
        # 한글 자음/모음과 특수문자 제거
        query = re.sub(self.BAD_PATTERN_HANGUL, '', query)
        query = re.sub(self.BAD_PATTERN_SYMBOLS, '', query)
        
        # 영문자만 있는 경우 비우기 (환경별로 오버라이드 가능)
        if self._should_remove_english_only() and re.match(self.BAD_PATTERN_ENGLISH, query):
            query = ''
        
        return {'cleansed_query': query.strip()}

    def _should_remove_english_only(self) -> bool:
        """영문자만 있는 쿼리를 제거할지 결정 (하위 클래스에서 오버라이드 가능)"""
        return True

    async def validate_length_bounds(self, query: str) -> dict:
        """쿼리 길이를 검증합니다."""
        query_len = len(query)
        is_valid = self.min_chars_thr <= query_len <= self.max_chars_thr
        return {"is_valid": is_valid}


class PrototypeQueryProcessor(BaseQueryProcessor):
    """Prototype 환경용 Query Processor"""
    pass


class DevelopmentQueryProcessor(BaseQueryProcessor):
    """Development 환경용 Query Processor"""
    pass


class ProductionQueryProcessor(BaseQueryProcessor):
    """Production 환경용 Query Processor"""
    
    def _should_remove_english_only(self) -> bool:
        """Production 환경에서는 영문자만 있는 쿼리도 허용"""
        return False
