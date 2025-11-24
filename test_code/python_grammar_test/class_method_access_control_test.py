"""
Python의 접근 제어 (Access Control) 예제
- 일반 메서드: 외부에서 자유롭게 접근 가능
- Protected 메서드 (_): 상속과 내부 사용을 위한 메서드
- Private 메서드 (__): 클래스 내부에서만 사용되는 메서드
"""
from typing import Any
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseClass:
    """기본 클래스: 다양한 접근 제어 방식을 보여주는 예제"""
    
    def __init__(self, name: str):
        self.name = name                    # 일반 속성
        self._internal_id = 0               # protected 속성
        self.__private_counter = 0          # private 속성
        
    def public_method(self, data: Any) -> str:
        """공개 메서드: 누구나 접근 가능"""
        logger.info(f"Processing data in {self.name}")
        processed = self._internal_process(data)
        self.__update_counter()
        return f"Processed: {processed}"
    
    def _internal_process(self, data: Any) -> Any:
        """Protected 메서드: 주로 상속용 내부 처리 메서드
        자식 클래스에서 오버라이드 가능"""
        return f"{data} (기본 처리)"
    
    def __update_counter(self) -> None:
        """Private 메서드: 클래스 내부에서만 사용"""
        self.__private_counter += 1
        logger.debug(f"Counter updated: {self.__private_counter}")
    
    def get_counter(self) -> int:
        """Private 속성에 접근하기 위한 공개 메서드"""
        return self.__private_counter


class ChildClass(BaseClass):
    """BaseClass를 상속받는 자식 클래스"""
    
    def __init__(self, name: str, processor_type: str):
        super().__init__(name)
        self.processor_type = processor_type
    
    def _internal_process(self, data: Any) -> Any:
        """Protected 메서드 오버라이드"""
        basic_result = super()._internal_process(data)
        return f"{basic_result} (추가 처리: {self.processor_type})"
    
    def advanced_process(self, data: Any) -> str:
        """자식 클래스의 새로운 공개 메서드"""
        return f"Advanced {self.public_method(data)}"


def main():
    """메인 실행 함수: 접근 제어 예제 실행"""
    # 기본 클래스 사용 예제
    logger.info("=== 기본 클래스 테스트 ===")
    base = BaseClass("Base")
    print(f"Public method: {base.public_method('test')}")
    print(f"Counter value: {base.get_counter()}")
    
    # Protected 메서드 접근 (가능하지만 경고)
    print(f"Protected method: {base._internal_process('test')}")
    
    # Private 메서드 직접 접근 시도 (에러 발생)
    try:
        base.__update_counter()
    except AttributeError as e:
        print(f"Private method access error: {e}")
    
    # 자식 클래스 사용 예제
    logger.info("\n=== 자식 클래스 테스트 ===")
    child = ChildClass("Child", "Special")
    print(f"Child public method: {child.public_method('test')}")
    print(f"Child advanced process: {child.advanced_process('test')}")
    print(f"Child counter value: {child.get_counter()}")


if __name__ == "__main__":
    main()