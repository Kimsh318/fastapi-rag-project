"""
# 파이썬 `abc` 모듈을 이용한 추상 클래스 사용법
## 개요
파이썬의 `abc` 모듈은 **추상 베이스 클래스(ABC, Abstract Base Class)**를 정의할 때 사용하는 모듈입니다. 추상 클래스는 특정 메서드를 하위 클래스에서 반드시 구현하도록 강제하며, 객체 지향 프로그래밍의 원칙을 쉽게 적용할 수 있도록 도와줍니다. 
이를 통해 클래스 계층 간의 일관성을 유지하고 코드의 재사용성을 높일 수 있습니다.

## `abc` 모듈의 주요 구성 요소
- **`ABC` 클래스**: 추상 클래스의 부모 클래스 역할을 하는 베이스 클래스입니다. 
    이를 상속받으면 해당 클래스는 추상 클래스로 취급됩니다.
- **`@abstractmethod` 데코레이터**: 추상 메서드를 정의할 때 사용하는 데코레이터입니다. 
    `@abstractmethod`가 적용된 메서드는 하위 클래스에서 반드시 구현해야 하며, 그렇지 않으면 인스턴스를 생성할 수 없습니다.

## `abc` 모듈 사용 예제
`abc` 모듈을 사용하여 동물(Animal) 추상 클래스를 정의하고, 이를 상속받는 구체적인 동물 클래스(Dog, Bird)를 만들어 보겠습니다.
"""
from abc import ABC, abstractmethod

# Animal 추상 클래스를 정의합니다.
class Animal(ABC):
    @abstractmethod
    def sound(self):
        """각 동물의 소리를 정의하는 추상 메서드"""
        pass

    @abstractmethod
    def move(self):
        """각 동물의 움직임을 정의하는 추상 메서드"""
        pass

# Dog 클래스는 Animal 클래스를 상속받고, 추상 메서드를 모두 구현해야 합니다.
class Dog(Animal):
    def sound(self):
        return "Bark"

    def move(self):
        return "Runs"

# Bird 클래스는 Animal 클래스를 상속받고, 추상 메서드를 모두 구현해야 합니다.
class Bird(Animal):
    def sound(self):
        return "Chirp"

    # def move(self):
    #     return "Flies"

if __name__=="__main__":
    # 구체적인 클래스 사용 예제
    dog = Dog()
    print(dog.sound())  # 출력: Bark
    print(dog.move())   # 출력: Runs

    # 이 코드라인에서는 에러를 출력할 것임 : move 메소드 구현하지 않음
    bird = Bird()
    try:
        print(bird.sound()) # 출력: Chirp
        print(bird.move())  # 출력: Flies
    except Exception as e:
        print(e)