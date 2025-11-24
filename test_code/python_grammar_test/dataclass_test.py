from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int
    city: str

# 객체 생성
person = Person(name="Alice", age=30, city="New York")

# 자동으로 생성된 __repr__ 메서드 출력
print(person)  # Person(name='Alice', age=30, city='New York')
