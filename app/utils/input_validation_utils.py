def validate_input(query):
    # 입력 검증 로직
    if not query or not isinstance(query, str):
        raise ValueError("Invalid query")
    return query
