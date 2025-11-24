from string import Template

def create_prompt_templates() -> dict:
    #모든 task에 대한 prompt_without_data 딕셔너리 생성 및 반환
    prompt_dict = {}

    # rag task에 대한 템플릿 정의
    prompt_template_rag = """You are an AI chatbot that provides detailed *ANSWER* to user *QUESTION*. The *ANSWER* must be written using only the *SEARCH_RESULTS* provided to you. *SEARCH_RESULTS* typically consist of multiple documents, each separated by a delimiter "참고문서[$NUMBER$]". If *SEARCH_RESULTS* do not exist (e.g., *SEARCH_RESULTS* : @No data@), you should respond with '질문과 관련된 정보를 찾을 수 없습니다. 질문을 다시 작성해 보세요.' After providing the *ANSWER*, please provide the *SOURCES* used to write the *ANSWER*. All *ANSWER* must be in Korean.\n"""
    query_template_rag = Template("*QUESTION* : $query\nContinue to answer the *QUESTION* by using ONLY the *SEARCH_RESULTS*: ")
    answer_template_rag = "\n*ANSWER* : "
    prompt_dict['rag'] = Template(prompt_template_rag + query_template_rag.template + answer_template_rag)

    # free_talking task에 대한 템플릿 정의
    prompt_template_free_talking = """You are an AI chatbot designed for friendly conversation. Engage in an open-ended dialogue with the user on any topic they choose. Your responses should be polite, casual, and engaging. Keep the conversation going naturally.\n"""
    query_template_free_talking = Template("*QUESTION* : $query\nPlease respond in a conversational manner: ")
    answer_template_free_talking = "\n*ANSWER* : "
    prompt_dict['free_talking'] = Template(prompt_template_free_talking + query_template_free_talking.template + answer_template_free_talking)

    return prompt_dict

# 사용 예시
prompt_templates = create_prompt_templates()

print("\n" + "="*50 + "\n")
print(type(prompt_templates['rag']))
print(type(prompt_templates['rag'].template))
print(isinstance(prompt_templates['rag'], Template))
print(f"RAG Template(type): ")
print(prompt_templates['rag'].template)

print("\n" + "="*50 + "\n")
print("Free Talking Template:")
print(prompt_templates['free_talking'].template)



print("\n" + "="*50 + "\n")
# 나중에 query 값을 입력받아 사용
query = "((사용자 쿼리입니다.))"

# rag task 프롬프트 생성
rag_prompt = prompt_templates['rag'].safe_substitute(query=query)
print("RAG Prompt:")
print(rag_prompt)

print("\n" + "="*50 + "\n")

# free_talking task 프롬프트 생성
free_talking_prompt = prompt_templates['free_talking'].safe_substitute(query=query)
print("Free Talking Prompt:")
print(free_talking_prompt)