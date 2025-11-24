## 외부 포트 : 8000
## ChromaDB 포트 : 9202
# docker run -d -it --network app-network -p 8000:8000 -p 9202:9202 -v ~/CursorAI_Project/langchain_fastapi_server_v9_application_logging:/app --name langchain_fastapi_container langchain_fastapi_server_image 
#docker run -d -it --network app-network -p 8000:8000 -p 9202:9202 -v ~/CursorAI_Project/langchain_fastapi_server_v9_applicadtion_logging:/app --name langchain_fastapi_container langchain_fastapi_server_image_light

docker run -d -it --network app-network -p 8000:8000 -p 9202:9202 -v ~/CursorAI_Project/langchain_fastapi_server_v24_refactoring:/app --name langchain_fastapi_container langchain_fastapi_server_image_light

#docker run -d -it -p 8000:8000 -p 9202:9202 -v ~/CursorAI_Project/langchain_fastapi_server_v4_w_legacy_code:/app --name langchain_fastapi_container langchain_fastapi_server_image 
#docker network connect --ip 172.18.0.5 app-network langchain_fastapi_container