docker run -d -it --network app-network -p 11434:11434 --name ollama_container ollama/ollama # Ollama 컨테이너 실행
docker exec -it ollama_container ollama run gemma2:2b # 모델 다운로드 및 실행