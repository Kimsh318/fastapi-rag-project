docker run -d \
  --network app-network \
  -p 9090:9090 \
  -v ~/CursorAI_Project/langchain_fastapi_server_v11_application_logging_streaming_logger_implementation/prometheus/prometheus.yaml:/etc/prometheus/prometheus.yml \
  --name prometheus_container prom/prometheus