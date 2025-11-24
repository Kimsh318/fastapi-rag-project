
# 단일 노드 모드 실행 : -e "discovery.type=single-node"
# ES apache 2.0 버전 실행 : elasticsearch:7.10.1

docker run -d --name es_container --network app-network -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.10.1