def ori_calculate_rrf_scores(es_result_ids: list, vs_result_ids: list, k_weight: float, v_weight: float) -> dict:
    """
    RRF(Reciprocal Rank Fusion) 점수를 산출하는 함수.
    문서 ID별로 계산된 RRF 점수를 담은 딕셔너리 (key: 문서 ID, value: RRF 점수)를 반환
    """
    
    # 두 검색 결과에서 문서의 순위를 계산
    def get_ranking(es_result_ids: list, vs_result_ids: list) -> dict: 
        all_ids = list(set(es_result_ids + vs_result_ids))  # 검색된 모든 문서 ID의 리스트
        ranks_of_both_retrievers = {}

        # 각 문서에 대해 ES와 VS에서의 순위를 계산
        for _id in all_ids:
            rank_es = es_result_ids.index(_id) + 1 if _id in es_result_ids else None
            rank_vs = vs_result_ids.index(_id) + 1 if _id in vs_result_ids else None
            ranks_of_both_retrievers[_id] = [rank_es, rank_vs]
        return ranks_of_both_retrievers  # 문서 ID별로 ES와 VS의 순위를 담은 딕셔너리 반환

    # 주어진 순위에 대한 Reciprocal Rank 계산
    def reciprocal_rank(rank):
        return 1 / rank if isinstance(rank, int) else 0.0  # 순위가 존재하면 Reciprocal Rank 계산, 없으면 0 반환
    
    all_ranks = get_ranking(es_result_ids, vs_result_ids)
    sorted_dict_for_ids_to_scores = {}
    scores = []

    # RRF 점수를 계산하고 스코어를 리스트에 저장
    for _key in all_ranks.keys():  # 각 문서 ID에 대해
        es_rank: int = all_ranks[_key][0]  # ES에서의 순위
        vs_rank: int = all_ranks[_key][1]  # VS에서의 순위

        # 각 검색 시스템의 Reciprocal Rank 계산
        es_rr: float = reciprocal_rank(es_rank)
        vs_rr: float = reciprocal_rank(vs_rank)

        # 가중치를 적용한 RRF 점수 계산
        rrf = (v_weight * vs_rr) + (k_weight * es_rr)  # 가중치를 적용한 RRF 점수 계산
        scores.append(rrf)

    # 계산된 RRF 점수에 따라 문서 ID를 내림차순으로 정렬
    sorted_scores = sorted(scores, reverse=True)
    sorted_ids = [_key for _, _key in sorted(zip(scores, all_ranks.keys()), reverse=True)]
    
    # 정렬된 스코어에 따라 문서 ID와 점수를 딕셔너리에 저장
    for i in range(len(sorted_ids)):
        sorted_dict_for_ids_to_scores[sorted_ids[i]] = sorted_scores[i]

    return sorted_dict_for_ids_to_scores  # 문서 ID별로 계산된 RRF 점수를 반환



def new_calculate_rrf_scores(es_result_ids: list, vs_result_ids: list, k_weight: float, v_weight: float) -> dict:
    """
    키워드 검색(ES) 결과와 벡터 검색(VS) 결과를 RRF(Reciprocal Rank Fusion) 방식으로 결합하여 RRF 점수를 산출하는 함수.
    
    매개변수:
    ----------
    es_result_ids : list
        Elasticsearch 키워드 검색 결과 문서 ID 리스트.
    
    vs_result_ids : list
        벡터 검색 결과 문서 ID 리스트.
    
    k_weight : float
        키워드 검색에 적용할 가중치.
    
    v_weight : float
        벡터 검색에 적용할 가중치.

    반환값:
    -------
    dict
        문서 ID별로 계산된 RRF 점수를 담은 딕셔너리.
    """
    
    def get_rankings(es_ids: list, vs_ids: list) -> dict:
        """두 검색 결과의 문서 ID에 대한 순위를 계산하여 반환."""
        all_ids = set(es_ids + vs_ids)  # 두 검색 결과에 포함된 모든 문서 ID
        ranks = {}

        for _id in all_ids:
            es_rank = es_ids.index(_id) + 1 if _id in es_ids else None
            vs_rank = vs_ids.index(_id) + 1 if _id in vs_ids else None
            ranks[_id] = [es_rank, vs_rank]
        
        return ranks

    def reciprocal_rank(rank: int) -> float:
        """주어진 순위에 대해 Reciprocal Rank를 계산."""
        return 1 / rank if rank is not None else 0.0

    # 각 문서에 대한 ES와 VS 순위를 계산
    doc_ranks = get_rankings(es_result_ids, vs_result_ids)

    # RRF 점수를 계산하여 딕셔너리로 저장
    rrf_scores = {
        doc_id: (k_weight * reciprocal_rank(ranks[0]) + v_weight * reciprocal_rank(ranks[1]))
        for doc_id, ranks in doc_ranks.items()
    }

    # RRF 점수 기준으로 내림차순 정렬
    sorted_rrf_scores = dict(sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True))
    
    return sorted_rrf_scores


def test(es_result_ids, vs_result_ids, k_weight, v_weight):
    # 기존 코드 결과
    original_result = ori_calculate_rrf_scores(es_result_ids, vs_result_ids, k_weight, v_weight)
    print("Original Result:", original_result)

    # 개선된 코드 결과
    new_result = new_calculate_rrf_scores(es_result_ids, vs_result_ids, k_weight, v_weight)
    print("New Result:", new_result)

    # 두 결과가 동일한지 확인
    print("Results are identical:", original_result == new_result)

print('\n\n==========')
es_result_ids = ['doc1', 'doc2', 'doc3', 'doc4']
vs_result_ids = ['doc2', 'doc3', 'doc5', 'doc6']
k_weight = 0.7  # 키워드 검색 가중치
v_weight = 0.3  # 벡터 검색 가중치
test(es_result_ids, vs_result_ids, k_weight, v_weight)

print('\n\n==========')
es_result_ids_1 = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
vs_result_ids_1 = ['doc3', 'doc4', 'doc6', 'doc7', 'doc8']
k_weight_1 = 0.7
v_weight_1 = 0.3
test(es_result_ids, vs_result_ids, k_weight, v_weight)


print('\n\n==========')
es_result_ids_2 = ['doc10', 'doc11', 'doc12', 'doc13']
vs_result_ids_2 = ['doc12', 'doc14', 'doc15']
k_weight_2 = 0.5
v_weight_2 = 0.5
test(es_result_ids, vs_result_ids, k_weight, v_weight)


print('\n\n==========')
es_result_ids_3 = ['doc20', 'doc21', 'doc22', 'doc23', 'doc24']
vs_result_ids_3 = ['doc22', 'doc25', 'doc26', 'doc27', 'doc28']
k_weight_3 = 0.6
v_weight_3 = 0.4
test(es_result_ids, vs_result_ids, k_weight, v_weight)


print('\n\n==========')
es_result_ids_4 = ['doc30', 'doc31', 'doc32']
vs_result_ids_4 = ['doc33', 'doc34', 'doc35', 'doc36']
k_weight_4 = 0.8
v_weight_4 = 0.2
test(es_result_ids, vs_result_ids, k_weight, v_weight)


print('\n\n==========')
es_result_ids_5 = ['doc40', 'doc41', 'doc42', 'doc43']
vs_result_ids_5 = ['doc41', 'doc43', 'doc44', 'doc45']
k_weight_5 = 0.9
v_weight_5 = 0.1
test(es_result_ids, vs_result_ids, k_weight, v_weight)
