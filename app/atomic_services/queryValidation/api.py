# app/services/queryValidate/api.py
from functools import lru_cache
from fastapi import APIRouter, Depends, HTTPException

from app.core.config import settings
from app.atomic_services.queryValidation.service import QueryValidationService
from app.atomic_services.queryValidation.models import QueryValidationRequest, QueryValidationResponse
from app.atomic_services.queryValidation.helpers import get_tokenizer_info, get_validation_criteria
from app.utils.logging_utils import log_api_call, log_execution_time_async

router = APIRouter()


@lru_cache(maxsize=None, typed=False)   
def get_query_validation_service():
    tokenizer_info = get_tokenizer_info(settings)
    validation_criteria = get_validation_criteria(settings)

    return QueryValidationService(tokenizer_info=tokenizer_info, validation_criteria=validation_criteria, app_env=settings.APP_ENVIRONMENT)


@router.post("/query_validate", response_model=QueryValidationResponse)
@log_execution_time_async # @log_api_call(endpoint='query_validate', index_type="api")
async def validate_query(request: QueryValidationRequest, 
                    service: QueryValidationService = Depends(get_query_validation_service)):
    
    validation_results = await service.validate(query=request.query)
    
    if not validation_results["is_valid"]:
        raise HTTPException(status_code=400, detail=f"유효하지 않은 질의입니다. 토큰 수: {validation_results['num_tokens']}")    
        
    return {"is_valid": validation_results["is_valid"], \
            "num_tokens": validation_results["num_tokens"]}
