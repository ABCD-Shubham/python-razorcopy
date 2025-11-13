from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.content_generator import generate_content

router = APIRouter()

class ArticleRequest(BaseModel):
    title: list[str]
    keywords: list[str]
    company_name: str

class ArticleResponse(BaseModel):
    article: str
    avg_word_count: float

@router.post("/generate-article", response_model=ArticleResponse)
async def generate_article(request_data: ArticleRequest):
    try:
        generated_content = await generate_content(
            title=request_data.title,
            keywords=request_data.keywords,
            company_name=request_data.company_name
        )
        return {
            "article": generated_content['article'],
            "avg_word_count": generated_content['avg_word_count']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))