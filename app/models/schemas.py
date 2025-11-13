from pydantic import BaseModel, Field, UUID4
from typing import List, Optional, Dict
import uuid
from enum import Enum

class RequestData(BaseModel):
    title: str
    keywords: list
    company_name: str
    language: str
    locations: List[str]

class RequestData2(BaseModel):
    company_name: str

class CompanyOutline(BaseModel):
    ProjectId: str

class Titleoutline(BaseModel):
    ProjectId: str
    title: str

class outline(BaseModel):
    articleId: str

class RequestData3(BaseModel):
    client_site_url: str
    title: str
    keyword: List[str]

class Sitemap(BaseModel):
    company_name: str

class CompanyDetails(BaseModel):
    company_details: str

class FindTitle(BaseModel):
    ProjectId: str
    title: str

class URLInput(BaseModel):
    url: str

class RequestDataSite(BaseModel):
    client_site_url : str
    competitor_urls: Optional[List[str]] = None
    keywords: List[str]
    language: Optional[str] = None
    locations: Optional[List[str]] = None


# class MetaAnalysisResult(BaseModel):
#     url: str
#     status: str
#     meta_title: Optional[str] = None
#     meta_description: Optional[str] = None
#     #generated_title: Optional[str] = None
#     #generated_description: Optional[str] = None
#     keywords: Optional[str] = None
#     error: Optional[str] = None
#     content_type: Optional[str] = None

class MetaAnalysisResult(BaseModel):
    url: str
    status: str
    meta_title: Optional[str] = None
    meta_description: Optional[str] = None
    keywords: Optional[str] = None
    error: Optional[str] = None
    content_type: Optional[str] = None

class SystemPromptsTypeEnum(str, Enum):
    ARTICLE = "article"

class ArticleRequest(BaseModel):
    articleId: str
    model: Optional[str] = None  
    requestId: str
    # avg_word_count: Optional[int] = None 

class KeywordItem(BaseModel):
    keyword: str
    promptTypeId: str

class TitlesRequest(BaseModel):
    ProjectId: str
    Keywords: List[KeywordItem]

class PdfData(BaseModel):
    filename: str
    extracted_text: str


