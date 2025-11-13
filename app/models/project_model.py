from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey, Enum, JSON
from sqlalchemy.sql import func
from app.core.database import Base
from sqlalchemy.orm import relationship
import uuid
from sqlalchemy.dialects.postgresql import UUID,ARRAY
from app.models.schemas import SystemPromptsTypeEnum

class Project(Base):
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    detailedsitemap = Column(Text)
    name = Column(Text)
    description = Column(Text)
    language = Column(Text)
    location = Column(Text)
    targeted_audience = Column(Text)
    guideline_description = Column(Text)
    website_url = Column(Text)
    keywords = Column(JSON) 
    competitors_websites = Column(Text)
    organization_archetype = Column(Text, nullable=True)
    brand_spokesperson = Column(Text, nullable=True)
    most_important_thing = Column(Text, nullable=True)
    unique_differentiator = Column(Text, nullable=True)
    author_bio = Column(Text, nullable=True)
    guideline_id = Column(UUID(as_uuid=True), ForeignKey("guideline.id"), nullable=True)
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now()) 
    articles = relationship("Article", back_populates="project", cascade="all, delete-orphan") 
    guideline = relationship("Guideline", back_populates="projects") 

class Article(Base):
    __tablename__ = "articles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    projectId = Column(Integer, ForeignKey("projects.id"), nullable=False)  
    name = Column(Text)
    generated_outline = Column(Text)
    article_rulesets = Column(Text, nullable=True)
    secondary_keywords = Column(ARRAY(String))
    keywords = Column(Text)
    prompt_type_id = Column(UUID(as_uuid=True))
    # systemPromptId = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # article_system_prompt_id = Column(UUID(as_uuid=True), ForeignKey("system_prompt.id")) 
    # outline_system_prompt_id = Column(UUID(as_uuid=True), ForeignKey("system_prompt.id")) 
    # title_system_prompt_id = Column(UUID(as_uuid=True), ForeignKey("system_prompt.id"))

    project = relationship("Project", back_populates="articles")

class PromptTypes(Base):
    __tablename__ = "prompt_types"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    topic_prompt_id = Column(UUID(as_uuid=True), ForeignKey("system_prompts.id"))
    outline_prompt_id = Column(UUID(as_uuid=True), ForeignKey("system_prompts.id"))
    article_prompt_id = Column(UUID(as_uuid=True), ForeignKey("system_prompts.id"))
    name = Column(Text) 


class Guideline(Base):
    __tablename__ = "guideline"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    # projectId = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)  
    description = Column(Text, nullable=True)

    projects = relationship("Project", back_populates="guideline")

class SystemPrompt(Base):
    __tablename__ = "system_prompts"  

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    name = Column(String, nullable=False) 
    type = Column(String, nullable=False)
    description = Column(Text, nullable=True) 
    is_default = Column(Boolean, default=False)
