from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class ModuleType(str, Enum):
    DEMOGRAPHICS = "demographics"
    FAMILY_ORIGIN = "family_origin"
    CORE_RELATIONSHIP = "core_relationship"
    FAMILY_CREATION = "family_creation"

class Framework(str, Enum):
    ATTACHMENT_THEORY = "Attachment Theory"
    GOTTMAN_METHOD = "Gottman Method"
    FAMILY_SYSTEMS = "Family Systems"

class Domain(str, Enum):
    VERBAL = "verbal"
    NONVERBAL = "nonverbal"
    EMOTIONAL = "emotional"
    CONFLICT = "conflict"
    INTIMACY = "intimacy"
    SUPPORT = "support"

class Question(BaseModel):
    id: str
    question: str
    category: str
    module: ModuleType
    framework: Optional[Framework] = None
    domain: Optional[Domain] = None

class Response(BaseModel):
    question_id: str
    response_text: str
    response_value: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class Session(BaseModel):
    session_id: str
    user_id: str
    module: ModuleType
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    responses: List[Response] = []
    status: str = "in_progress"

class Analysis(BaseModel):
    analysis_id: str
    session_id: str
    analyzer_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    results: Dict[str, Any]
    confidence_score: float = Field(ge=0, le=1)
    recommendations: List[str]

class RelationshipProfile(BaseModel):
    profile_id: str
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    demographics: Dict[str, Any]
    attachment_style: str
    communication_style: str
    conflict_style: str
    family_patterns: List[str]
    relationship_goals: List[str]
    health_indicators: Dict[str, float]
    risk_factors: List[str]
    strengths: List[str]
    areas_for_growth: List[str]

class AgentResponse(BaseModel):
    status: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_id: str
    agent_type: str

class AnalysisRequest(BaseModel):
    session_id: str
    module: ModuleType
    framework: Optional[Framework] = None
    domain: Optional[Domain] = None
    data: Dict[str, Any]

class RecommendationRequest(BaseModel):
    analysis_id: str
    focus_areas: List[str]
    context: Dict[str, Any]

class ScoringMetrics(BaseModel):
    objective_health: float = Field(ge=0, le=100)
    subjective_satisfaction: float = Field(ge=0, le=100)
    family_pattern_alignment: float = Field(ge=0, le=100)
    partner_compatibility: float = Field(ge=0, le=100)
    confidence_ratings: Dict[str, float]

class RelationshipAssessment(BaseModel):
    assessment_id: str
    profile_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    scoring_metrics: ScoringMetrics
    analyses: List[Analysis]
    recommendations: List[str]
    next_steps: List[str]
    review_date: datetime