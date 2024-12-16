from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from ..models.schemas import (
    Session as SessionModel,
    Analysis, RelationshipProfile, AnalysisRequest,
    RecommendationRequest, RelationshipAssessment, Question, Response,
    ModuleType
)
from ..models.database import get_db
from ..processing.data_pipeline import DataProcessor
from ..processing.validation import DataValidator, ValidationLevel
from ..reporting.report_generator import ReportGenerator, ReportConfig, ReportType
from ..reporting.recommendation_engine import RecommendationEngine
from datetime import datetime
from loguru import logger

router = APIRouter()
data_processor = DataProcessor()
data_validator = DataValidator()
recommendation_engine = RecommendationEngine()

@router.post("/sessions/", response_model=SessionModel)
async def create_session(
    module: ModuleType,
    user_id: str,
    db: Session = Depends(get_db)
):
    try:
        session = SessionModel(
            session_id=f"sess_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            module=module,
            start_time=datetime.utcnow(),
            status="in_progress"
        )
        
        # Validate session
        validation_result = data_validator.validate_session_data(session, [])
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid session data",
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings
                }
            )
        
        # Persist session
        db.add(session)
        db.commit()
        db.refresh(session)
        
        return session
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/sessions/{session_id}", response_model=SessionModel)
async def get_session(session_id: str, db: Session = Depends(get_db)):
    session = db.query(SessionModel).filter(
        SessionModel.session_id == session_id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session

@router.post("/sessions/{session_id}/responses", response_model=Response)
async def add_response(
    session_id: str,
    response: Response,
    db: Session = Depends(get_db)
):
    try:
        # Get session
        session = db.query(SessionModel).filter(
            SessionModel.session_id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Validate response
        validation_result = data_validator.validate_responses([response], session)
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid response data",
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings
                }
            )
        
        # Add response to session
        response.session_id = session_id
        db.add(response)
        db.commit()
        db.refresh(response)
        
        return response
    except Exception as e:
        logger.error(f"Error adding response: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/analysis/request", response_model=Analysis)
async def request_analysis(
    request: AnalysisRequest,
    db: Session = Depends(get_db)
):
    try:
        # Get session and responses
        session = db.query(SessionModel).filter(
            SessionModel.session_id == request.session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        responses = db.query(Response).filter(
            Response.session_id == request.session_id
        ).all()
        
        # Process data
        processing_result = await data_processor.process_session_data(
            session, responses
        )
        
        if processing_result["status"] == "error":
            raise HTTPException(
                status_code=400,
                detail=processing_result["error"]
            )
        
        # Create analysis
        analysis = Analysis(
            analysis_id=f"ana_{datetime.utcnow().timestamp()}",
            session_id=request.session_id,
            analyzer_id="relationship_analyzer",
            results=processing_result["results"],
            recommendations=processing_result["recommendations"],
            confidence_score=processing_result["results"].get(
                "confidence_score", 0.0
            )
        )
        
        # Validate analysis
        validation_result = data_validator.validate_analysis_data(analysis)
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid analysis data",
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings
                }
            )
        
        # Persist analysis
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
        
        return analysis
    except Exception as e:
        logger.error(f"Error processing analysis: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/analysis/{analysis_id}", response_model=Analysis)
async def get_analysis(analysis_id: str, db: Session = Depends(get_db)):
    analysis = db.query(Analysis).filter(
        Analysis.analysis_id == analysis_id
    ).first()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis

@router.post("/profiles/", response_model=RelationshipProfile)
async def create_profile(
    user_id: str,
    demographics: Dict[str, Any],
    db: Session = Depends(get_db)
):
    try:
        profile = RelationshipProfile(
            profile_id=f"prof_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            demographics=demographics,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Validate profile
        validation_result = data_validator.validate_profile_data(profile)
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid profile data",
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings
                }
            )
        
        # Persist profile
        db.add(profile)
        db.commit()
        db.refresh(profile)
        
        return profile
    except Exception as e:
        logger.error(f"Error creating profile: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/profiles/{profile_id}", response_model=RelationshipProfile)
async def get_profile(profile_id: str, db: Session = Depends(get_db)):
    profile = db.query(RelationshipProfile).filter(
        RelationshipProfile.profile_id == profile_id
    ).first()
    
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    return profile

@router.post("/recommendations/request", response_model=List[str])
async def request_recommendations(
    request: RecommendationRequest,
    db: Session = Depends(get_db)
):
    try:
        # Get analysis
        analysis = db.query(Analysis).filter(
            Analysis.analysis_id == request.analysis_id
        ).first()
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Generate recommendations
        recommendations = recommendation_engine.generate_recommendations({
            "analysis": analysis.results,
            "focus_areas": request.focus_areas,
            "context": request.context
        })
        
        return recommendations
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/assessments/", response_model=RelationshipAssessment)
async def create_assessment(
    profile_id: str,
    db: Session = Depends(get_db)
):
    try:
        # Get profile
        profile = db.query(RelationshipProfile).filter(
            RelationshipProfile.profile_id == profile_id
        ).first()
        
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        # Get analyses for profile
        analyses = db.query(Analysis).filter(
            Analysis.profile_id == profile_id
        ).all()
        
        # Create assessment
        assessment = RelationshipAssessment(
            assessment_id=f"assess_{datetime.utcnow().timestamp()}",
            profile_id=profile_id,
            timestamp=datetime.utcnow(),
            scoring_metrics=data_processor._calculate_scoring_metrics(analyses),
            analyses=analyses,
            recommendations=recommendation_engine.generate_recommendations({
                "profile": profile,
                "analyses": analyses
            }),
            next_steps=data_processor._generate_next_steps(profile, analyses),
            review_date=datetime.utcnow()
        )
        
        # Persist assessment
        db.add(assessment)
        db.commit()
        db.refresh(assessment)
        
        return assessment
    except Exception as e:
        logger.error(f"Error creating assessment: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/assessments/{assessment_id}", response_model=RelationshipAssessment)
async def get_assessment(assessment_id: str, db: Session = Depends(get_db)):
    assessment = db.query(RelationshipAssessment).filter(
        RelationshipAssessment.assessment_id == assessment_id
    ).first()
    
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
    
    return assessment

@router.get("/questions/", response_model=List[Question])
async def get_questions(
    module: Optional[ModuleType] = None,
    framework: Optional[str] = None,
    domain: Optional[str] = None,
    db: Session = Depends(get_db)
):
    try:
        query = db.query(Question)
        
        if module:
            query = query.filter(Question.module == module)
        if framework:
            query = query.filter(Question.framework == framework)
        if domain:
            query = query.filter(Question.domain == domain)
        
        questions = query.all()
        return questions
    except Exception as e:
        logger.error(f"Error retrieving questions: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "relationship-assessment-system",
        "timestamp": datetime.utcnow().isoformat()
    }