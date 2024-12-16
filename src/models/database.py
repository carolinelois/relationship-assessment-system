from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Table
from sqlalchemy.orm import relationship
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./relationship_assessment.db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Association tables
session_analysis = Table(
    'session_analysis',
    Base.metadata,
    Column('session_id', String, ForeignKey('sessions.session_id')),
    Column('analysis_id', String, ForeignKey('analyses.analysis_id'))
)

class DBSession(Base):
    __tablename__ = "sessions"

    session_id = Column(String, primary_key=True)
    user_id = Column(String, index=True)
    module = Column(String)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    status = Column(String, default="in_progress")
    responses = relationship("DBResponse", back_populates="session")
    analyses = relationship("DBAnalysis", secondary=session_analysis, back_populates="sessions")

class DBResponse(Base):
    __tablename__ = "responses"

    response_id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.session_id"))
    question_id = Column(String)
    response_text = Column(String)
    response_value = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    session = relationship("DBSession", back_populates="responses")

class DBAnalysis(Base):
    __tablename__ = "analyses"

    analysis_id = Column(String, primary_key=True)
    analyzer_id = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    results = Column(JSON)
    confidence_score = Column(Float)
    recommendations = Column(JSON)
    sessions = relationship("DBSession", secondary=session_analysis, back_populates="analyses")

class DBProfile(Base):
    __tablename__ = "profiles"

    profile_id = Column(String, primary_key=True)
    user_id = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    demographics = Column(JSON)
    attachment_style = Column(String)
    communication_style = Column(String)
    conflict_style = Column(String)
    family_patterns = Column(JSON)
    relationship_goals = Column(JSON)
    health_indicators = Column(JSON)
    risk_factors = Column(JSON)
    strengths = Column(JSON)
    areas_for_growth = Column(JSON)
    assessments = relationship("DBAssessment", back_populates="profile")

class DBAssessment(Base):
    __tablename__ = "assessments"

    assessment_id = Column(String, primary_key=True)
    profile_id = Column(String, ForeignKey("profiles.profile_id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    scoring_metrics = Column(JSON)
    recommendations = Column(JSON)
    next_steps = Column(JSON)
    review_date = Column(DateTime)
    profile = relationship("DBProfile", back_populates="assessments")

class DBQuestion(Base):
    __tablename__ = "questions"

    question_id = Column(String, primary_key=True)
    question_text = Column(String)
    category = Column(String)
    module = Column(String)
    framework = Column(String, nullable=True)
    domain = Column(String, nullable=True)
    metadata = Column(JSON)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(bind=engine)