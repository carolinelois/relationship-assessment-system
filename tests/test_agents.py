import pytest
from ..src.agents.data_collection_agent import DataCollectionAgent
from ..src.agents.expert_agents import RelationshipPsychologistAgent, BehavioralPsychologistAgent
from ..src.agents.analysis_agents import DemographicAnalyzer, AttachmentAnalyzer
from ..src.models.schemas import Framework, Domain

@pytest.mark.asyncio
async def test_data_collection_agent():
    agent = DataCollectionAgent()
    
    # Test session start
    start_response = await agent.process({
        "action": "start_session",
        "session_id": "test_session",
        "module": "demographics"
    })
    
    assert start_response["status"] == "success"
    assert start_response["data"]["session_id"] == "test_session"
    
    # Test response handling
    response_data = await agent.process({
        "action": "submit_response",
        "session_id": "test_session",
        "response": {
            "question_id": "q1",
            "response_text": "Test response"
        }
    })
    
    assert response_data["status"] == "success"
    assert response_data["data"]["status"] == "response_recorded"

@pytest.mark.asyncio
async def test_relationship_psychologist_agent():
    agent = RelationshipPsychologistAgent()
    
    # Test question generation
    questions_response = await agent.process({
        "analysis_type": "generate_questions",
        "framework": Framework.ATTACHMENT_THEORY
    })
    
    assert questions_response["status"] == "success"
    assert "questions" in questions_response["data"]
    assert len(questions_response["data"]["questions"]) > 0
    
    # Test response analysis
    analysis_response = await agent.process({
        "analysis_type": "analyze_responses",
        "framework": Framework.ATTACHMENT_THEORY,
        "responses": [
            {
                "question_id": "at_1",
                "response_text": "I feel anxious",
                "response_value": 4
            }
        ]
    })
    
    assert analysis_response["status"] == "success"
    assert "analysis" in analysis_response["data"]

@pytest.mark.asyncio
async def test_behavioral_psychologist_agent():
    agent = BehavioralPsychologistAgent()
    
    # Test behavior analysis
    analysis_response = await agent.process({
        "action": "analyze_behavior",
        "domain": Domain.VERBAL,
        "behaviors": [
            {
                "type": "communication",
                "description": "Clear and direct",
                "frequency": "often"
            }
        ]
    })
    
    assert analysis_response["status"] == "success"
    assert "analysis" in analysis_response["data"]
    
    # Test recommendation generation
    recommendations_response = await agent.process({
        "action": "generate_recommendations",
        "domain": Domain.VERBAL,
        "analysis": analysis_response["data"]["analysis"]
    })
    
    assert recommendations_response["status"] == "success"
    assert "recommendations" in recommendations_response["data"]

@pytest.mark.asyncio
async def test_demographic_analyzer():
    analyzer = DemographicAnalyzer()
    
    analysis_response = await analyzer.process({
        "demographics": {
            "age": 30,
            "culture": "Western",
            "religion": "None",
            "education": "Bachelor's",
            "occupation": "Professional",
            "family_structure": "Nuclear"
        }
    })
    
    assert analysis_response["status"] == "success"
    assert "analysis" in analysis_response["data"]
    assert "cultural_analysis" in analysis_response["data"]["analysis"]
    assert "socioeconomic_analysis" in analysis_response["data"]["analysis"]

@pytest.mark.asyncio
async def test_attachment_analyzer():
    analyzer = AttachmentAnalyzer()
    
    analysis_response = await analyzer.process({
        "responses": {
            "childhood_experiences": [
                {
                    "category": "parental_relationship",
                    "description": "Supportive and consistent"
                }
            ],
            "current_patterns": [
                {
                    "category": "intimacy",
                    "description": "Comfortable with closeness"
                }
            ]
        }
    })
    
    assert analysis_response["status"] == "success"
    assert "analysis" in analysis_response["data"]
    assert "primary_style" in analysis_response["data"]["analysis"]
    assert "recommendations" in analysis_response["data"]["analysis"]