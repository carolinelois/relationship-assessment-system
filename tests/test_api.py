import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from datetime import datetime
from ..src.models.schemas import ModuleType, Session as SessionModel
from ..src.utils.security import create_access_token

def test_health_check(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data
    assert "timestamp" in data
    assert "metrics" in data

def test_create_session_unauthorized(client: TestClient):
    response = client.post("/api/v1/sessions/", json={
        "module": ModuleType.DEMOGRAPHICS,
        "user_id": "test_user"
    })
    assert response.status_code == 401

def test_create_session_authorized(client: TestClient, db_session: Session):
    # Create access token
    access_token = create_access_token({"sub": "test_user", "role": "user"})
    
    response = client.post(
        "/api/v1/sessions/",
        json={
            "module": ModuleType.DEMOGRAPHICS,
            "user_id": "test_user"
        },
        headers={"Authorization": f"Bearer {access_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["user_id"] == "test_user"
    assert data["module"] == ModuleType.DEMOGRAPHICS
    assert data["status"] == "in_progress"

def test_get_session(client: TestClient, db_session: Session):
    # Create access token
    access_token = create_access_token({"sub": "test_user", "role": "user"})
    
    # First create a session
    create_response = client.post(
        "/api/v1/sessions/",
        json={
            "module": ModuleType.DEMOGRAPHICS,
            "user_id": "test_user"
        },
        headers={"Authorization": f"Bearer {access_token}"}
    )
    
    session_id = create_response.json()["session_id"]
    
    # Now get the session
    get_response = client.get(
        f"/api/v1/sessions/{session_id}",
        headers={"Authorization": f"Bearer {access_token}"}
    )
    
    assert get_response.status_code == 200
    data = get_response.json()
    assert data["session_id"] == session_id
    assert data["user_id"] == "test_user"

def test_add_response(client: TestClient, db_session: Session):
    # Create access token
    access_token = create_access_token({"sub": "test_user", "role": "user"})
    
    # Create a session
    session_response = client.post(
        "/api/v1/sessions/",
        json={
            "module": ModuleType.DEMOGRAPHICS,
            "user_id": "test_user"
        },
        headers={"Authorization": f"Bearer {access_token}"}
    )
    
    session_id = session_response.json()["session_id"]
    
    # Add a response
    response = client.post(
        f"/api/v1/sessions/{session_id}/responses",
        json={
            "question_id": "q1",
            "response_text": "Test response",
            "response_value": 5.0
        },
        headers={"Authorization": f"Bearer {access_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["question_id"] == "q1"
    assert data["response_text"] == "Test response"
    assert data["response_value"] == 5.0

def test_request_analysis(client: TestClient, db_session: Session):
    # Create access token
    access_token = create_access_token({"sub": "test_user", "role": "user"})
    
    # Create a session
    session_response = client.post(
        "/api/v1/sessions/",
        json={
            "module": ModuleType.DEMOGRAPHICS,
            "user_id": "test_user"
        },
        headers={"Authorization": f"Bearer {access_token}"}
    )
    
    session_id = session_response.json()["session_id"]
    
    # Request analysis
    response = client.post(
        "/api/v1/analysis/request",
        json={
            "session_id": session_id,
            "module": ModuleType.DEMOGRAPHICS,
            "data": {
                "responses": [
                    {
                        "question_id": "q1",
                        "response_text": "Test response",
                        "response_value": 5.0
                    }
                ]
            }
        },
        headers={"Authorization": f"Bearer {access_token}"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "analysis_id" in data
    assert data["session_id"] == session_id