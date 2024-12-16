from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from datetime import datetime
import time
from . import endpoints
from ..utils.logging import setup_logging, log_api_request, log_api_response
from ..utils.monitoring import APIMonitor
from ..utils.security import security_manager, User, get_current_user
from ..utils.config import config
from ..models.database import init_db

# Set up logging
logger = setup_logging()

# Initialize FastAPI app
app = FastAPI(
    title="Relationship Assessment System",
    description="A multi-agent system for relationship assessment and analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get_value("api", "cors", "origins", default=["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize API monitor
api_monitor = APIMonitor()

# Middleware for logging and monitoring
@app.middleware("http")
async def log_and_monitor_requests(request, call_next):
    start_time = time.time()
    
    # Log request
    log_api_request(request.method, request.url.path, dict(request.query_params))
    
    # Process request
    response = await call_next(request)
    
    # Calculate response time
    response_time = time.time() - start_time
    
    # Log and monitor response
    log_api_response(request.url.path, response.status_code, response_time)
    await api_monitor.record_request(request.url.path, response_time, response.status_code)
    
    return response

# Include API routes
app.include_router(
    endpoints.router,
    prefix="/api/v1",
    dependencies=[Depends(get_current_user)]
)

@app.on_event("startup")
async def startup_event():
    # Initialize database
    init_db()
    
    # Connect to monitoring system
    await api_monitor.connect()
    
    logger.info("Application started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    # Close monitoring system
    await api_monitor.close()
    
    logger.info("Application shut down successfully")

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = security_manager.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return security_manager.create_user_token(user)

@app.get("/health")
async def health_check():
    # Get system metrics
    metrics = await api_monitor.get_metrics()
    
    return {
        "status": "healthy",
        "service": "relationship-assessment-system",
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": metrics
    }

@app.get("/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user