# Relationship Assessment System

A multi-agent system for comprehensive relationship assessment and analysis.

## System Overview

This system provides a modular approach to relationship assessment using distributed agent architecture for data collection, analysis, and recommendation generation.

## Features

- Multi-module relationship assessment
- Distributed agent architecture
- Comprehensive data collection and analysis
- Secure data handling
- Real-time processing
- Standardized API responses

## Architecture

The system consists of several key components:

1. Data Collection Agent
2. Expert Agents
   - Relationship Psychologist
   - Behavioral Psychologist
3. Analysis Agents
   - Demographic Analyzer
   - Attachment Analyzer
   - Communication Analyzer
   - Family Systems Analyzer
   - Pattern Integrator
   - Cultural Religious Analyzer

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Run the application:
```bash
uvicorn src.api.main:app --reload
```

## API Documentation

Once the application is running, visit `/docs` for the OpenAPI documentation.

## Testing

Run tests using pytest:
```bash
pytest tests/
```

## Security

- End-to-end encryption
- Secure authentication
- Role-based access control
- Data anonymization
- Comprehensive audit logging

## License

MIT License