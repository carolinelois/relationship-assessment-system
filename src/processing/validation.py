from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, validator, Field
from enum import Enum
from ..models.schemas import (
    ModuleType, Response, Session, Analysis,
    RelationshipProfile, RelationshipAssessment
)

class ValidationLevel(str, Enum):
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"

class ValidationResult(BaseModel):
    is_valid: bool
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    validation_level: ValidationLevel
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class DataValidator:
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        self.validation_level = validation_level
        self.error_threshold = self._get_error_threshold()
        self.warning_threshold = self._get_warning_threshold()

    def validate_session_data(
        self, session: Session, responses: List[Response]
    ) -> ValidationResult:
        errors = []
        warnings = []
        
        # Validate session
        session_errors, session_warnings = self._validate_session(session)
        errors.extend(session_errors)
        warnings.extend(session_warnings)
        
        # Validate responses
        response_errors, response_warnings = self._validate_responses(
            responses, session
        )
        errors.extend(response_errors)
        warnings.extend(response_warnings)
        
        # Check module completion
        completion_errors, completion_warnings = self._check_module_completion(
            session, responses
        )
        errors.extend(completion_errors)
        warnings.extend(completion_warnings)
        
        # Determine validity based on validation level
        is_valid = self._determine_validity(len(errors), len(warnings))
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            validation_level=self.validation_level
        )

    def validate_analysis_data(
        self, analysis: Analysis
    ) -> ValidationResult:
        errors = []
        warnings = []
        
        # Validate analysis structure
        structure_errors, structure_warnings = self._validate_analysis_structure(
            analysis
        )
        errors.extend(structure_errors)
        warnings.extend(structure_warnings)
        
        # Validate metrics
        metric_errors, metric_warnings = self._validate_metrics(
            analysis.results
        )
        errors.extend(metric_errors)
        warnings.extend(metric_warnings)
        
        # Validate recommendations
        rec_errors, rec_warnings = self._validate_recommendations(
            analysis.recommendations
        )
        errors.extend(rec_errors)
        warnings.extend(rec_warnings)
        
        # Determine validity based on validation level
        is_valid = self._determine_validity(len(errors), len(warnings))
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            validation_level=self.validation_level
        )

    def validate_profile_data(
        self, profile: RelationshipProfile
    ) -> ValidationResult:
        errors = []
        warnings = []
        
        # Validate profile structure
        structure_errors, structure_warnings = self._validate_profile_structure(
            profile
        )
        errors.extend(structure_errors)
        warnings.extend(structure_warnings)
        
        # Validate demographics
        demo_errors, demo_warnings = self._validate_demographics(
            profile.demographics
        )
        errors.extend(demo_errors)
        warnings.extend(demo_warnings)
        
        # Validate patterns
        pattern_errors, pattern_warnings = self._validate_patterns(
            profile.family_patterns
        )
        errors.extend(pattern_errors)
        warnings.extend(pattern_warnings)
        
        # Determine validity based on validation level
        is_valid = self._determine_validity(len(errors), len(warnings))
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            validation_level=self.validation_level
        )

    def _get_error_threshold(self) -> int:
        if self.validation_level == ValidationLevel.STRICT:
            return 0
        elif self.validation_level == ValidationLevel.MODERATE:
            return 3
        else:  # LENIENT
            return 5

    def _get_warning_threshold(self) -> int:
        if self.validation_level == ValidationLevel.STRICT:
            return 3
        elif self.validation_level == ValidationLevel.MODERATE:
            return 5
        else:  # LENIENT
            return 10

    def _validate_session(
        self, session: Session
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        errors = []
        warnings = []
        
        # Validate session ID
        if not session.session_id:
            errors.append({
                "type": "session_error",
                "field": "session_id",
                "message": "Session ID is required"
            })
        
        # Validate module type
        if not session.module:
            errors.append({
                "type": "session_error",
                "field": "module",
                "message": "Module type is required"
            })
        elif session.module not in ModuleType.__members__:
            errors.append({
                "type": "session_error",
                "field": "module",
                "message": f"Invalid module type: {session.module}"
            })
        
        # Validate timestamps
        if session.end_time and session.start_time > session.end_time:
            errors.append({
                "type": "session_error",
                "field": "timestamps",
                "message": "End time cannot be before start time"
            })
        
        return errors, warnings

    def _validate_responses(
        self,
        responses: List[Response],
        session: Session
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        errors = []
        warnings = []
        
        # Check for empty responses
        if not responses:
            errors.append({
                "type": "response_error",
                "field": "responses",
                "message": "No responses provided"
            })
            return errors, warnings
        
        # Validate each response
        for response in responses:
            # Check response belongs to session
            if response.session_id != session.session_id:
                errors.append({
                    "type": "response_error",
                    "field": "session_id",
                    "message": f"Response {response.response_id} does not match session"
                })
            
            # Validate response value if present
            if response.response_value is not None:
                if not 0 <= response.response_value <= 1:
                    errors.append({
                        "type": "response_error",
                        "field": "response_value",
                        "message": f"Invalid response value in {response.response_id}"
                    })
            
            # Check for empty response text
            if not response.response_text.strip():
                warnings.append({
                    "type": "response_warning",
                    "field": "response_text",
                    "message": f"Empty response text in {response.response_id}"
                })
        
        return errors, warnings

    def _check_module_completion(
        self,
        session: Session,
        responses: List[Response]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        errors = []
        warnings = []
        
        # Get required questions for module
        required_questions = self._get_required_questions(session.module)
        
        # Check for missing required responses
        answered_questions = {r.question_id for r in responses}
        missing_questions = required_questions - answered_questions
        
        if missing_questions:
            if self.validation_level == ValidationLevel.STRICT:
                errors.append({
                    "type": "completion_error",
                    "field": "responses",
                    "message": f"Missing required questions: {missing_questions}"
                })
            else:
                warnings.append({
                    "type": "completion_warning",
                    "field": "responses",
                    "message": f"Missing recommended questions: {missing_questions}"
                })
        
        return errors, warnings

    def _validate_analysis_structure(
        self, analysis: Analysis
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        errors = []
        warnings = []
        
        # Check required fields
        if not analysis.results:
            errors.append({
                "type": "analysis_error",
                "field": "results",
                "message": "Analysis results are required"
            })
        
        # Validate confidence score
        if not 0 <= analysis.confidence_score <= 1:
            errors.append({
                "type": "analysis_error",
                "field": "confidence_score",
                "message": "Invalid confidence score"
            })
        elif analysis.confidence_score < 0.5:
            warnings.append({
                "type": "analysis_warning",
                "field": "confidence_score",
                "message": "Low confidence score"
            })
        
        return errors, warnings

    def _validate_metrics(
        self, results: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        errors = []
        warnings = []
        
        required_metrics = {
            "relationship_health",
            "pattern_consistency",
            "stability_score"
        }
        
        # Check for required metrics
        for metric in required_metrics:
            if metric not in results:
                errors.append({
                    "type": "metric_error",
                    "field": metric,
                    "message": f"Required metric {metric} is missing"
                })
            elif not isinstance(results[metric], (int, float)):
                errors.append({
                    "type": "metric_error",
                    "field": metric,
                    "message": f"Invalid value type for metric {metric}"
                })
            elif not 0 <= results[metric] <= 1:
                errors.append({
                    "type": "metric_error",
                    "field": metric,
                    "message": f"Invalid value range for metric {metric}"
                })
        
        return errors, warnings

    def _validate_recommendations(
        self, recommendations: List[str]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        errors = []
        warnings = []
        
        # Check for empty recommendations
        if not recommendations:
            warnings.append({
                "type": "recommendation_warning",
                "field": "recommendations",
                "message": "No recommendations provided"
            })
            return errors, warnings
        
        # Validate each recommendation
        for i, rec in enumerate(recommendations):
            if not rec.strip():
                errors.append({
                    "type": "recommendation_error",
                    "field": f"recommendation_{i}",
                    "message": "Empty recommendation"
                })
            elif len(rec) < 10:
                warnings.append({
                    "type": "recommendation_warning",
                    "field": f"recommendation_{i}",
                    "message": "Recommendation too short"
                })
        
        return errors, warnings

    def _validate_profile_structure(
        self, profile: RelationshipProfile
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        errors = []
        warnings = []
        
        # Validate required fields
        required_fields = [
            "demographics",
            "attachment_style",
            "communication_style",
            "conflict_style"
        ]
        
        for field in required_fields:
            if not getattr(profile, field):
                errors.append({
                    "type": "profile_error",
                    "field": field,
                    "message": f"Required field {field} is missing"
                })
        
        return errors, warnings

    def _validate_demographics(
        self, demographics: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        errors = []
        warnings = []
        
        required_fields = [
            "cultural_religious",
            "socioeconomic",
            "family_structure"
        ]
        
        for field in required_fields:
            if field not in demographics:
                if self.validation_level == ValidationLevel.STRICT:
                    errors.append({
                        "type": "demographics_error",
                        "field": field,
                        "message": f"Required demographic field {field} is missing"
                    })
                else:
                    warnings.append({
                        "type": "demographics_warning",
                        "field": field,
                        "message": f"Recommended demographic field {field} is missing"
                    })
        
        return errors, warnings

    def _validate_patterns(
        self, patterns: List[str]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        errors = []
        warnings = []
        
        if not patterns:
            warnings.append({
                "type": "pattern_warning",
                "field": "patterns",
                "message": "No patterns identified"
            })
        
        return errors, warnings

    def _get_required_questions(self, module: ModuleType) -> set:
        # Define required questions for each module
        required_questions = {
            ModuleType.DEMOGRAPHICS: {
                "cultural_background",
                "religious_beliefs",
                "socioeconomic_status"
            },
            ModuleType.FAMILY_ORIGIN: {
                "attachment_style",
                "family_dynamics",
                "childhood_experiences"
            },
            ModuleType.CORE_RELATIONSHIP: {
                "communication_style",
                "conflict_resolution",
                "emotional_intimacy"
            },
            ModuleType.FAMILY_CREATION: {
                "family_planning",
                "parenting_style",
                "work_life_balance"
            }
        }
        
        return required_questions.get(module, set())

    def _determine_validity(self, error_count: int, warning_count: int) -> bool:
        if error_count > self.error_threshold:
            return False
        
        if self.validation_level == ValidationLevel.STRICT and warning_count > self.warning_threshold:
            return False
        
        return True