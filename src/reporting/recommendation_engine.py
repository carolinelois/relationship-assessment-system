from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from ..analysis.pattern_integration import RelationshipStyle
from ..analysis.attachment_analysis import AttachmentStyle
from ..analysis.communication_analysis import CommunicationStyle, ConflictStyle

class RecommendationType(str, Enum):
    BEHAVIORAL = "behavioral"
    EMOTIONAL = "emotional"
    COGNITIVE = "cognitive"
    RELATIONAL = "relational"
    THERAPEUTIC = "therapeutic"
    EDUCATIONAL = "educational"

class PriorityLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MAINTENANCE = "maintenance"

@dataclass
class Recommendation:
    type: RecommendationType
    priority: PriorityLevel
    description: str
    rationale: str
    action_steps: List[str]
    resources: Optional[List[str]] = None
    timeframe: Optional[str] = None
    expected_outcomes: Optional[List[str]] = None

class RecommendationEngine:
    def __init__(self):
        self.recommendation_templates = self._initialize_templates()

    def generate_recommendations(
        self, assessment_data: Dict[str, Any]
    ) -> List[Recommendation]:
        recommendations = []
        
        # Generate style-based recommendations
        style_recommendations = self._generate_style_recommendations(
            assessment_data
        )
        recommendations.extend(style_recommendations)
        
        # Generate pattern-based recommendations
        pattern_recommendations = self._generate_pattern_recommendations(
            assessment_data
        )
        recommendations.extend(pattern_recommendations)
        
        # Generate risk-based recommendations
        risk_recommendations = self._generate_risk_recommendations(
            assessment_data
        )
        recommendations.extend(risk_recommendations)
        
        # Prioritize and filter recommendations
        final_recommendations = self._prioritize_recommendations(recommendations)
        
        return final_recommendations

    def _initialize_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "attachment": {
                AttachmentStyle.ANXIOUS: [
                    {
                        "type": RecommendationType.EMOTIONAL,
                        "priority": PriorityLevel.HIGH,
                        "description": "Develop emotional self-regulation",
                        "rationale": "Reduce anxiety in relationships",
                        "action_steps": [
                            "Practice mindfulness meditation",
                            "Learn self-soothing techniques",
                            "Identify anxiety triggers"
                        ]
                    },
                    {
                        "type": RecommendationType.BEHAVIORAL,
                        "priority": PriorityLevel.HIGH,
                        "description": "Build independent activities",
                        "rationale": "Develop secure sense of self",
                        "action_steps": [
                            "Schedule regular solo activities",
                            "Pursue personal interests",
                            "Practice self-care routines"
                        ]
                    }
                ],
                AttachmentStyle.AVOIDANT: [
                    {
                        "type": RecommendationType.RELATIONAL,
                        "priority": PriorityLevel.HIGH,
                        "description": "Increase emotional intimacy",
                        "rationale": "Build closer connections",
                        "action_steps": [
                            "Share feelings regularly",
                            "Practice vulnerability",
                            "Engage in intimacy-building exercises"
                        ]
                    }
                ]
            },
            "communication": {
                CommunicationStyle.AGGRESSIVE: [
                    {
                        "type": RecommendationType.BEHAVIORAL,
                        "priority": PriorityLevel.CRITICAL,
                        "description": "Develop assertive communication",
                        "rationale": "Replace aggressive patterns",
                        "action_steps": [
                            "Learn 'I' statements",
                            "Practice active listening",
                            "Use time-outs when escalated"
                        ]
                    }
                ],
                CommunicationStyle.PASSIVE: [
                    {
                        "type": RecommendationType.BEHAVIORAL,
                        "priority": PriorityLevel.HIGH,
                        "description": "Build assertiveness skills",
                        "rationale": "Express needs effectively",
                        "action_steps": [
                            "Practice stating needs directly",
                            "Set small boundaries",
                            "Express opinions in safe contexts"
                        ]
                    }
                ]
            }
        }

    def _generate_style_recommendations(
        self, assessment_data: Dict[str, Any]
    ) -> List[Recommendation]:
        recommendations = []
        
        # Get styles from assessment
        attachment_style = assessment_data.get("attachment_style")
        communication_style = assessment_data.get("communication_style")
        relationship_style = assessment_data.get("relationship_style")
        
        # Generate attachment style recommendations
        if attachment_style:
            style_templates = self.recommendation_templates["attachment"].get(
                attachment_style, []
            )
            for template in style_templates:
                recommendations.append(
                    Recommendation(**template)
                )
        
        # Generate communication style recommendations
        if communication_style:
            style_templates = self.recommendation_templates["communication"].get(
                communication_style, []
            )
            for template in style_templates:
                recommendations.append(
                    Recommendation(**template)
                )
        
        return recommendations

    def _generate_pattern_recommendations(
        self, assessment_data: Dict[str, Any]
    ) -> List[Recommendation]:
        recommendations = []
        patterns = assessment_data.get("patterns", {})
        
        # Generate recommendations based on identified patterns
        for pattern_type, pattern_data in patterns.items():
            if isinstance(pattern_data, dict):
                strength = pattern_data.get("strength", 0)
                if strength < 0.3:  # Weak pattern
                    recommendations.extend(
                        self._get_pattern_improvement_recommendations(
                            pattern_type, pattern_data
                        )
                    )
                elif strength > 0.7:  # Strong pattern
                    recommendations.extend(
                        self._get_pattern_maintenance_recommendations(
                            pattern_type, pattern_data
                        )
                    )
        
        return recommendations

    def _generate_risk_recommendations(
        self, assessment_data: Dict[str, Any]
    ) -> List[Recommendation]:
        recommendations = []
        risk_factors = assessment_data.get("risk_factors", [])
        
        for risk in risk_factors:
            if risk.get("severity", 0) > 0.7:  # High severity
                recommendations.append(
                    Recommendation(
                        type=RecommendationType.THERAPEUTIC,
                        priority=PriorityLevel.CRITICAL,
                        description=f"Address {risk['type']} risk factor",
                        rationale="Reduce relationship risk",
                        action_steps=self._generate_risk_action_steps(risk),
                        resources=self._get_risk_resources(risk)
                    )
                )
        
        return recommendations

    def _get_pattern_improvement_recommendations(
        self, pattern_type: str, pattern_data: Dict[str, Any]
    ) -> List[Recommendation]:
        recommendations = []
        
        if pattern_type == "emotional_connection":
            recommendations.append(
                Recommendation(
                    type=RecommendationType.EMOTIONAL,
                    priority=PriorityLevel.HIGH,
                    description="Strengthen emotional connection",
                    rationale="Build relationship intimacy",
                    action_steps=[
                        "Schedule daily emotional check-ins",
                        "Practice empathetic listening",
                        "Share feelings and experiences"
                    ]
                )
            )
        
        elif pattern_type == "conflict_resolution":
            recommendations.append(
                Recommendation(
                    type=RecommendationType.BEHAVIORAL,
                    priority=PriorityLevel.HIGH,
                    description="Improve conflict resolution skills",
                    rationale="Develop healthy conflict patterns",
                    action_steps=[
                        "Learn conflict resolution techniques",
                        "Practice active listening during disagreements",
                        "Establish conflict ground rules"
                    ]
                )
            )
        
        return recommendations

    def _get_pattern_maintenance_recommendations(
        self, pattern_type: str, pattern_data: Dict[str, Any]
    ) -> List[Recommendation]:
        recommendations = []
        
        if pattern_type == "communication":
            recommendations.append(
                Recommendation(
                    type=RecommendationType.BEHAVIORAL,
                    priority=PriorityLevel.MAINTENANCE,
                    description="Maintain strong communication",
                    rationale="Preserve relationship strength",
                    action_steps=[
                        "Continue regular check-ins",
                        "Practice active listening",
                        "Express appreciation daily"
                    ]
                )
            )
        
        return recommendations

    def _generate_risk_action_steps(self, risk: Dict[str, Any]) -> List[str]:
        risk_type = risk.get("type", "")
        
        if risk_type == "communication":
            return [
                "Seek professional communication coaching",
                "Learn de-escalation techniques",
                "Practice time-outs when needed"
            ]
        elif risk_type == "attachment":
            return [
                "Consider individual therapy",
                "Read about attachment patterns",
                "Practice self-awareness exercises"
            ]
        else:
            return [
                "Discuss concerns with a professional",
                "Develop awareness of patterns",
                "Create safety plan if needed"
            ]

    def _get_risk_resources(self, risk: Dict[str, Any]) -> List[str]:
        risk_type = risk.get("type", "")
        
        if risk_type == "communication":
            return [
                "Communication skills workbook",
                "Couples communication workshop",
                "Professional counseling resources"
            ]
        elif risk_type == "attachment":
            return [
                "Attachment theory resources",
                "Individual therapy referrals",
                "Self-help books on attachment"
            ]
        else:
            return [
                "Professional counseling resources",
                "Self-help materials",
                "Support group information"
            ]

    def _prioritize_recommendations(
        self, recommendations: List[Recommendation]
    ) -> List[Recommendation]:
        # Sort by priority
        priority_order = {
            PriorityLevel.CRITICAL: 0,
            PriorityLevel.HIGH: 1,
            PriorityLevel.MEDIUM: 2,
            PriorityLevel.LOW: 3,
            PriorityLevel.MAINTENANCE: 4
        }
        
        sorted_recommendations = sorted(
            recommendations,
            key=lambda x: priority_order[x.priority]
        )
        
        # Limit number of recommendations based on priority
        max_recommendations = {
            PriorityLevel.CRITICAL: 3,
            PriorityLevel.HIGH: 5,
            PriorityLevel.MEDIUM: 3,
            PriorityLevel.LOW: 2,
            PriorityLevel.MAINTENANCE: 2
        }
        
        filtered_recommendations = []
        priority_counts = {level: 0 for level in PriorityLevel}
        
        for rec in sorted_recommendations:
            if priority_counts[rec.priority] < max_recommendations[rec.priority]:
                filtered_recommendations.append(rec)
                priority_counts[rec.priority] += 1
        
        return filtered_recommendations