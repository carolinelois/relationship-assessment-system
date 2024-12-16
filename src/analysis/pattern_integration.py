from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum
from .attachment_analysis import AttachmentStyle
from .communication_analysis import CommunicationStyle, ConflictStyle

class RelationshipStyle(str, Enum):
    SECURE_BALANCED = "secure_balanced"
    GROWTH_ORIENTED = "growth_oriented"
    SUPPORT_FOCUSED = "support_focused"
    INDEPENDENT = "independent"
    VOLATILE = "volatile"
    DISCONNECTED = "disconnected"

@dataclass
class IntegrationMetrics:
    pattern_consistency: float
    growth_potential: float
    relationship_health: float
    risk_level: float
    stability_score: float

class PatternIntegrator:
    def __init__(self):
        self.pattern_weights = {
            "attachment": 0.3,
            "communication": 0.3,
            "conflict": 0.2,
            "support": 0.2
        }

    def integrate_patterns(
        self,
        attachment_data: Dict[str, Any],
        communication_data: Dict[str, Any],
        behavioral_data: Dict[str, Any],
        demographic_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        # Calculate integrated metrics
        metrics = self._calculate_integration_metrics(
            attachment_data,
            communication_data,
            behavioral_data
        )
        
        # Determine relationship style
        style = self._determine_relationship_style(
            attachment_data.get("attachment_style"),
            communication_data.get("communication_style"),
            communication_data.get("conflict_style"),
            metrics
        )
        
        # Generate comprehensive analysis
        analysis = self._generate_integrated_analysis(
            style,
            metrics,
            attachment_data,
            communication_data,
            behavioral_data,
            demographic_data
        )
        
        return analysis

    def _calculate_integration_metrics(
        self,
        attachment_data: Dict[str, Any],
        communication_data: Dict[str, Any],
        behavioral_data: Dict[str, Any]
    ) -> IntegrationMetrics:
        # Calculate pattern consistency
        pattern_consistency = self._calculate_pattern_consistency(
            attachment_data,
            communication_data,
            behavioral_data
        )
        
        # Calculate growth potential
        growth_potential = self._calculate_growth_potential(
            attachment_data,
            communication_data,
            behavioral_data
        )
        
        # Calculate relationship health
        relationship_health = self._calculate_relationship_health(
            attachment_data,
            communication_data,
            behavioral_data
        )
        
        # Calculate risk level
        risk_level = self._calculate_risk_level(
            attachment_data,
            communication_data,
            behavioral_data
        )
        
        # Calculate stability score
        stability_score = self._calculate_stability_score(
            attachment_data,
            communication_data,
            behavioral_data
        )
        
        return IntegrationMetrics(
            pattern_consistency=pattern_consistency,
            growth_potential=growth_potential,
            relationship_health=relationship_health,
            risk_level=risk_level,
            stability_score=stability_score
        )

    def _calculate_pattern_consistency(
        self,
        attachment_data: Dict[str, Any],
        communication_data: Dict[str, Any],
        behavioral_data: Dict[str, Any]
    ) -> float:
        patterns = []
        
        # Collect relevant scores
        if "metrics" in attachment_data:
            patterns.append(attachment_data["metrics"].get("consistency_score", 0))
        
        if "metrics" in communication_data:
            patterns.append(communication_data["metrics"].get("nonverbal_congruence", 0))
        
        if "patterns" in behavioral_data:
            pattern_scores = [p.get("consistency", 0) for p in behavioral_data["patterns"]]
            patterns.extend(pattern_scores)
        
        return np.mean(patterns) if patterns else 0.0

    def _calculate_growth_potential(
        self,
        attachment_data: Dict[str, Any],
        communication_data: Dict[str, Any],
        behavioral_data: Dict[str, Any]
    ) -> float:
        factors = []
        
        # Attachment factors
        if attachment_data.get("attachment_style") == AttachmentStyle.SECURE:
            factors.append(0.8)
        
        # Communication factors
        if communication_data.get("communication_style") == CommunicationStyle.ASSERTIVE:
            factors.append(0.8)
        
        # Behavioral factors
        adaptability = behavioral_data.get("adaptability_score", 0)
        if adaptability:
            factors.append(adaptability)
        
        return np.mean(factors) if factors else 0.5

    def _calculate_relationship_health(
        self,
        attachment_data: Dict[str, Any],
        communication_data: Dict[str, Any],
        behavioral_data: Dict[str, Any]
    ) -> float:
        scores = []
        
        # Attachment health indicators
        if "metrics" in attachment_data:
            scores.append(attachment_data["metrics"].get("security_score", 0))
        
        # Communication health indicators
        if "metrics" in communication_data:
            scores.append(communication_data["metrics"].get("clarity", 0))
            scores.append(communication_data["metrics"].get("emotional_awareness", 0))
        
        # Behavioral health indicators
        if "health_indicators" in behavioral_data:
            scores.extend(behavioral_data["health_indicators"].values())
        
        return np.mean(scores) if scores else 0.0

    def _calculate_risk_level(
        self,
        attachment_data: Dict[str, Any],
        communication_data: Dict[str, Any],
        behavioral_data: Dict[str, Any]
    ) -> float:
        risk_factors = []
        
        # Attachment risk factors
        if attachment_data.get("attachment_style") == AttachmentStyle.DISORGANIZED:
            risk_factors.append(0.8)
        
        # Communication risk factors
        if communication_data.get("communication_style") == CommunicationStyle.AGGRESSIVE:
            risk_factors.append(0.7)
        
        # Behavioral risk factors
        if "risk_factors" in behavioral_data:
            risk_scores = [r.get("severity", 0) for r in behavioral_data["risk_factors"]]
            risk_factors.extend(risk_scores)
        
        return np.mean(risk_factors) if risk_factors else 0.0

    def _calculate_stability_score(
        self,
        attachment_data: Dict[str, Any],
        communication_data: Dict[str, Any],
        behavioral_data: Dict[str, Any]
    ) -> float:
        factors = []
        
        # Attachment stability
        if "metrics" in attachment_data:
            factors.append(attachment_data["metrics"].get("consistency_score", 0))
        
        # Communication stability
        if "metrics" in communication_data:
            factors.append(communication_data["metrics"].get("conflict_resolution", 0))
        
        # Behavioral stability
        if "stability_indicators" in behavioral_data:
            factors.extend(behavioral_data["stability_indicators"].values())
        
        return np.mean(factors) if factors else 0.0

    def _determine_relationship_style(
        self,
        attachment_style: AttachmentStyle,
        communication_style: CommunicationStyle,
        conflict_style: ConflictStyle,
        metrics: IntegrationMetrics
    ) -> RelationshipStyle:
        # Define thresholds
        HIGH_THRESHOLD = 0.7
        LOW_THRESHOLD = 0.3
        
        # Check for secure balanced style
        if (attachment_style == AttachmentStyle.SECURE and
            communication_style == CommunicationStyle.ASSERTIVE and
            metrics.stability_score > HIGH_THRESHOLD):
            return RelationshipStyle.SECURE_BALANCED
        
        # Check for growth oriented style
        if (metrics.growth_potential > HIGH_THRESHOLD and
            metrics.risk_level < LOW_THRESHOLD):
            return RelationshipStyle.GROWTH_ORIENTED
        
        # Check for support focused style
        if (communication_style == CommunicationStyle.ASSERTIVE and
            conflict_style == ConflictStyle.COLLABORATIVE):
            return RelationshipStyle.SUPPORT_FOCUSED
        
        # Check for independent style
        if (attachment_style == AttachmentStyle.AVOIDANT and
            metrics.stability_score > HIGH_THRESHOLD):
            return RelationshipStyle.INDEPENDENT
        
        # Check for volatile style
        if (metrics.pattern_consistency < LOW_THRESHOLD and
            metrics.risk_level > HIGH_THRESHOLD):
            return RelationshipStyle.VOLATILE
        
        # Check for disconnected style
        if (metrics.relationship_health < LOW_THRESHOLD and
            metrics.stability_score < LOW_THRESHOLD):
            return RelationshipStyle.DISCONNECTED
        
        # Default to growth oriented if no clear pattern
        return RelationshipStyle.GROWTH_ORIENTED

    def _generate_integrated_analysis(
        self,
        style: RelationshipStyle,
        metrics: IntegrationMetrics,
        attachment_data: Dict[str, Any],
        communication_data: Dict[str, Any],
        behavioral_data: Dict[str, Any],
        demographic_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        analysis = {
            "relationship_style": style,
            "metrics": {
                "pattern_consistency": metrics.pattern_consistency,
                "growth_potential": metrics.growth_potential,
                "relationship_health": metrics.relationship_health,
                "risk_level": metrics.risk_level,
                "stability_score": metrics.stability_score
            },
            "patterns": self._identify_integrated_patterns(
                attachment_data,
                communication_data,
                behavioral_data
            ),
            "strengths": self._identify_integrated_strengths(
                style,
                metrics,
                attachment_data,
                communication_data,
                behavioral_data
            ),
            "challenges": self._identify_integrated_challenges(
                style,
                metrics,
                attachment_data,
                communication_data,
                behavioral_data
            ),
            "recommendations": self._generate_integrated_recommendations(
                style,
                metrics,
                attachment_data,
                communication_data,
                behavioral_data
            ),
            "risk_factors": self._identify_risk_factors(
                attachment_data,
                communication_data,
                behavioral_data
            )
        }
        
        if demographic_data:
            analysis["contextual_factors"] = self._analyze_contextual_factors(
                demographic_data
            )
        
        return analysis

    def _identify_integrated_patterns(
        self,
        attachment_data: Dict[str, Any],
        communication_data: Dict[str, Any],
        behavioral_data: Dict[str, Any]
    ) -> List[str]:
        patterns = []
        
        # Combine patterns from different analyses
        if "patterns" in attachment_data:
            patterns.extend(attachment_data["patterns"])
        
        if "patterns" in communication_data:
            patterns.extend(communication_data["patterns"])
        
        if "patterns" in behavioral_data:
            patterns.extend(behavioral_data["patterns"])
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(patterns))

    def _identify_integrated_strengths(
        self,
        style: RelationshipStyle,
        metrics: IntegrationMetrics,
        attachment_data: Dict[str, Any],
        communication_data: Dict[str, Any],
        behavioral_data: Dict[str, Any]
    ) -> List[str]:
        strengths = []
        
        # Style-based strengths
        if style == RelationshipStyle.SECURE_BALANCED:
            strengths.extend([
                "Strong foundation of security and trust",
                "Balanced approach to relationship needs",
                "Effective communication and problem-solving"
            ])
        elif style == RelationshipStyle.GROWTH_ORIENTED:
            strengths.extend([
                "Strong commitment to personal and relationship growth",
                "Ability to learn from challenges",
                "Open to feedback and change"
            ])
        
        # Metrics-based strengths
        if metrics.pattern_consistency > 0.7:
            strengths.append("Consistent relationship patterns")
        if metrics.growth_potential > 0.7:
            strengths.append("High potential for growth and development")
        
        # Combine strengths from different analyses
        if "strengths" in attachment_data:
            strengths.extend(attachment_data["strengths"])
        if "strengths" in communication_data:
            strengths.extend(communication_data["strengths"])
        if "strengths" in behavioral_data:
            strengths.extend(behavioral_data["strengths"])
        
        return list(dict.fromkeys(strengths))

    def _identify_integrated_challenges(
        self,
        style: RelationshipStyle,
        metrics: IntegrationMetrics,
        attachment_data: Dict[str, Any],
        communication_data: Dict[str, Any],
        behavioral_data: Dict[str, Any]
    ) -> List[str]:
        challenges = []
        
        # Style-based challenges
        if style == RelationshipStyle.VOLATILE:
            challenges.extend([
                "Difficulty maintaining stability",
                "High emotional reactivity",
                "Inconsistent relationship patterns"
            ])
        elif style == RelationshipStyle.DISCONNECTED:
            challenges.extend([
                "Limited emotional connection",
                "Difficulty with intimacy",
                "Poor communication patterns"
            ])
        
        # Metrics-based challenges
        if metrics.risk_level > 0.7:
            challenges.append("High level of relationship risk factors")
        if metrics.stability_score < 0.3:
            challenges.append("Unstable relationship patterns")
        
        # Combine challenges from different analyses
        if "challenges" in attachment_data:
            challenges.extend(attachment_data["challenges"])
        if "challenges" in communication_data:
            challenges.extend(communication_data["challenges"])
        if "challenges" in behavioral_data:
            challenges.extend(behavioral_data["challenges"])
        
        return list(dict.fromkeys(challenges))

    def _generate_integrated_recommendations(
        self,
        style: RelationshipStyle,
        metrics: IntegrationMetrics,
        attachment_data: Dict[str, Any],
        communication_data: Dict[str, Any],
        behavioral_data: Dict[str, Any]
    ) -> List[str]:
        recommendations = []
        
        # Style-based recommendations
        if style == RelationshipStyle.VOLATILE:
            recommendations.extend([
                "Focus on developing relationship stability",
                "Practice emotional regulation techniques",
                "Establish consistent communication patterns"
            ])
        elif style == RelationshipStyle.DISCONNECTED:
            recommendations.extend([
                "Work on building emotional connection",
                "Increase quality time together",
                "Practice active listening and sharing"
            ])
        
        # Metrics-based recommendations
        if metrics.risk_level > 0.5:
            recommendations.append(
                "Address identified risk factors with professional support"
            )
        if metrics.stability_score < 0.5:
            recommendations.append(
                "Focus on building consistent relationship patterns"
            )
        
        # Combine recommendations from different analyses
        if "recommendations" in attachment_data:
            recommendations.extend(attachment_data["recommendations"])
        if "recommendations" in communication_data:
            recommendations.extend(communication_data["recommendations"])
        if "recommendations" in behavioral_data:
            recommendations.extend(behavioral_data["recommendations"])
        
        return list(dict.fromkeys(recommendations))

    def _identify_risk_factors(
        self,
        attachment_data: Dict[str, Any],
        communication_data: Dict[str, Any],
        behavioral_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        risk_factors = []
        
        # Attachment-based risks
        if attachment_data.get("attachment_style") == AttachmentStyle.DISORGANIZED:
            risk_factors.append({
                "type": "attachment",
                "description": "Disorganized attachment pattern",
                "severity": 0.8,
                "impact": "High impact on relationship stability"
            })
        
        # Communication-based risks
        if communication_data.get("communication_style") == CommunicationStyle.AGGRESSIVE:
            risk_factors.append({
                "type": "communication",
                "description": "Aggressive communication pattern",
                "severity": 0.7,
                "impact": "High impact on relationship safety"
            })
        
        # Behavioral risks
        if "risk_factors" in behavioral_data:
            risk_factors.extend(behavioral_data["risk_factors"])
        
        return risk_factors

    def _analyze_contextual_factors(
        self, demographic_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "cultural_factors": self._analyze_cultural_impact(demographic_data),
            "life_stage_factors": self._analyze_life_stage_impact(demographic_data),
            "environmental_factors": self._analyze_environmental_impact(demographic_data)
        }

    def _analyze_cultural_impact(
        self, demographic_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "cultural_alignment": demographic_data.get("cultural_alignment", "unknown"),
            "value_system_compatibility": demographic_data.get("value_compatibility", "unknown"),
            "cultural_challenges": demographic_data.get("cultural_challenges", [])
        }

    def _analyze_life_stage_impact(
        self, demographic_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "life_stage": demographic_data.get("life_stage", "unknown"),
            "developmental_tasks": demographic_data.get("developmental_tasks", []),
            "life_transitions": demographic_data.get("life_transitions", [])
        }

    def _analyze_environmental_impact(
        self, demographic_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "support_system": demographic_data.get("support_system", "unknown"),
            "stress_factors": demographic_data.get("stress_factors", []),
            "resources": demographic_data.get("resources", [])
        }