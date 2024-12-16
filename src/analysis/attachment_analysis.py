from typing import Dict, List, Any, Tuple
import numpy as np
from enum import Enum
from dataclasses import dataclass

class AttachmentStyle(str, Enum):
    SECURE = "secure"
    ANXIOUS = "anxious"
    AVOIDANT = "avoidant"
    DISORGANIZED = "disorganized"

@dataclass
class AttachmentMetrics:
    anxiety_score: float
    avoidance_score: float
    security_score: float
    consistency_score: float

class AttachmentAnalyzer:
    def __init__(self):
        self.anxiety_weights = {
            "separation_anxiety": 0.3,
            "fear_abandonment": 0.3,
            "need_reassurance": 0.2,
            "self_worth": 0.2
        }
        
        self.avoidance_weights = {
            "comfort_intimacy": 0.3,
            "trust_others": 0.3,
            "emotional_expression": 0.2,
            "independence": 0.2
        }
        
        self.security_weights = {
            "stable_relationships": 0.3,
            "emotional_regulation": 0.3,
            "self_confidence": 0.2,
            "trust_building": 0.2
        }

    def analyze_attachment_pattern(
        self, responses: List[Dict[str, Any]]
    ) -> Tuple[AttachmentStyle, AttachmentMetrics, Dict[str, Any]]:
        # Calculate core metrics
        metrics = self._calculate_metrics(responses)
        
        # Determine primary attachment style
        style = self._determine_attachment_style(metrics)
        
        # Generate detailed analysis
        analysis = self._generate_analysis(style, metrics, responses)
        
        return style, metrics, analysis

    def _calculate_metrics(self, responses: List[Dict[str, Any]]) -> AttachmentMetrics:
        # Calculate anxiety score
        anxiety_score = self._calculate_weighted_score(
            responses, "anxiety", self.anxiety_weights
        )
        
        # Calculate avoidance score
        avoidance_score = self._calculate_weighted_score(
            responses, "avoidance", self.avoidance_weights
        )
        
        # Calculate security score
        security_score = self._calculate_weighted_score(
            responses, "security", self.security_weights
        )
        
        # Calculate consistency score
        consistency_score = self._calculate_consistency_score(responses)
        
        return AttachmentMetrics(
            anxiety_score=anxiety_score,
            avoidance_score=avoidance_score,
            security_score=security_score,
            consistency_score=consistency_score
        )

    def _calculate_weighted_score(
        self, responses: List[Dict[str, Any]], dimension: str, weights: Dict[str, float]
    ) -> float:
        scores = []
        total_weight = 0
        
        for category, weight in weights.items():
            category_responses = [
                r for r in responses if r.get("category") == category
            ]
            if category_responses:
                category_score = np.mean([
                    r.get("value", 0) for r in category_responses
                ])
                scores.append(category_score * weight)
                total_weight += weight
        
        return sum(scores) / total_weight if total_weight > 0 else 0

    def _calculate_consistency_score(self, responses: List[Dict[str, Any]]) -> float:
        # Group responses by category
        category_responses = {}
        for response in responses:
            category = response.get("category")
            if category:
                if category not in category_responses:
                    category_responses[category] = []
                category_responses[category].append(response.get("value", 0))
        
        # Calculate variance within each category
        variances = []
        for responses in category_responses.values():
            if len(responses) > 1:
                variances.append(np.var(responses))
        
        # Convert variance to consistency score (inverse relationship)
        if variances:
            mean_variance = np.mean(variances)
            consistency_score = 1 / (1 + mean_variance)
            return consistency_score
        return 1.0

    def _determine_attachment_style(self, metrics: AttachmentMetrics) -> AttachmentStyle:
        # Define thresholds
        HIGH_THRESHOLD = 0.7
        LOW_THRESHOLD = 0.3
        
        # Check for disorganized attachment
        if (metrics.anxiety_score > HIGH_THRESHOLD and 
            metrics.avoidance_score > HIGH_THRESHOLD):
            return AttachmentStyle.DISORGANIZED
        
        # Check for secure attachment
        if (metrics.security_score > HIGH_THRESHOLD and 
            metrics.consistency_score > HIGH_THRESHOLD):
            return AttachmentStyle.SECURE
        
        # Check for anxious attachment
        if metrics.anxiety_score > HIGH_THRESHOLD:
            return AttachmentStyle.ANXIOUS
        
        # Check for avoidant attachment
        if metrics.avoidance_score > HIGH_THRESHOLD:
            return AttachmentStyle.AVOIDANT
        
        # Default to secure if no clear pattern emerges
        return AttachmentStyle.SECURE

    def _generate_analysis(
        self, 
        style: AttachmentStyle, 
        metrics: AttachmentMetrics, 
        responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        analysis = {
            "attachment_style": style,
            "metrics": {
                "anxiety_score": metrics.anxiety_score,
                "avoidance_score": metrics.avoidance_score,
                "security_score": metrics.security_score,
                "consistency_score": metrics.consistency_score
            },
            "patterns": self._identify_patterns(responses),
            "strengths": self._identify_strengths(style, metrics),
            "challenges": self._identify_challenges(style, metrics),
            "recommendations": self._generate_recommendations(style, metrics)
        }
        
        return analysis

    def _identify_patterns(self, responses: List[Dict[str, Any]]) -> List[str]:
        patterns = []
        
        # Analyze response patterns
        high_anxiety_responses = [
            r for r in responses 
            if r.get("dimension") == "anxiety" and r.get("value", 0) > 0.7
        ]
        
        high_avoidance_responses = [
            r for r in responses 
            if r.get("dimension") == "avoidance" and r.get("value", 0) > 0.7
        ]
        
        # Identify specific patterns
        if high_anxiety_responses:
            patterns.append("Shows heightened sensitivity to rejection")
        
        if high_avoidance_responses:
            patterns.append("Tends to maintain emotional distance")
        
        # Add more pattern identification logic here
        
        return patterns

    def _identify_strengths(
        self, style: AttachmentStyle, metrics: AttachmentMetrics
    ) -> List[str]:
        strengths = []
        
        if style == AttachmentStyle.SECURE:
            strengths.extend([
                "Able to form stable relationships",
                "Good emotional regulation",
                "Healthy balance of independence and intimacy"
            ])
        
        elif style == AttachmentStyle.ANXIOUS:
            strengths.extend([
                "Strong capacity for emotional connection",
                "Highly attuned to others' needs",
                "Values relationships deeply"
            ])
        
        elif style == AttachmentStyle.AVOIDANT:
            strengths.extend([
                "Strong sense of independence",
                "Self-reliant",
                "Values personal growth"
            ])
        
        if metrics.consistency_score > 0.7:
            strengths.append("Shows consistent relationship patterns")
        
        return strengths

    def _identify_challenges(
        self, style: AttachmentStyle, metrics: AttachmentMetrics
    ) -> List[str]:
        challenges = []
        
        if style == AttachmentStyle.ANXIOUS:
            challenges.extend([
                "May struggle with fear of abandonment",
                "Could benefit from developing more independence",
                "May need to work on self-validation"
            ])
        
        elif style == AttachmentStyle.AVOIDANT:
            challenges.extend([
                "May struggle with emotional intimacy",
                "Could benefit from increased vulnerability",
                "May need to work on trust-building"
            ])
        
        elif style == AttachmentStyle.DISORGANIZED:
            challenges.extend([
                "May experience conflicting relationship needs",
                "Could benefit from developing consistent coping strategies",
                "May need support in building relationship stability"
            ])
        
        if metrics.consistency_score < 0.3:
            challenges.append("Shows inconsistent relationship patterns")
        
        return challenges

    def _generate_recommendations(
        self, style: AttachmentStyle, metrics: AttachmentMetrics
    ) -> List[str]:
        recommendations = []
        
        # General recommendations
        recommendations.append(
            "Practice self-awareness and mindfulness in relationships"
        )
        
        # Style-specific recommendations
        if style == AttachmentStyle.ANXIOUS:
            recommendations.extend([
                "Work on self-soothing techniques",
                "Develop independent interests and activities",
                "Practice setting healthy boundaries"
            ])
        
        elif style == AttachmentStyle.AVOIDANT:
            recommendations.extend([
                "Practice expressing emotions and needs",
                "Work on gradual trust-building exercises",
                "Challenge avoidance patterns gently"
            ])
        
        elif style == AttachmentStyle.DISORGANIZED:
            recommendations.extend([
                "Seek professional support for attachment work",
                "Develop consistent self-care routines",
                "Work on emotional regulation skills"
            ])
        
        # Metrics-based recommendations
        if metrics.consistency_score < 0.5:
            recommendations.append(
                "Work on developing more consistent relationship patterns"
            )
        
        if metrics.security_score < 0.5:
            recommendations.append(
                "Focus on building secure attachment through small, consistent steps"
            )
        
        return recommendations