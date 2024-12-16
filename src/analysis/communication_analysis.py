from typing import Dict, List, Any, Tuple
import numpy as np
from enum import Enum
from dataclasses import dataclass

class CommunicationStyle(str, Enum):
    ASSERTIVE = "assertive"
    PASSIVE = "passive"
    AGGRESSIVE = "aggressive"
    PASSIVE_AGGRESSIVE = "passive_aggressive"

class ConflictStyle(str, Enum):
    COLLABORATIVE = "collaborative"
    COMPROMISING = "compromising"
    ACCOMMODATING = "accommodating"
    COMPETING = "competing"
    AVOIDING = "avoiding"

@dataclass
class CommunicationMetrics:
    clarity_score: float
    emotional_awareness_score: float
    listening_score: float
    conflict_resolution_score: float
    nonverbal_congruence_score: float

class CommunicationAnalyzer:
    def __init__(self):
        self.clarity_weights = {
            "message_clarity": 0.3,
            "directness": 0.3,
            "consistency": 0.2,
            "feedback_seeking": 0.2
        }
        
        self.emotional_awareness_weights = {
            "emotion_recognition": 0.3,
            "emotion_expression": 0.3,
            "empathy": 0.2,
            "self_awareness": 0.2
        }
        
        self.listening_weights = {
            "active_listening": 0.3,
            "response_quality": 0.3,
            "attention": 0.2,
            "understanding": 0.2
        }
        
        self.conflict_weights = {
            "problem_solving": 0.3,
            "compromise": 0.2,
            "emotional_regulation": 0.3,
            "resolution_effectiveness": 0.2
        }

    def analyze_communication_pattern(
        self, responses: List[Dict[str, Any]]
    ) -> Tuple[CommunicationStyle, ConflictStyle, CommunicationMetrics, Dict[str, Any]]:
        # Calculate metrics
        metrics = self._calculate_metrics(responses)
        
        # Determine communication styles
        comm_style = self._determine_communication_style(metrics, responses)
        conflict_style = self._determine_conflict_style(metrics, responses)
        
        # Generate analysis
        analysis = self._generate_analysis(
            comm_style, conflict_style, metrics, responses
        )
        
        return comm_style, conflict_style, metrics, analysis

    def _calculate_metrics(
        self, responses: List[Dict[str, Any]]
    ) -> CommunicationMetrics:
        clarity_score = self._calculate_weighted_score(
            responses, "clarity", self.clarity_weights
        )
        
        emotional_awareness_score = self._calculate_weighted_score(
            responses, "emotional_awareness", self.emotional_awareness_weights
        )
        
        listening_score = self._calculate_weighted_score(
            responses, "listening", self.listening_weights
        )
        
        conflict_resolution_score = self._calculate_weighted_score(
            responses, "conflict", self.conflict_weights
        )
        
        nonverbal_congruence_score = self._calculate_nonverbal_congruence(
            responses
        )
        
        return CommunicationMetrics(
            clarity_score=clarity_score,
            emotional_awareness_score=emotional_awareness_score,
            listening_score=listening_score,
            conflict_resolution_score=conflict_resolution_score,
            nonverbal_congruence_score=nonverbal_congruence_score
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

    def _calculate_nonverbal_congruence(
        self, responses: List[Dict[str, Any]]
    ) -> float:
        verbal_responses = [
            r for r in responses if r.get("type") == "verbal"
        ]
        nonverbal_responses = [
            r for r in responses if r.get("type") == "nonverbal"
        ]
        
        if not verbal_responses or not nonverbal_responses:
            return 1.0
        
        verbal_scores = [r.get("value", 0) for r in verbal_responses]
        nonverbal_scores = [r.get("value", 0) for r in nonverbal_responses]
        
        # Calculate correlation between verbal and nonverbal scores
        correlation = np.corrcoef(verbal_scores, nonverbal_scores)[0, 1]
        
        # Convert correlation to a positive score between 0 and 1
        congruence_score = (correlation + 1) / 2
        
        return congruence_score

    def _determine_communication_style(
        self, metrics: CommunicationMetrics, responses: List[Dict[str, Any]]
    ) -> CommunicationStyle:
        # Define thresholds
        HIGH_THRESHOLD = 0.7
        LOW_THRESHOLD = 0.3
        
        # Calculate assertiveness and aggressiveness scores
        assertiveness = np.mean([
            r.get("value", 0) for r in responses 
            if r.get("category") == "assertiveness"
        ])
        
        aggressiveness = np.mean([
            r.get("value", 0) for r in responses 
            if r.get("category") == "aggressiveness"
        ])
        
        # Determine style based on scores
        if assertiveness > HIGH_THRESHOLD and aggressiveness < LOW_THRESHOLD:
            return CommunicationStyle.ASSERTIVE
        
        elif assertiveness < LOW_THRESHOLD and aggressiveness < LOW_THRESHOLD:
            return CommunicationStyle.PASSIVE
        
        elif assertiveness > HIGH_THRESHOLD and aggressiveness > HIGH_THRESHOLD:
            return CommunicationStyle.AGGRESSIVE
        
        elif assertiveness < LOW_THRESHOLD and aggressiveness > LOW_THRESHOLD:
            return CommunicationStyle.PASSIVE_AGGRESSIVE
        
        # Default to assertive if no clear pattern
        return CommunicationStyle.ASSERTIVE

    def _determine_conflict_style(
        self, metrics: CommunicationMetrics, responses: List[Dict[str, Any]]
    ) -> ConflictStyle:
        # Calculate style scores
        style_scores = {
            ConflictStyle.COLLABORATIVE: self._calculate_collaborative_score(responses),
            ConflictStyle.COMPROMISING: self._calculate_compromising_score(responses),
            ConflictStyle.ACCOMMODATING: self._calculate_accommodating_score(responses),
            ConflictStyle.COMPETING: self._calculate_competing_score(responses),
            ConflictStyle.AVOIDING: self._calculate_avoiding_score(responses)
        }
        
        # Return style with highest score
        return max(style_scores.items(), key=lambda x: x[1])[0]

    def _calculate_collaborative_score(
        self, responses: List[Dict[str, Any]]
    ) -> float:
        relevant_responses = [
            r for r in responses 
            if r.get("category") in ["problem_solving", "cooperation"]
        ]
        return np.mean([r.get("value", 0) for r in relevant_responses]) if relevant_responses else 0

    def _calculate_compromising_score(
        self, responses: List[Dict[str, Any]]
    ) -> float:
        relevant_responses = [
            r for r in responses 
            if r.get("category") in ["flexibility", "negotiation"]
        ]
        return np.mean([r.get("value", 0) for r in relevant_responses]) if relevant_responses else 0

    def _calculate_accommodating_score(
        self, responses: List[Dict[str, Any]]
    ) -> float:
        relevant_responses = [
            r for r in responses 
            if r.get("category") in ["yielding", "harmony_seeking"]
        ]
        return np.mean([r.get("value", 0) for r in relevant_responses]) if relevant_responses else 0

    def _calculate_competing_score(
        self, responses: List[Dict[str, Any]]
    ) -> float:
        relevant_responses = [
            r for r in responses 
            if r.get("category") in ["dominance", "winning"]
        ]
        return np.mean([r.get("value", 0) for r in relevant_responses]) if relevant_responses else 0

    def _calculate_avoiding_score(
        self, responses: List[Dict[str, Any]]
    ) -> float:
        relevant_responses = [
            r for r in responses 
            if r.get("category") in ["withdrawal", "conflict_avoidance"]
        ]
        return np.mean([r.get("value", 0) for r in relevant_responses]) if relevant_responses else 0

    def _generate_analysis(
        self,
        comm_style: CommunicationStyle,
        conflict_style: ConflictStyle,
        metrics: CommunicationMetrics,
        responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        analysis = {
            "communication_style": comm_style,
            "conflict_style": conflict_style,
            "metrics": {
                "clarity": metrics.clarity_score,
                "emotional_awareness": metrics.emotional_awareness_score,
                "listening": metrics.listening_score,
                "conflict_resolution": metrics.conflict_resolution_score,
                "nonverbal_congruence": metrics.nonverbal_congruence_score
            },
            "patterns": self._identify_patterns(responses),
            "strengths": self._identify_strengths(comm_style, conflict_style, metrics),
            "challenges": self._identify_challenges(comm_style, conflict_style, metrics),
            "recommendations": self._generate_recommendations(
                comm_style, conflict_style, metrics
            )
        }
        
        return analysis

    def _identify_patterns(self, responses: List[Dict[str, Any]]) -> List[str]:
        patterns = []
        
        # Analyze emotional expression patterns
        emotional_responses = [
            r for r in responses 
            if r.get("category") == "emotion_expression"
        ]
        if emotional_responses:
            avg_emotional_expression = np.mean([
                r.get("value", 0) for r in emotional_responses
            ])
            if avg_emotional_expression > 0.7:
                patterns.append("High emotional expressiveness")
            elif avg_emotional_expression < 0.3:
                patterns.append("Limited emotional expression")
        
        # Analyze conflict patterns
        conflict_responses = [
            r for r in responses 
            if r.get("category") == "conflict"
        ]
        if conflict_responses:
            avg_conflict_handling = np.mean([
                r.get("value", 0) for r in conflict_responses
            ])
            if avg_conflict_handling > 0.7:
                patterns.append("Effective conflict management")
            elif avg_conflict_handling < 0.3:
                patterns.append("Difficulty with conflict resolution")
        
        return patterns

    def _identify_strengths(
        self,
        comm_style: CommunicationStyle,
        conflict_style: ConflictStyle,
        metrics: CommunicationMetrics
    ) -> List[str]:
        strengths = []
        
        # Communication style strengths
        if comm_style == CommunicationStyle.ASSERTIVE:
            strengths.extend([
                "Clear and direct communication",
                "Balanced self-expression",
                "Respect for others' boundaries"
            ])
        elif comm_style == CommunicationStyle.PASSIVE:
            strengths.extend([
                "Good listening skills",
                "Non-threatening presence",
                "Patience in communication"
            ])
        
        # Conflict style strengths
        if conflict_style == ConflictStyle.COLLABORATIVE:
            strengths.extend([
                "Strong problem-solving skills",
                "Win-win approach to conflicts",
                "Creative solution finding"
            ])
        elif conflict_style == ConflictStyle.COMPROMISING:
            strengths.extend([
                "Flexibility in negotiations",
                "Balanced approach to conflicts",
                "Practical problem-solving"
            ])
        
        # Metrics-based strengths
        if metrics.emotional_awareness_score > 0.7:
            strengths.append("High emotional intelligence")
        if metrics.listening_score > 0.7:
            strengths.append("Excellent listening skills")
        if metrics.nonverbal_congruence_score > 0.7:
            strengths.append("Strong nonverbal communication")
        
        return strengths

    def _identify_challenges(
        self,
        comm_style: CommunicationStyle,
        conflict_style: ConflictStyle,
        metrics: CommunicationMetrics
    ) -> List[str]:
        challenges = []
        
        # Communication style challenges
        if comm_style == CommunicationStyle.PASSIVE:
            challenges.extend([
                "Difficulty expressing needs",
                "May avoid important discussions",
                "Risk of unexpressed resentment"
            ])
        elif comm_style == CommunicationStyle.AGGRESSIVE:
            challenges.extend([
                "May intimidate others",
                "Risk of damaging relationships",
                "Difficulty with emotional regulation"
            ])
        
        # Conflict style challenges
        if conflict_style == ConflictStyle.AVOIDING:
            challenges.extend([
                "Unresolved issues may accumulate",
                "Missing opportunities for growth",
                "May lead to relationship stagnation"
            ])
        elif conflict_style == ConflictStyle.COMPETING:
            challenges.extend([
                "May create win-lose situations",
                "Risk of relationship strain",
                "Potential for escalating conflicts"
            ])
        
        # Metrics-based challenges
        if metrics.clarity_score < 0.3:
            challenges.append("Need to improve message clarity")
        if metrics.emotional_awareness_score < 0.3:
            challenges.append("Limited emotional awareness")
        if metrics.nonverbal_congruence_score < 0.3:
            challenges.append("Inconsistent nonverbal communication")
        
        return challenges

    def _generate_recommendations(
        self,
        comm_style: CommunicationStyle,
        conflict_style: ConflictStyle,
        metrics: CommunicationMetrics
    ) -> List[str]:
        recommendations = []
        
        # Style-based recommendations
        if comm_style == CommunicationStyle.PASSIVE:
            recommendations.extend([
                "Practice expressing needs and opinions",
                "Start with small, low-stakes conversations",
                "Use 'I' statements to express feelings"
            ])
        elif comm_style == CommunicationStyle.AGGRESSIVE:
            recommendations.extend([
                "Practice active listening",
                "Count to ten before responding when angry",
                "Focus on understanding others' perspectives"
            ])
        
        # Conflict style recommendations
        if conflict_style == ConflictStyle.AVOIDING:
            recommendations.extend([
                "Address issues when they're small",
                "Set aside regular time for discussions",
                "Start with expressing appreciation"
            ])
        elif conflict_style == ConflictStyle.COMPETING:
            recommendations.extend([
                "Practice finding win-win solutions",
                "Focus on long-term relationship health",
                "Consider others' needs and feelings"
            ])
        
        # Metrics-based recommendations
        if metrics.clarity_score < 0.5:
            recommendations.append(
                "Practice organizing thoughts before speaking"
            )
        if metrics.emotional_awareness_score < 0.5:
            recommendations.append(
                "Keep an emotion journal to increase awareness"
            )
        if metrics.listening_score < 0.5:
            recommendations.append(
                "Practice reflective listening techniques"
            )
        
        return recommendations