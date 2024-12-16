from typing import Any, Dict, List
from .base_agent import BaseAgent

class RelationshipPsychologistAgent(BaseAgent):
    def __init__(self):
        super().__init__("relationship_psychologist", "expert")
        self.frameworks = ["Attachment Theory", "Gottman Method", "Family Systems"]

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            analysis_type = data.get("analysis_type")
            if analysis_type == "generate_questions":
                return await self.generate_questions(data)
            elif analysis_type == "analyze_responses":
                return await self.analyze_responses(data)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
        except Exception as e:
            return await self.handle_error(e)

    async def generate_questions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        framework = data.get("framework")
        if framework not in self.frameworks:
            raise ValueError(f"Unsupported framework: {framework}")

        questions = await self._get_framework_questions(framework)
        return self.create_response({
            "framework": framework,
            "questions": questions
        })

    async def analyze_responses(self, data: Dict[str, Any]) -> Dict[str, Any]:
        responses = data.get("responses", [])
        framework = data.get("framework")
        
        analysis = await self._analyze_by_framework(framework, responses)
        return self.create_response({
            "framework": framework,
            "analysis": analysis
        })

    async def _get_framework_questions(self, framework: str) -> List[Dict[str, Any]]:
        questions_map = {
            "Attachment Theory": [
                {"id": "at_1", "question": "How do you typically react when your partner is away?", "category": "separation_anxiety"},
                {"id": "at_2", "question": "How comfortable are you depending on your partner?", "category": "dependency_comfort"},
                {"id": "at_3", "question": "How do you handle emotional intimacy?", "category": "emotional_intimacy"}
            ],
            "Gottman Method": [
                {"id": "gm_1", "question": "How do you handle conflicts with your partner?", "category": "conflict_resolution"},
                {"id": "gm_2", "question": "How do you show appreciation to your partner?", "category": "positive_sentiment"},
                {"id": "gm_3", "question": "How do you respond to your partner's emotional needs?", "category": "emotional_bidding"}
            ],
            "Family Systems": [
                {"id": "fs_1", "question": "How are boundaries maintained in your relationship?", "category": "boundaries"},
                {"id": "fs_2", "question": "How are decisions made in your relationship?", "category": "hierarchy"},
                {"id": "fs_3", "question": "How does your family of origin influence your current relationship?", "category": "generational_patterns"}
            ]
        }
        return questions_map.get(framework, [])

    async def _analyze_by_framework(self, framework: str, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        analysis_templates = {
            "Attachment Theory": {
                "attachment_style": self._determine_attachment_style(responses),
                "relationship_patterns": self._analyze_relationship_patterns(responses),
                "recommendations": self._generate_attachment_recommendations(responses)
            },
            "Gottman Method": {
                "four_horsemen_presence": self._analyze_four_horsemen(responses),
                "positive_negative_ratio": self._calculate_sentiment_ratio(responses),
                "recommendations": self._generate_gottman_recommendations(responses)
            },
            "Family Systems": {
                "boundary_analysis": self._analyze_boundaries(responses),
                "family_patterns": self._analyze_family_patterns(responses),
                "recommendations": self._generate_family_systems_recommendations(responses)
            }
        }
        return analysis_templates.get(framework, {})

    def _determine_attachment_style(self, responses: List[Dict[str, Any]]) -> str:
        # Implement attachment style determination logic
        return "Secure"  # Placeholder

    def _analyze_relationship_patterns(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"primary_pattern": "Collaborative", "secondary_pattern": "Supportive"}

    def _generate_attachment_recommendations(self, responses: List[Dict[str, Any]]) -> List[str]:
        return ["Practice open communication", "Build trust through consistency"]

    def _analyze_four_horsemen(self, responses: List[Dict[str, Any]]) -> Dict[str, bool]:
        return {
            "criticism": False,
            "contempt": False,
            "defensiveness": False,
            "stonewalling": False
        }

    def _calculate_sentiment_ratio(self, responses: List[Dict[str, Any]]) -> float:
        return 5.0  # Placeholder

    def _generate_gottman_recommendations(self, responses: List[Dict[str, Any]]) -> List[str]:
        return ["Build love maps", "Turn towards instead of away"]

    def _analyze_boundaries(self, responses: List[Dict[str, Any]]) -> Dict[str, str]:
        return {"personal": "healthy", "family": "needs_attention"}

    def _analyze_family_patterns(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"inherited_patterns": [], "current_patterns": []}

    def _generate_family_systems_recommendations(self, responses: List[Dict[str, Any]]) -> List[str]:
        return ["Establish clear boundaries", "Address generational patterns"]


class BehavioralPsychologistAgent(BaseAgent):
    def __init__(self):
        super().__init__("behavioral_psychologist", "expert")
        self.domains = ["verbal", "nonverbal", "emotional", "conflict", "intimacy", "support"]

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            domain = data.get("domain")
            if domain not in self.domains:
                raise ValueError(f"Unsupported domain: {domain}")

            action = data.get("action")
            if action == "analyze_behavior":
                return await self.analyze_behavior(data)
            elif action == "generate_recommendations":
                return await self.generate_recommendations(data)
            else:
                raise ValueError(f"Unknown action: {action}")
        except Exception as e:
            return await self.handle_error(e)

    async def analyze_behavior(self, data: Dict[str, Any]) -> Dict[str, Any]:
        domain = data.get("domain")
        behaviors = data.get("behaviors", [])
        
        analysis = await self._analyze_domain_behaviors(domain, behaviors)
        return self.create_response({
            "domain": domain,
            "analysis": analysis
        })

    async def generate_recommendations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        domain = data.get("domain")
        analysis = data.get("analysis", {})
        
        recommendations = await self._generate_domain_recommendations(domain, analysis)
        return self.create_response({
            "domain": domain,
            "recommendations": recommendations
        })

    async def _analyze_domain_behaviors(self, domain: str, behaviors: List[Dict[str, Any]]) -> Dict[str, Any]:
        analysis_map = {
            "verbal": self._analyze_verbal_communication,
            "nonverbal": self._analyze_nonverbal_communication,
            "emotional": self._analyze_emotional_patterns,
            "conflict": self._analyze_conflict_patterns,
            "intimacy": self._analyze_intimacy_patterns,
            "support": self._analyze_support_patterns
        }
        
        analyzer = analysis_map.get(domain)
        if not analyzer:
            raise ValueError(f"No analyzer found for domain: {domain}")
            
        return analyzer(behaviors)

    async def _generate_domain_recommendations(self, domain: str, analysis: Dict[str, Any]) -> List[str]:
        recommendation_map = {
            "verbal": self._generate_verbal_recommendations,
            "nonverbal": self._generate_nonverbal_recommendations,
            "emotional": self._generate_emotional_recommendations,
            "conflict": self._generate_conflict_recommendations,
            "intimacy": self._generate_intimacy_recommendations,
            "support": self._generate_support_recommendations
        }
        
        recommender = recommendation_map.get(domain)
        if not recommender:
            raise ValueError(f"No recommender found for domain: {domain}")
            
        return recommender(analysis)

    def _analyze_verbal_communication(self, behaviors: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "communication_style": "assertive",
            "clarity": "high",
            "effectiveness": "moderate"
        }

    def _analyze_nonverbal_communication(self, behaviors: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "body_language": "open",
            "facial_expressions": "positive",
            "physical_proximity": "comfortable"
        }

    def _analyze_emotional_patterns(self, behaviors: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "emotional_awareness": "high",
            "regulation": "moderate",
            "expression": "healthy"
        }

    def _analyze_conflict_patterns(self, behaviors: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "conflict_style": "collaborative",
            "resolution_effectiveness": "high",
            "pattern_health": "positive"
        }

    def _analyze_intimacy_patterns(self, behaviors: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "emotional_intimacy": "strong",
            "physical_intimacy": "healthy",
            "intellectual_intimacy": "developing"
        }

    def _analyze_support_patterns(self, behaviors: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "emotional_support": "consistent",
            "practical_support": "reliable",
            "reciprocity": "balanced"
        }

    def _generate_verbal_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        return [
            "Practice active listening",
            "Use 'I' statements",
            "Verify understanding"
        ]

    def _generate_nonverbal_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        return [
            "Maintain appropriate eye contact",
            "Mirror partner's body language",
            "Be mindful of personal space"
        ]

    def _generate_emotional_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        return [
            "Practice emotional awareness",
            "Develop healthy coping mechanisms",
            "Express emotions constructively"
        ]

    def _generate_conflict_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        return [
            "Address issues promptly",
            "Focus on solutions",
            "Take cooling-off periods when needed"
        ]

    def _generate_intimacy_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        return [
            "Schedule quality time together",
            "Share thoughts and feelings regularly",
            "Maintain physical affection"
        ]

    def _generate_support_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        return [
            "Offer proactive support",
            "Communicate support needs clearly",
            "Balance giving and receiving support"
        ]