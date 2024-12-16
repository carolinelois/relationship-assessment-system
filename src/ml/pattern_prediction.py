from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger

class PatternPredictor:
    def __init__(self):
        self.relationship_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.outcome_predictor = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.feature_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_importance = {}
        self.prediction_confidence = {}

    async def train_models(
        self,
        training_data: List[Dict[str, Any]],
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        try:
            # Prepare data
            X, y_relationship, y_outcome = self._prepare_training_data(
                training_data
            )
            
            # Split data
            X_train, X_val, y_rel_train, y_rel_val, y_out_train, y_out_val = (
                train_test_split(
                    X, y_relationship, y_outcome,
                    test_size=validation_split,
                    random_state=42
                )
            )
            
            # Scale features
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_val_scaled = self.feature_scaler.transform(X_val)
            
            # Train relationship classifier
            self.relationship_classifier.fit(X_train_scaled, y_rel_train)
            rel_accuracy = accuracy_score(
                y_rel_val,
                self.relationship_classifier.predict(X_val_scaled)
            )
            
            # Train outcome predictor
            self.outcome_predictor.fit(X_train_scaled, y_out_train)
            outcome_mse = mean_squared_error(
                y_out_val,
                self.outcome_predictor.predict(X_val_scaled)
            )
            
            # Calculate feature importance
            self._calculate_feature_importance()
            
            self.is_trained = True
            
            return {
                "status": "success",
                "metrics": {
                    "relationship_accuracy": rel_accuracy,
                    "outcome_mse": outcome_mse
                },
                "feature_importance": self.feature_importance
            }
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def predict_patterns(
        self, assessment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not self.is_trained:
            return {
                "status": "error",
                "error": "Models not trained"
            }
        
        try:
            # Extract features
            features = self._extract_prediction_features(assessment_data)
            
            # Scale features
            features_scaled = self.feature_scaler.transform([features])
            
            # Make predictions
            relationship_pred = self.relationship_classifier.predict(
                features_scaled
            )[0]
            relationship_prob = self.relationship_classifier.predict_proba(
                features_scaled
            )[0]
            
            outcome_pred = self.outcome_predictor.predict(features_scaled)[0]
            
            # Calculate prediction confidence
            confidence = self._calculate_prediction_confidence(
                relationship_prob, features
            )
            
            # Generate insights
            insights = self._generate_prediction_insights(
                relationship_pred,
                outcome_pred,
                features,
                confidence
            )
            
            return {
                "status": "success",
                "predictions": {
                    "relationship_type": relationship_pred,
                    "outcome_score": float(outcome_pred),
                    "confidence": confidence
                },
                "insights": insights,
                "risk_factors": self._identify_risk_factors(
                    features, relationship_pred, outcome_pred
                ),
                "recommendations": self._generate_recommendations(
                    relationship_pred, outcome_pred, features
                )
            }
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def save_models(self, path: str):
        try:
            model_data = {
                "relationship_classifier": self.relationship_classifier,
                "outcome_predictor": self.outcome_predictor,
                "feature_scaler": self.feature_scaler,
                "label_encoder": self.label_encoder,
                "feature_importance": self.feature_importance,
                "metadata": {
                    "saved_at": datetime.utcnow().isoformat(),
                    "version": "1.0.0"
                }
            }
            joblib.dump(model_data, path)
            logger.info(f"Models saved to {path}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise

    def load_models(self, path: str):
        try:
            model_data = joblib.load(path)
            self.relationship_classifier = model_data["relationship_classifier"]
            self.outcome_predictor = model_data["outcome_predictor"]
            self.feature_scaler = model_data["feature_scaler"]
            self.label_encoder = model_data["label_encoder"]
            self.feature_importance = model_data["feature_importance"]
            self.is_trained = True
            logger.info(f"Models loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def _prepare_training_data(
        self, training_data: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Extract features and labels
        features = []
        relationship_labels = []
        outcome_scores = []
        
        for data in training_data:
            # Extract features
            features.append(self._extract_training_features(data))
            
            # Extract labels
            relationship_labels.append(data.get("relationship_type"))
            outcome_scores.append(data.get("outcome_score", 0))
        
        # Convert to numpy arrays
        X = np.array(features)
        y_relationship = self.label_encoder.fit_transform(relationship_labels)
        y_outcome = np.array(outcome_scores)
        
        return X, y_relationship, y_outcome

    def _extract_training_features(
        self, data: Dict[str, Any]
    ) -> List[float]:
        features = []
        
        # Extract attachment features
        attachment = data.get("attachment", {})
        features.extend([
            attachment.get("anxiety_score", 0),
            attachment.get("avoidance_score", 0),
            attachment.get("security_score", 0)
        ])
        
        # Extract communication features
        communication = data.get("communication", {})
        features.extend([
            communication.get("clarity_score", 0),
            communication.get("emotional_awareness_score", 0),
            communication.get("conflict_resolution_score", 0)
        ])
        
        # Extract relationship dynamic features
        dynamics = data.get("dynamics", {})
        features.extend([
            dynamics.get("trust_score", 0),
            dynamics.get("intimacy_score", 0),
            dynamics.get("commitment_score", 0)
        ])
        
        return features

    def _extract_prediction_features(
        self, data: Dict[str, Any]
    ) -> List[float]:
        return self._extract_training_features(data)

    def _calculate_feature_importance(self):
        # Get feature importance from both models
        rel_importance = self.relationship_classifier.feature_importances_
        out_importance = self.outcome_predictor.feature_importances_
        
        # Combine importance scores
        feature_names = [
            "anxiety_score", "avoidance_score", "security_score",
            "clarity_score", "emotional_awareness_score", "conflict_resolution_score",
            "trust_score", "intimacy_score", "commitment_score"
        ]
        
        self.feature_importance = {
            name: {
                "relationship_importance": float(rel_imp),
                "outcome_importance": float(out_imp),
                "overall_importance": float((rel_imp + out_imp) / 2)
            }
            for name, rel_imp, out_imp in zip(
                feature_names, rel_importance, out_importance
            )
        }

    def _calculate_prediction_confidence(
        self,
        relationship_prob: np.ndarray,
        features: List[float]
    ) -> float:
        # Calculate confidence based on:
        # 1. Probability of predicted class
        class_confidence = float(max(relationship_prob))
        
        # 2. Feature reliability
        feature_reliability = self._calculate_feature_reliability(features)
        
        # 3. Model confidence based on feature importance
        model_confidence = self._calculate_model_confidence(features)
        
        # Combine confidence scores
        overall_confidence = (
            class_confidence * 0.4 +
            feature_reliability * 0.3 +
            model_confidence * 0.3
        )
        
        return overall_confidence

    def _calculate_feature_reliability(
        self, features: List[float]
    ) -> float:
        # Check if features are within expected ranges
        in_range_count = sum(
            1 for f in features if 0 <= f <= 1
        )
        
        # Calculate reliability score
        reliability = in_range_count / len(features)
        
        return reliability

    def _calculate_model_confidence(
        self, features: List[float]
    ) -> float:
        # Calculate confidence based on feature importance
        weighted_importance = sum(
            f * self.feature_importance[name]["overall_importance"]
            for f, name in zip(features, self.feature_importance.keys())
        )
        
        return min(max(weighted_importance, 0), 1)

    def _generate_prediction_insights(
        self,
        relationship_pred: str,
        outcome_pred: float,
        features: List[float],
        confidence: float
    ) -> Dict[str, Any]:
        insights = {
            "key_factors": self._identify_key_factors(features),
            "pattern_analysis": self._analyze_prediction_patterns(
                relationship_pred, outcome_pred
            ),
            "confidence_analysis": {
                "overall_confidence": confidence,
                "confidence_factors": self._analyze_confidence_factors(
                    features, confidence
                )
            }
        }
        
        if confidence < 0.6:
            insights["confidence_analysis"]["low_confidence_reasons"] = (
                self._analyze_low_confidence(features, confidence)
            )
        
        return insights

    def _identify_key_factors(
        self, features: List[float]
    ) -> List[Dict[str, Any]]:
        # Get feature names and their importance
        feature_names = list(self.feature_importance.keys())
        
        # Identify significant factors
        significant_factors = []
        for name, value in zip(feature_names, features):
            importance = self.feature_importance[name]["overall_importance"]
            if importance * value > 0.3:  # Threshold for significance
                significant_factors.append({
                    "factor": name,
                    "value": float(value),
                    "importance": float(importance),
                    "impact": "high" if value > 0.7 else "moderate"
                })
        
        return sorted(
            significant_factors,
            key=lambda x: x["importance"] * x["value"],
            reverse=True
        )

    def _analyze_prediction_patterns(
        self,
        relationship_pred: str,
        outcome_pred: float
    ) -> Dict[str, Any]:
        return {
            "relationship_type": {
                "prediction": relationship_pred,
                "characteristics": self._get_relationship_characteristics(
                    relationship_pred
                )
            },
            "outcome_analysis": {
                "score": float(outcome_pred),
                "interpretation": self._interpret_outcome_score(outcome_pred),
                "factors": self._identify_outcome_factors(outcome_pred)
            }
        }

    def _get_relationship_characteristics(
        self, relationship_type: str
    ) -> List[str]:
        # Define characteristics for each relationship type
        characteristics = {
            "secure": [
                "Strong emotional connection",
                "Effective communication",
                "Healthy boundaries"
            ],
            "anxious": [
                "Strong desire for closeness",
                "Fear of abandonment",
                "Emotional dependency"
            ],
            "avoidant": [
                "Values independence",
                "Difficulty with emotional intimacy",
                "Maintains emotional distance"
            ]
        }
        
        return characteristics.get(relationship_type.lower(), [])

    def _interpret_outcome_score(self, score: float) -> str:
        if score >= 0.8:
            return "Very positive outlook"
        elif score >= 0.6:
            return "Positive outlook with areas for growth"
        elif score >= 0.4:
            return "Moderate outlook with significant growth opportunities"
        else:
            return "Challenging outlook requiring focused intervention"

    def _identify_outcome_factors(
        self, score: float
    ) -> List[Dict[str, Any]]:
        factors = []
        
        if score >= 0.8:
            factors.extend([
                {
                    "factor": "Strong foundation",
                    "impact": "positive",
                    "significance": "high"
                },
                {
                    "factor": "Effective communication",
                    "impact": "positive",
                    "significance": "high"
                }
            ])
        elif score <= 0.4:
            factors.extend([
                {
                    "factor": "Communication challenges",
                    "impact": "negative",
                    "significance": "high"
                },
                {
                    "factor": "Trust issues",
                    "impact": "negative",
                    "significance": "high"
                }
            ])
        
        return factors

    def _analyze_confidence_factors(
        self,
        features: List[float],
        confidence: float
    ) -> Dict[str, Any]:
        return {
            "data_quality": self._assess_data_quality(features),
            "model_confidence": self._assess_model_confidence(features),
            "prediction_stability": self._assess_prediction_stability(
                features, confidence
            )
        }

    def _assess_data_quality(
        self, features: List[float]
    ) -> Dict[str, Any]:
        # Check feature completeness
        completeness = sum(1 for f in features if f != 0) / len(features)
        
        # Check feature validity
        validity = sum(1 for f in features if 0 <= f <= 1) / len(features)
        
        return {
            "completeness": float(completeness),
            "validity": float(validity),
            "quality_score": float((completeness + validity) / 2)
        }

    def _assess_model_confidence(
        self, features: List[float]
    ) -> Dict[str, Any]:
        # Calculate confidence based on feature importance
        feature_confidence = {}
        for name, value in zip(self.feature_importance.keys(), features):
            importance = self.feature_importance[name]["overall_importance"]
            feature_confidence[name] = float(importance * value)
        
        return {
            "feature_confidence": feature_confidence,
            "overall_model_confidence": float(
                sum(feature_confidence.values()) / len(feature_confidence)
            )
        }

    def _assess_prediction_stability(
        self,
        features: List[float],
        confidence: float
    ) -> Dict[str, Any]:
        # Analyze prediction stability
        stability_factors = []
        
        # Check feature stability
        for name, value in zip(self.feature_importance.keys(), features):
            importance = self.feature_importance[name]["overall_importance"]
            if abs(value - 0.5) < 0.1 and importance > 0.3:
                stability_factors.append({
                    "factor": name,
                    "issue": "borderline_value",
                    "impact": "moderate"
                })
        
        return {
            "stability_score": float(confidence),
            "stability_factors": stability_factors,
            "is_stable": len(stability_factors) == 0
        }

    def _analyze_low_confidence(
        self,
        features: List[float],
        confidence: float
    ) -> List[Dict[str, Any]]:
        reasons = []
        
        # Check data quality
        data_quality = self._assess_data_quality(features)
        if data_quality["quality_score"] < 0.7:
            reasons.append({
                "reason": "low_data_quality",
                "details": "Incomplete or invalid feature data",
                "score": float(data_quality["quality_score"])
            })
        
        # Check model confidence
        model_conf = self._assess_model_confidence(features)
        if model_conf["overall_model_confidence"] < 0.7:
            reasons.append({
                "reason": "low_model_confidence",
                "details": "Model uncertainty in prediction",
                "score": float(model_conf["overall_model_confidence"])
            })
        
        # Check prediction stability
        stability = self._assess_prediction_stability(features, confidence)
        if not stability["is_stable"]:
            reasons.append({
                "reason": "low_stability",
                "details": "Borderline feature values affecting prediction",
                "factors": stability["stability_factors"]
            })
        
        return reasons

    def _identify_risk_factors(
        self,
        features: List[float],
        relationship_pred: str,
        outcome_pred: float
    ) -> List[Dict[str, Any]]:
        risk_factors = []
        
        # Check for concerning feature values
        feature_names = list(self.feature_importance.keys())
        for name, value in zip(feature_names, features):
            if value < 0.3 and self.feature_importance[name]["overall_importance"] > 0.3:
                risk_factors.append({
                    "factor": name,
                    "severity": "high",
                    "value": float(value),
                    "impact": "significant"
                })
        
        # Check relationship type risks
        if relationship_pred.lower() in ["anxious", "avoidant"]:
            risk_factors.append({
                "factor": f"{relationship_pred.lower()}_attachment",
                "severity": "moderate",
                "impact": "moderate"
            })
        
        # Check outcome risks
        if outcome_pred < 0.4:
            risk_factors.append({
                "factor": "low_outcome_prediction",
                "severity": "high",
                "value": float(outcome_pred),
                "impact": "significant"
            })
        
        return risk_factors

    def _generate_recommendations(
        self,
        relationship_pred: str,
        outcome_pred: float,
        features: List[float]
    ) -> List[Dict[str, Any]]:
        recommendations = []
        
        # Generate type-specific recommendations
        type_recommendations = self._get_type_recommendations(relationship_pred)
        recommendations.extend(type_recommendations)
        
        # Generate outcome-based recommendations
        outcome_recommendations = self._get_outcome_recommendations(outcome_pred)
        recommendations.extend(outcome_recommendations)
        
        # Generate feature-based recommendations
        feature_recommendations = self._get_feature_recommendations(features)
        recommendations.extend(feature_recommendations)
        
        # Prioritize recommendations
        return self._prioritize_recommendations(recommendations)

    def _get_type_recommendations(
        self, relationship_type: str
    ) -> List[Dict[str, Any]]:
        # Define recommendations for each type
        type_recommendations = {
            "secure": [
                {
                    "area": "maintenance",
                    "recommendation": "Continue building emotional connection",
                    "priority": "medium"
                }
            ],
            "anxious": [
                {
                    "area": "self_development",
                    "recommendation": "Work on self-soothing techniques",
                    "priority": "high"
                },
                {
                    "area": "independence",
                    "recommendation": "Develop personal interests and activities",
                    "priority": "high"
                }
            ],
            "avoidant": [
                {
                    "area": "emotional_intimacy",
                    "recommendation": "Practice emotional vulnerability",
                    "priority": "high"
                },
                {
                    "area": "communication",
                    "recommendation": "Share feelings more frequently",
                    "priority": "high"
                }
            ]
        }
        
        return type_recommendations.get(relationship_type.lower(), [])

    def _get_outcome_recommendations(
        self, outcome_score: float
    ) -> List[Dict[str, Any]]:
        recommendations = []
        
        if outcome_score < 0.4:
            recommendations.extend([
                {
                    "area": "intervention",
                    "recommendation": "Seek professional counseling",
                    "priority": "critical"
                },
                {
                    "area": "communication",
                    "recommendation": "Establish regular check-ins",
                    "priority": "high"
                }
            ])
        elif outcome_score < 0.6:
            recommendations.extend([
                {
                    "area": "skill_building",
                    "recommendation": "Learn conflict resolution techniques",
                    "priority": "high"
                },
                {
                    "area": "connection",
                    "recommendation": "Increase quality time together",
                    "priority": "medium"
                }
            ])
        
        return recommendations

    def _get_feature_recommendations(
        self, features: List[float]
    ) -> List[Dict[str, Any]]:
        recommendations = []
        feature_names = list(self.feature_importance.keys())
        
        for name, value in zip(feature_names, features):
            if value < 0.5:
                recommendations.append({
                    "area": name,
                    "recommendation": f"Work on improving {name.replace('_', ' ')}",
                    "priority": "high" if value < 0.3 else "medium"
                })
        
        return recommendations

    def _prioritize_recommendations(
        self, recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        # Define priority order
        priority_order = {
            "critical": 0,
            "high": 1,
            "medium": 2,
            "low": 3
        }
        
        # Sort recommendations by priority
        sorted_recommendations = sorted(
            recommendations,
            key=lambda x: priority_order.get(x["priority"], 999)
        )
        
        # Limit to top 5 recommendations
        return sorted_recommendations[:5]