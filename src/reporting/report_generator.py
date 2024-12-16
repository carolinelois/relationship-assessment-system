from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from ..analysis.pattern_integration import RelationshipStyle
from ..analysis.attachment_analysis import AttachmentStyle
from ..analysis.communication_analysis import CommunicationStyle, ConflictStyle

class ReportType(str, Enum):
    SUMMARY = "summary"
    DETAILED = "detailed"
    PROFESSIONAL = "professional"
    PROGRESS = "progress"

@dataclass
class ReportConfig:
    report_type: ReportType
    include_visualizations: bool = True
    include_recommendations: bool = True
    include_risk_factors: bool = True
    include_progress_tracking: bool = False
    professional_terminology: bool = False

class ReportGenerator:
    def __init__(self, config: ReportConfig):
        self.config = config
        self.plt_style = 'seaborn'
        plt.style.use(self.plt_style)

    def generate_report(
        self,
        assessment_data: Dict[str, Any],
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        report = {
            "metadata": self._generate_metadata(assessment_data),
            "summary": self._generate_summary(assessment_data),
            "analysis": self._generate_analysis_section(assessment_data)
        }

        if self.config.include_recommendations:
            report["recommendations"] = self._generate_recommendations_section(
                assessment_data
            )

        if self.config.include_risk_factors:
            report["risk_factors"] = self._generate_risk_factors_section(
                assessment_data
            )

        if self.config.include_visualizations:
            report["visualizations"] = self._generate_visualizations(
                assessment_data
            )

        if self.config.include_progress_tracking and historical_data:
            report["progress"] = self._generate_progress_section(
                assessment_data, historical_data
            )

        return report

    def _generate_metadata(self, assessment_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "report_id": f"report_{datetime.utcnow().timestamp()}",
            "generated_at": datetime.utcnow().isoformat(),
            "report_type": self.config.report_type,
            "assessment_id": assessment_data.get("assessment_id"),
            "profile_id": assessment_data.get("profile_id")
        }

    def _generate_summary(self, assessment_data: Dict[str, Any]) -> Dict[str, Any]:
        relationship_style = assessment_data.get("relationship_style", 
            RelationshipStyle.GROWTH_ORIENTED)
        
        metrics = assessment_data.get("metrics", {})
        
        summary = {
            "relationship_style": relationship_style,
            "overall_health": self._calculate_overall_health(metrics),
            "key_strengths": self._identify_key_strengths(assessment_data),
            "primary_challenges": self._identify_primary_challenges(assessment_data),
            "critical_areas": self._identify_critical_areas(assessment_data)
        }

        if self.config.professional_terminology:
            summary["clinical_observations"] = self._generate_clinical_observations(
                assessment_data
            )

        return summary

    def _generate_analysis_section(
        self, assessment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        analysis = {
            "attachment": self._analyze_attachment_patterns(
                assessment_data.get("attachment_data", {})
            ),
            "communication": self._analyze_communication_patterns(
                assessment_data.get("communication_data", {})
            ),
            "relationship_dynamics": self._analyze_relationship_dynamics(
                assessment_data
            )
        }

        if self.config.report_type in [ReportType.DETAILED, ReportType.PROFESSIONAL]:
            analysis["pattern_integration"] = self._analyze_integrated_patterns(
                assessment_data
            )

        return analysis

    def _generate_recommendations_section(
        self, assessment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        recommendations = {
            "priority_actions": self._identify_priority_actions(assessment_data),
            "growth_areas": self._identify_growth_areas(assessment_data),
            "maintenance_strategies": self._identify_maintenance_strategies(
                assessment_data
            )
        }

        if self.config.report_type in [ReportType.DETAILED, ReportType.PROFESSIONAL]:
            recommendations["long_term_goals"] = self._identify_long_term_goals(
                assessment_data
            )
            recommendations["resource_recommendations"] = (
                self._generate_resource_recommendations(assessment_data)
            )

        return recommendations

    def _generate_risk_factors_section(
        self, assessment_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        risk_factors = assessment_data.get("risk_factors", [])
        
        return {
            "critical_risks": self._identify_critical_risks(risk_factors),
            "warning_signs": self._identify_warning_signs(risk_factors),
            "protective_factors": self._identify_protective_factors(assessment_data),
            "risk_mitigation": self._generate_risk_mitigation_strategies(
                risk_factors, assessment_data
            )
        }

    def _generate_visualizations(
        self, assessment_data: Dict[str, Any]
    ) -> Dict[str, str]:
        visualizations = {}

        # Generate metric radar chart
        metrics_chart = self._create_metrics_radar_chart(
            assessment_data.get("metrics", {})
        )
        visualizations["metrics_radar"] = metrics_chart

        # Generate pattern comparison chart
        pattern_chart = self._create_pattern_comparison_chart(assessment_data)
        visualizations["pattern_comparison"] = pattern_chart

        # Generate relationship health timeline
        if self.config.include_progress_tracking:
            health_timeline = self._create_health_timeline(assessment_data)
            visualizations["health_timeline"] = health_timeline

        return visualizations

    def _create_metrics_radar_chart(self, metrics: Dict[str, float]) -> str:
        # Create radar chart
        categories = list(metrics.keys())
        values = list(metrics.values())

        # Number of variables
        num_vars = len(categories)

        # Compute angle for each axis
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]

        # Initialize the spider plot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

        # Plot data
        values += values[:1]
        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.25)

        # Set the labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)

        # Save plot
        plt.close()
        return "radar_chart_data"  # In practice, save to file or convert to base64

    def _create_pattern_comparison_chart(
        self, assessment_data: Dict[str, Any]
    ) -> str:
        # Create comparison chart
        patterns = assessment_data.get("patterns", {})
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        
        # Plot patterns
        x = list(patterns.keys())
        y = list(patterns.values())
        plt.bar(x, y)
        
        # Customize chart
        plt.title("Pattern Comparison")
        plt.xlabel("Pattern Type")
        plt.ylabel("Strength")
        
        # Save plot
        plt.close()
        return "pattern_chart_data"  # In practice, save to file or convert to base64

    def _create_health_timeline(self, assessment_data: Dict[str, Any]) -> str:
        # Create health timeline
        timeline_data = assessment_data.get("health_timeline", [])
        
        if not timeline_data:
            return ""
        
        # Create line plot
        plt.figure(figsize=(12, 6))
        
        # Plot timeline
        dates = [d["date"] for d in timeline_data]
        scores = [d["score"] for d in timeline_data]
        plt.plot(dates, scores)
        
        # Customize chart
        plt.title("Relationship Health Timeline")
        plt.xlabel("Date")
        plt.ylabel("Health Score")
        
        # Save plot
        plt.close()
        return "timeline_chart_data"  # In practice, save to file or convert to base64

    def _generate_progress_section(
        self,
        current_assessment: Dict[str, Any],
        historical_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        return {
            "progress_metrics": self._calculate_progress_metrics(
                current_assessment, historical_data
            ),
            "trend_analysis": self._analyze_trends(current_assessment, historical_data),
            "milestone_tracking": self._track_milestones(
                current_assessment, historical_data
            ),
            "growth_indicators": self._identify_growth_indicators(
                current_assessment, historical_data
            )
        }

    def _calculate_overall_health(self, metrics: Dict[str, float]) -> float:
        if not metrics:
            return 0.0
        
        # Calculate weighted average of metrics
        weights = {
            "relationship_health": 0.3,
            "pattern_consistency": 0.2,
            "growth_potential": 0.2,
            "stability_score": 0.2,
            "risk_level": 0.1
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                total_score += metrics[metric] * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0

    def _identify_key_strengths(
        self, assessment_data: Dict[str, Any]
    ) -> List[str]:
        strengths = []
        
        # Get strengths from different analysis components
        if "attachment_data" in assessment_data:
            strengths.extend(assessment_data["attachment_data"].get("strengths", []))
        
        if "communication_data" in assessment_data:
            strengths.extend(assessment_data["communication_data"].get("strengths", []))
        
        # Filter and prioritize strengths
        return self._prioritize_items(strengths, max_items=5)

    def _identify_primary_challenges(
        self, assessment_data: Dict[str, Any]
    ) -> List[str]:
        challenges = []
        
        # Get challenges from different analysis components
        if "attachment_data" in assessment_data:
            challenges.extend(assessment_data["attachment_data"].get("challenges", []))
        
        if "communication_data" in assessment_data:
            challenges.extend(assessment_data["communication_data"].get("challenges", []))
        
        # Filter and prioritize challenges
        return self._prioritize_items(challenges, max_items=5)

    def _identify_critical_areas(
        self, assessment_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        critical_areas = []
        metrics = assessment_data.get("metrics", {})
        
        # Check for critical scores in metrics
        for metric, score in metrics.items():
            if score < 0.3:  # Critical threshold
                critical_areas.append({
                    "area": metric,
                    "score": score,
                    "severity": "high",
                    "impact": "significant"
                })
        
        return critical_areas

    def _generate_clinical_observations(
        self, assessment_data: Dict[str, Any]
    ) -> List[str]:
        observations = []
        
        # Generate clinical observations based on patterns
        attachment_style = assessment_data.get("attachment_style")
        if attachment_style == AttachmentStyle.DISORGANIZED:
            observations.append(
                "Presents with disorganized attachment patterns indicating potential "
                "complex relational trauma"
            )
        
        communication_style = assessment_data.get("communication_style")
        if communication_style == CommunicationStyle.AGGRESSIVE:
            observations.append(
                "Demonstrates marked difficulty with affect regulation in "
                "interpersonal communications"
            )
        
        return observations

    def _prioritize_items(
        self, items: List[str], max_items: int = 5
    ) -> List[str]:
        # Remove duplicates while preserving order
        unique_items = list(dict.fromkeys(items))
        
        # Sort by importance (in practice, implement importance scoring)
        # For now, just return first max_items
        return unique_items[:max_items]

    def _calculate_progress_metrics(
        self,
        current_assessment: Dict[str, Any],
        historical_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not historical_data:
            return {}
        
        # Get previous assessment
        previous_assessment = historical_data[-1]
        
        # Calculate changes in metrics
        current_metrics = current_assessment.get("metrics", {})
        previous_metrics = previous_assessment.get("metrics", {})
        
        changes = {}
        for metric, current_value in current_metrics.items():
            if metric in previous_metrics:
                changes[metric] = current_value - previous_metrics[metric]
        
        return {
            "metric_changes": changes,
            "overall_progress": sum(changes.values()) / len(changes) if changes else 0
        }

    def _analyze_trends(
        self,
        current_assessment: Dict[str, Any],
        historical_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not historical_data:
            return {}
        
        # Combine current and historical data
        all_data = historical_data + [current_assessment]
        
        # Analyze metric trends
        metric_trends = {}
        for metric in current_assessment.get("metrics", {}):
            values = [
                assessment.get("metrics", {}).get(metric, 0) 
                for assessment in all_data
            ]
            metric_trends[metric] = self._calculate_trend(values)
        
        return {
            "metric_trends": metric_trends,
            "overall_direction": self._determine_overall_trend(metric_trends)
        }

    def _calculate_trend(self, values: List[float]) -> str:
        if len(values) < 2:
            return "stable"
        
        # Calculate simple linear regression
        x = list(range(len(values)))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return "improving"
        elif slope < -0.1:
            return "declining"
        else:
            return "stable"

    def _determine_overall_trend(self, metric_trends: Dict[str, str]) -> str:
        # Count trend directions
        improving = sum(1 for trend in metric_trends.values() if trend == "improving")
        declining = sum(1 for trend in metric_trends.values() if trend == "declining")
        
        if improving > declining:
            return "positive"
        elif declining > improving:
            return "negative"
        else:
            return "stable"

    def _track_milestones(
        self,
        current_assessment: Dict[str, Any],
        historical_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        milestones = []
        
        # Define milestone thresholds
        thresholds = {
            "relationship_health": 0.7,
            "pattern_consistency": 0.7,
            "stability_score": 0.7
        }
        
        # Check for reached milestones
        current_metrics = current_assessment.get("metrics", {})
        for metric, threshold in thresholds.items():
            if metric in current_metrics and current_metrics[metric] >= threshold:
                milestones.append({
                    "type": metric,
                    "threshold": threshold,
                    "achieved_value": current_metrics[metric],
                    "date_achieved": datetime.utcnow().isoformat()
                })
        
        return milestones

    def _identify_growth_indicators(
        self,
        current_assessment: Dict[str, Any],
        historical_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        indicators = []
        
        if not historical_data:
            return indicators
        
        # Compare current assessment with previous
        previous_assessment = historical_data[-1]
        
        # Check for improvements in key areas
        current_metrics = current_assessment.get("metrics", {})
        previous_metrics = previous_assessment.get("metrics", {})
        
        for metric, current_value in current_metrics.items():
            if metric in previous_metrics:
                improvement = current_value - previous_metrics[metric]
                if improvement > 0.1:  # Significant improvement threshold
                    indicators.append({
                        "area": metric,
                        "improvement": improvement,
                        "from_value": previous_metrics[metric],
                        "to_value": current_value
                    })
        
        return indicators