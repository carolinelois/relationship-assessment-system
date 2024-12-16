from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from io import BytesIO
import base64
from datetime import datetime, timedelta

class VisualizationGenerator:
    def __init__(self):
        self.plt_style = 'seaborn'
        plt.style.use(self.plt_style)
        self.color_palette = sns.color_palette("husl", 8)

    def generate_visualizations(
        self,
        assessment_data: Dict[str, Any],
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, str]:
        visualizations = {}
        
        # Generate core visualizations
        visualizations["metrics_radar"] = self.create_metrics_radar(
            assessment_data.get("metrics", {})
        )
        
        visualizations["pattern_comparison"] = self.create_pattern_comparison(
            assessment_data.get("patterns", {})
        )
        
        visualizations["strength_weakness"] = self.create_strength_weakness_chart(
            assessment_data
        )
        
        # Generate historical visualizations if data available
        if historical_data:
            visualizations["progress_timeline"] = self.create_progress_timeline(
                assessment_data, historical_data
            )
            
            visualizations["trend_analysis"] = self.create_trend_analysis(
                assessment_data, historical_data
            )
        
        return visualizations

    def create_metrics_radar(self, metrics: Dict[str, float]) -> str:
        # Create figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        
        # Prepare data
        categories = list(metrics.keys())
        values = list(metrics.values())
        num_vars = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]
        
        # Plot data
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, color=self.color_palette[0])
        ax.fill(angles, values, alpha=0.25, color=self.color_palette[0])
        
        # Set the labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        # Add title
        plt.title("Relationship Metrics Analysis", size=20, y=1.05)
        
        # Convert to base64
        img_data = self._fig_to_base64(fig)
        plt.close(fig)
        
        return img_data

    def create_pattern_comparison(self, patterns: Dict[str, Any]) -> str:
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data
        pattern_names = list(patterns.keys())
        pattern_values = [
            p["strength"] if isinstance(p, dict) else p 
            for p in patterns.values()
        ]
        
        # Create bar chart
        bars = ax.bar(
            pattern_names,
            pattern_values,
            color=self.color_palette
        )
        
        # Customize chart
        ax.set_title("Pattern Comparison Analysis", size=16)
        ax.set_xlabel("Pattern Type", size=12)
        ax.set_ylabel("Strength", size=12)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.2f}',
                ha='center',
                va='bottom'
            )
        
        # Convert to base64
        img_data = self._fig_to_base64(fig)
        plt.close(fig)
        
        return img_data

    def create_strength_weakness_chart(
        self, assessment_data: Dict[str, Any]
    ) -> str:
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Prepare data
        strengths = assessment_data.get("strengths", [])
        challenges = assessment_data.get("challenges", [])
        
        # Create strength chart
        if strengths:
            strength_names = [s["area"] for s in strengths]
            strength_values = [s["score"] for s in strengths]
            ax1.barh(
                strength_names,
                strength_values,
                color=self.color_palette[0]
            )
            ax1.set_title("Strengths", size=14)
        
        # Create challenge chart
        if challenges:
            challenge_names = [c["area"] for c in challenges]
            challenge_values = [c["score"] for c in challenges]
            ax2.barh(
                challenge_names,
                challenge_values,
                color=self.color_palette[1]
            )
            ax2.set_title("Areas for Growth", size=14)
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert to base64
        img_data = self._fig_to_base64(fig)
        plt.close(fig)
        
        return img_data

    def create_progress_timeline(
        self,
        current_assessment: Dict[str, Any],
        historical_data: List[Dict[str, Any]]
    ) -> str:
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data
        all_data = historical_data + [current_assessment]
        dates = [
            datetime.fromisoformat(d["timestamp"]) 
            for d in all_data
        ]
        health_scores = [
            d.get("metrics", {}).get("relationship_health", 0) 
            for d in all_data
        ]
        
        # Create line plot
        ax.plot(
            dates,
            health_scores,
            marker='o',
            linestyle='-',
            color=self.color_palette[0]
        )
        
        # Customize chart
        ax.set_title("Relationship Health Progress", size=16)
        ax.set_xlabel("Date", size=12)
        ax.set_ylabel("Health Score", size=12)
        
        # Format dates
        fig.autofmt_xdate()
        
        # Add trend line
        z = np.polyfit(range(len(dates)), health_scores, 1)
        p = np.poly1d(z)
        ax.plot(
            dates,
            p(range(len(dates))),
            "r--",
            alpha=0.8,
            label="Trend"
        )
        
        ax.legend()
        
        # Convert to base64
        img_data = self._fig_to_base64(fig)
        plt.close(fig)
        
        return img_data

    def create_trend_analysis(
        self,
        current_assessment: Dict[str, Any],
        historical_data: List[Dict[str, Any]]
    ) -> str:
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Prepare data
        all_data = historical_data + [current_assessment]
        metrics = [
            "relationship_health",
            "pattern_consistency",
            "growth_potential",
            "stability_score"
        ]
        
        # Create subplots for each metric
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            dates = [
                datetime.fromisoformat(d["timestamp"]) 
                for d in all_data
            ]
            values = [
                d.get("metrics", {}).get(metric, 0) 
                for d in all_data
            ]
            
            # Plot metric trend
            ax.plot(
                dates,
                values,
                marker='o',
                linestyle='-',
                color=self.color_palette[idx]
            )
            
            # Add trend line
            z = np.polyfit(range(len(dates)), values, 1)
            p = np.poly1d(z)
            ax.plot(
                dates,
                p(range(len(dates))),
                "r--",
                alpha=0.8
            )
            
            # Customize subplot
            ax.set_title(metric.replace("_", " ").title(), size=12)
            ax.tick_params(axis='x', rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert to base64
        img_data = self._fig_to_base64(fig)
        plt.close(fig)
        
        return img_data

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        buf.close()
        return img_str

    def create_custom_visualization(
        self,
        data: Dict[str, Any],
        viz_type: str,
        **kwargs
    ) -> str:
        """Create custom visualization based on specific requirements."""
        if viz_type == "relationship_network":
            return self._create_relationship_network(data, **kwargs)
        elif viz_type == "pattern_evolution":
            return self._create_pattern_evolution(data, **kwargs)
        elif viz_type == "risk_assessment":
            return self._create_risk_assessment(data, **kwargs)
        else:
            raise ValueError(f"Unsupported visualization type: {viz_type}")

    def _create_relationship_network(
        self, data: Dict[str, Any], **kwargs
    ) -> str:
        # Create network visualization
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Implementation would depend on specific requirements
        # This is a placeholder
        
        # Convert to base64
        img_data = self._fig_to_base64(fig)
        plt.close(fig)
        
        return img_data

    def _create_pattern_evolution(
        self, data: Dict[str, Any], **kwargs
    ) -> str:
        # Create pattern evolution visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Implementation would depend on specific requirements
        # This is a placeholder
        
        # Convert to base64
        img_data = self._fig_to_base64(fig)
        plt.close(fig)
        
        return img_data

    def _create_risk_assessment(
        self, data: Dict[str, Any], **kwargs
    ) -> str:
        # Create risk assessment visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Implementation would depend on specific requirements
        # This is a placeholder
        
        # Convert to base64
        img_data = self._fig_to_base64(fig)
        plt.close(fig)
        
        return img_data