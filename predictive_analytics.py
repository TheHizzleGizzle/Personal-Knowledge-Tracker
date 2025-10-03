"""
Predictive Analytics Engine
==========================

This module implements advanced predictive analytics for the Knowledge Metabolism Tracker,
focusing on predicting idea importance, forecasting knowledge gaps, and identifying
optimal learning paths.

Key Features:
- Idea importance prediction using temporal patterns
- Knowledge gap forecasting based on project requirements
- Learning path optimization
- Confidence scoring for predictions
- Explainable AI reasoning
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
import json
from collections import defaultdict

@dataclass
class PredictionResult:
    """Container for prediction results"""
    target_id: str
    prediction_type: str
    predicted_value: float
    confidence: float
    reasoning: str
    contributing_factors: Dict[str, float]
    prediction_date: datetime
    target_date: datetime

@dataclass
class KnowledgeGapPrediction:
    """Container for knowledge gap predictions"""
    gap_topic: str
    urgency_score: float
    predicted_need_date: datetime
    confidence: float
    context: str
    recommendations: List[str]

class FeatureExtractor:
    """Extract features for machine learning models"""
    
    def __init__(self, temporal_graph):
        self.temporal_graph = temporal_graph
        
    def extract_concept_features(self, concept_id: str) -> Dict[str, float]:
        """Extract features for a specific concept"""
        if concept_id not in self.temporal_graph.graph.nodes:
            return {}
            
        node_data = self.temporal_graph.graph.nodes[concept_id]
        current_time = datetime.utcnow()
        
        # Temporal features
        created_at = node_data['created_at']
        last_accessed = node_data.get('last_accessed', created_at)
        age_days = (current_time - created_at).days
        days_since_access = (current_time - last_accessed).days
        
        # Access pattern features
        access_frequency = node_data.get('access_frequency', 0)
        access_rate = access_frequency / max(1, age_days)  # accesses per day
        
        # Relationship features
        out_degree = self.temporal_graph.graph.out_degree(concept_id)
        in_degree = self.temporal_graph.graph.in_degree(concept_id)
        total_degree = out_degree + in_degree
        
        # Relationship strength features
        relationship_strengths = [
            data.get('strength', 0.5)
            for _, _, data in self.temporal_graph.graph.edges(concept_id, data=True)
        ]
        avg_relationship_strength = np.mean(relationship_strengths) if relationship_strengths else 0.0
        max_relationship_strength = np.max(relationship_strengths) if relationship_strengths else 0.0
        
        # Domain diversity
        domain_tags = node_data.get('domain_tags', [])
        domain_count = len(set(domain_tags))
        
        # Content features
        content_length = len(node_data.get('content_summary', ''))
        
        # Temporal activity features
        recent_activity = self._calculate_recent_activity(concept_id, days=30)
        activity_trend = self._calculate_activity_trend(concept_id, days=90)
        
        return {
            'age_days': age_days,
            'days_since_access': days_since_access,
            'access_frequency': access_frequency,
            'access_rate': access_rate,
            'out_degree': out_degree,
            'in_degree': in_degree,
            'total_degree': total_degree,
            'avg_relationship_strength': avg_relationship_strength,
            'max_relationship_strength': max_relationship_strength,
            'domain_count': domain_count,
            'content_length': content_length,
            'recent_activity': recent_activity,
            'activity_trend': activity_trend,
            'current_importance': node_data.get('importance_score', 0.0)
        }
        
    def extract_user_features(self, user_id: int) -> Dict[str, float]:
        """Extract user-level features"""
        # Get all concepts for user
        user_concepts = [
            concept_id for concept_id in self.temporal_graph.graph.nodes
            if self.temporal_graph.graph.nodes[concept_id].get('user_id') == user_id
        ]
        
        if not user_concepts:
            return {}
            
        # Calculate user-level statistics
        total_concepts = len(user_concepts)
        
        access_frequencies = [
            self.temporal_graph.graph.nodes[concept_id].get('access_frequency', 0)
            for concept_id in user_concepts
        ]
        avg_access_frequency = np.mean(access_frequencies)
        
        importance_scores = [
            self.temporal_graph.graph.nodes[concept_id].get('importance_score', 0.0)
            for concept_id in user_concepts
        ]
        avg_importance = np.mean(importance_scores)
        
        # Domain diversity
        all_domains = []
        for concept_id in user_concepts:
            domains = self.temporal_graph.graph.nodes[concept_id].get('domain_tags', [])
            all_domains.extend(domains)
        unique_domains = len(set(all_domains))
        
        # Learning velocity (concepts created per week)
        current_time = datetime.utcnow()
        recent_concepts = [
            concept_id for concept_id in user_concepts
            if (current_time - self.temporal_graph.graph.nodes[concept_id]['created_at']).days <= 7
        ]
        learning_velocity = len(recent_concepts)
        
        return {
            'total_concepts': total_concepts,
            'avg_access_frequency': avg_access_frequency,
            'avg_importance': avg_importance,
            'unique_domains': unique_domains,
            'learning_velocity': learning_velocity
        }
        
    def _calculate_recent_activity(self, concept_id: str, days: int) -> float:
        """Calculate recent activity level for a concept"""
        events = self.temporal_graph.concept_timeline.get(concept_id, [])
        current_time = datetime.utcnow()
        cutoff_time = current_time - timedelta(days=days)
        
        recent_events = [
            event for event in events
            if event['timestamp'] >= cutoff_time
        ]
        
        return len(recent_events) / max(1, days)  # events per day
        
    def _calculate_activity_trend(self, concept_id: str, days: int) -> float:
        """Calculate activity trend (positive = increasing, negative = decreasing)"""
        events = self.temporal_graph.concept_timeline.get(concept_id, [])
        current_time = datetime.utcnow()
        
        # Split time period in half
        mid_time = current_time - timedelta(days=days//2)
        cutoff_time = current_time - timedelta(days=days)
        
        recent_events = [
            event for event in events
            if mid_time <= event['timestamp'] <= current_time
        ]
        
        older_events = [
            event for event in events
            if cutoff_time <= event['timestamp'] < mid_time
        ]
        
        recent_rate = len(recent_events) / max(1, days//2)
        older_rate = len(older_events) / max(1, days//2)
        
        if older_rate == 0:
            return 1.0 if recent_rate > 0 else 0.0
        
        return (recent_rate - older_rate) / older_rate

class ImportancePredictor:
    """Predict future importance of knowledge concepts"""
    
    def __init__(self, temporal_graph, feature_extractor):
        self.temporal_graph = temporal_graph
        self.feature_extractor = feature_extractor
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train_model(self, training_data: List[Dict]) -> Dict:
        """Train the importance prediction model"""
        if len(training_data) < 10:
            # Generate synthetic training data if insufficient real data
            training_data = self._generate_synthetic_training_data()
            
        # Prepare features and targets
        features = []
        targets = []
        
        for sample in training_data:
            concept_features = self.feature_extractor.extract_concept_features(sample['concept_id'])
            if concept_features:
                feature_vector = list(concept_features.values())
                features.append(feature_vector)
                targets.append(sample['future_importance'])
                
        if len(features) < 5:
            return {'error': 'Insufficient training data'}
            
        X = np.array(features)
        y = np.array(targets)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        self.is_trained = True
        
        return {
            'training_samples': len(training_data),
            'mse': mse,
            'feature_importance': dict(zip(
                self.feature_extractor.extract_concept_features(training_data[0]['concept_id']).keys(),
                self.model.feature_importances_
            ))
        }
        
    def predict_importance(self, concept_id: str, months_ahead: int = 6) -> PredictionResult:
        """Predict future importance of a concept"""
        if not self.is_trained:
            # Train with synthetic data if no model exists
            self.train_model([])
            
        # Extract features
        features = self.feature_extractor.extract_concept_features(concept_id)
        if not features:
            return PredictionResult(
                target_id=concept_id,
                prediction_type='importance',
                predicted_value=0.5,
                confidence=0.1,
                reasoning="Insufficient data for prediction",
                contributing_factors={},
                prediction_date=datetime.utcnow(),
                target_date=datetime.utcnow() + timedelta(days=30*months_ahead)
            )
            
        # Make prediction
        feature_vector = np.array([list(features.values())])
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        predicted_importance = self.model.predict(feature_vector_scaled)[0]
        predicted_importance = max(0.0, min(1.0, predicted_importance))
        
        # Calculate confidence based on feature quality
        confidence = self._calculate_prediction_confidence(features)
        
        # Generate reasoning
        reasoning = self._generate_importance_reasoning(features, predicted_importance)
        
        # Get contributing factors
        contributing_factors = self._get_contributing_factors(features)
        
        return PredictionResult(
            target_id=concept_id,
            prediction_type='importance',
            predicted_value=predicted_importance,
            confidence=confidence,
            reasoning=reasoning,
            contributing_factors=contributing_factors,
            prediction_date=datetime.utcnow(),
            target_date=datetime.utcnow() + timedelta(days=30*months_ahead)
        )
        
    def _generate_synthetic_training_data(self) -> List[Dict]:
        """Generate synthetic training data for model training"""
        synthetic_data = []
        
        # Get all concepts from the graph
        concepts = list(self.temporal_graph.graph.nodes.keys())
        
        for concept_id in concepts[:50]:  # Use up to 50 concepts
            # Simulate future importance based on current patterns
            current_features = self.feature_extractor.extract_concept_features(concept_id)
            if not current_features:
                continue
                
            # Simple heuristic for future importance
            current_importance = current_features.get('current_importance', 0.5)
            access_rate = current_features.get('access_rate', 0.0)
            relationship_strength = current_features.get('avg_relationship_strength', 0.5)
            activity_trend = current_features.get('activity_trend', 0.0)
            
            # Simulate future importance
            future_importance = current_importance + (
                0.3 * access_rate +
                0.2 * relationship_strength +
                0.2 * activity_trend +
                np.random.normal(0, 0.1)  # Add noise
            )
            
            future_importance = max(0.0, min(1.0, future_importance))
            
            synthetic_data.append({
                'concept_id': concept_id,
                'future_importance': future_importance
            })
            
        return synthetic_data
        
    def _calculate_prediction_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence in prediction based on feature quality"""
        confidence_factors = []
        
        # Data recency
        days_since_access = features.get('days_since_access', 365)
        recency_factor = max(0.0, 1.0 - (days_since_access / 365))
        confidence_factors.append(recency_factor)
        
        # Access frequency
        access_frequency = features.get('access_frequency', 0)
        frequency_factor = min(1.0, access_frequency / 10)
        confidence_factors.append(frequency_factor)
        
        # Relationship connectivity
        total_degree = features.get('total_degree', 0)
        connectivity_factor = min(1.0, total_degree / 5)
        confidence_factors.append(connectivity_factor)
        
        # Content richness
        content_length = features.get('content_length', 0)
        content_factor = min(1.0, content_length / 500)
        confidence_factors.append(content_factor)
        
        return np.mean(confidence_factors)
        
    def _generate_importance_reasoning(self, features: Dict[str, float], 
                                     predicted_importance: float) -> str:
        """Generate human-readable reasoning for importance prediction"""
        current_importance = features.get('current_importance', 0.5)
        change = predicted_importance - current_importance
        
        reasoning_parts = []
        
        if change > 0.1:
            reasoning_parts.append("Expected to increase in importance")
        elif change < -0.1:
            reasoning_parts.append("Expected to decrease in importance")
        else:
            reasoning_parts.append("Expected to maintain current importance level")
            
        # Key contributing factors
        access_rate = features.get('access_rate', 0.0)
        if access_rate > 0.1:
            reasoning_parts.append("due to high access frequency")
        elif access_rate < 0.01:
            reasoning_parts.append("despite low recent access")
            
        relationship_strength = features.get('avg_relationship_strength', 0.5)
        if relationship_strength > 0.7:
            reasoning_parts.append("and strong connections to other concepts")
        elif relationship_strength < 0.3:
            reasoning_parts.append("but weak connections to other concepts")
            
        activity_trend = features.get('activity_trend', 0.0)
        if activity_trend > 0.2:
            reasoning_parts.append("with increasing activity trend")
        elif activity_trend < -0.2:
            reasoning_parts.append("with decreasing activity trend")
            
        return " ".join(reasoning_parts) + "."
        
    def _get_contributing_factors(self, features: Dict[str, float]) -> Dict[str, float]:
        """Get the most important contributing factors"""
        if not self.is_trained:
            return {}
            
        # Get feature importance from trained model
        feature_names = list(features.keys())
        feature_importance = self.model.feature_importances_
        
        # Create factor dictionary
        factors = dict(zip(feature_names, feature_importance))
        
        # Return top 5 factors
        sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_factors[:5])

class KnowledgeGapPredictor:
    """Predict future knowledge gaps based on project requirements and trends"""
    
    def __init__(self, temporal_graph):
        self.temporal_graph = temporal_graph
        self.domain_trends = self._load_domain_trends()
        
    def predict_knowledge_gaps(self, user_id: int, current_projects: List[str],
                             time_horizon_days: int = 90) -> List[KnowledgeGapPrediction]:
        """Predict knowledge gaps for a user"""
        gaps = []
        
        # Analyze current projects
        project_requirements = self._analyze_project_requirements(current_projects)
        
        # Get user's current knowledge
        user_knowledge = self._get_user_knowledge_domains(user_id)
        
        # Identify potential gaps
        for requirement in project_requirements:
            gap_info = self._assess_knowledge_gap(
                requirement, user_knowledge, time_horizon_days
            )
            if gap_info:
                gaps.append(gap_info)
                
        # Add trend-based gaps
        trend_gaps = self._identify_trend_based_gaps(user_knowledge, time_horizon_days)
        gaps.extend(trend_gaps)
        
        # Sort by urgency
        gaps.sort(key=lambda x: x.urgency_score, reverse=True)
        
        return gaps[:10]  # Return top 10 gaps
        
    def _analyze_project_requirements(self, projects: List[str]) -> List[Dict]:
        """Analyze project descriptions to extract knowledge requirements"""
        requirements = []
        
        # Knowledge domain mapping
        domain_keywords = {
            'data_science': ['data', 'analysis', 'statistics', 'machine learning', 'ai'],
            'web_development': ['web', 'frontend', 'backend', 'javascript', 'react', 'api'],
            'design': ['design', 'ui', 'ux', 'user experience', 'interface', 'visual'],
            'project_management': ['project', 'agile', 'scrum', 'management', 'planning'],
            'cloud_computing': ['cloud', 'aws', 'azure', 'deployment', 'infrastructure'],
            'mobile_development': ['mobile', 'ios', 'android', 'app', 'react native'],
            'cybersecurity': ['security', 'encryption', 'authentication', 'privacy'],
            'devops': ['devops', 'ci/cd', 'docker', 'kubernetes', 'automation']
        }
        
        for project in projects:
            project_lower = project.lower()
            project_domains = []
            
            for domain, keywords in domain_keywords.items():
                if any(keyword in project_lower for keyword in keywords):
                    project_domains.append(domain)
                    
            requirements.append({
                'project': project,
                'required_domains': project_domains,
                'complexity': self._estimate_project_complexity(project),
                'timeline': self._estimate_project_timeline(project)
            })
            
        return requirements
        
    def _get_user_knowledge_domains(self, user_id: int) -> Dict[str, float]:
        """Get user's current knowledge strength in different domains"""
        user_concepts = [
            (concept_id, data) for concept_id, data in self.temporal_graph.graph.nodes(data=True)
            if data.get('user_id') == user_id
        ]
        
        domain_strength = defaultdict(list)
        
        for concept_id, data in user_concepts:
            domain_tags = data.get('domain_tags', [])
            importance = data.get('importance_score', 0.5)
            
            for tag in domain_tags:
                domain_strength[tag].append(importance)
                
        # Calculate average strength per domain
        domain_scores = {}
        for domain, scores in domain_strength.items():
            domain_scores[domain] = np.mean(scores)
            
        return domain_scores
        
    def _assess_knowledge_gap(self, requirement: Dict, user_knowledge: Dict[str, float],
                            time_horizon_days: int) -> Optional[KnowledgeGapPrediction]:
        """Assess if there's a knowledge gap for a specific requirement"""
        required_domains = requirement['required_domains']
        project_complexity = requirement['complexity']
        
        # Calculate knowledge coverage
        coverage_scores = []
        missing_domains = []
        
        for domain in required_domains:
            user_strength = user_knowledge.get(domain, 0.0)
            required_strength = 0.6 + (project_complexity * 0.2)  # Higher complexity needs more knowledge
            
            if user_strength < required_strength:
                gap_size = required_strength - user_strength
                coverage_scores.append(user_strength / required_strength)
                missing_domains.append((domain, gap_size))
            else:
                coverage_scores.append(1.0)
                
        if not missing_domains:
            return None  # No gap identified
            
        # Calculate urgency based on project timeline and gap size
        avg_coverage = np.mean(coverage_scores) if coverage_scores else 0.0
        urgency_score = (1.0 - avg_coverage) * (1.0 / max(1, requirement['timeline'] / 30))
        
        # Find the most critical missing domain
        critical_domain = max(missing_domains, key=lambda x: x[1])
        
        # Generate recommendations
        recommendations = self._generate_gap_recommendations(critical_domain[0], critical_domain[1])
        
        # Calculate predicted need date
        predicted_need_date = datetime.utcnow() + timedelta(days=requirement['timeline'] - 14)
        
        return KnowledgeGapPrediction(
            gap_topic=critical_domain[0].replace('_', ' ').title(),
            urgency_score=min(1.0, urgency_score),
            predicted_need_date=predicted_need_date,
            confidence=0.7,
            context=f"Required for project: {requirement['project'][:100]}...",
            recommendations=recommendations
        )
        
    def _identify_trend_based_gaps(self, user_knowledge: Dict[str, float],
                                 time_horizon_days: int) -> List[KnowledgeGapPrediction]:
        """Identify gaps based on industry trends"""
        trend_gaps = []
        
        # Check trending domains against user knowledge
        for domain, trend_data in self.domain_trends.items():
            if trend_data['growth_rate'] > 0.2:  # High growth domains
                user_strength = user_knowledge.get(domain, 0.0)
                
                if user_strength < 0.5:  # User has low knowledge in trending domain
                    urgency = trend_data['growth_rate'] * (1.0 - user_strength)
                    
                    predicted_need_date = datetime.utcnow() + timedelta(
                        days=int(time_horizon_days * (1.0 - trend_data['growth_rate']))
                    )
                    
                    recommendations = self._generate_trend_recommendations(domain, trend_data)
                    
                    trend_gaps.append(KnowledgeGapPrediction(
                        gap_topic=domain.replace('_', ' ').title(),
                        urgency_score=min(1.0, urgency),
                        predicted_need_date=predicted_need_date,
                        confidence=0.6,
                        context=f"Trending domain with {trend_data['growth_rate']:.1%} growth",
                        recommendations=recommendations
                    ))
                    
        return trend_gaps
        
    def _load_domain_trends(self) -> Dict[str, Dict]:
        """Load domain trend data (simplified implementation)"""
        # In practice, this would load from external trend APIs or databases
        return {
            'artificial_intelligence': {'growth_rate': 0.35, 'relevance': 0.9},
            'machine_learning': {'growth_rate': 0.30, 'relevance': 0.85},
            'cloud_computing': {'growth_rate': 0.25, 'relevance': 0.8},
            'cybersecurity': {'growth_rate': 0.28, 'relevance': 0.82},
            'data_science': {'growth_rate': 0.22, 'relevance': 0.78},
            'blockchain': {'growth_rate': 0.15, 'relevance': 0.6},
            'quantum_computing': {'growth_rate': 0.40, 'relevance': 0.7},
            'sustainability': {'growth_rate': 0.32, 'relevance': 0.75},
            'remote_work': {'growth_rate': 0.20, 'relevance': 0.85},
            'user_experience': {'growth_rate': 0.18, 'relevance': 0.8}
        }
        
    def _estimate_project_complexity(self, project_description: str) -> float:
        """Estimate project complexity (0.0 to 1.0)"""
        complexity_indicators = [
            'complex', 'advanced', 'enterprise', 'scalable', 'distributed',
            'machine learning', 'ai', 'big data', 'real-time', 'high-performance'
        ]
        
        project_lower = project_description.lower()
        complexity_score = sum(
            1 for indicator in complexity_indicators
            if indicator in project_lower
        ) / len(complexity_indicators)
        
        # Also consider project length as complexity indicator
        length_factor = min(1.0, len(project_description) / 500)
        
        return min(1.0, (complexity_score + length_factor) / 2)
        
    def _estimate_project_timeline(self, project_description: str) -> int:
        """Estimate project timeline in days"""
        timeline_keywords = {
            'urgent': 7,
            'asap': 7,
            'week': 7,
            'month': 30,
            'quarter': 90,
            'semester': 120,
            'year': 365
        }
        
        project_lower = project_description.lower()
        
        for keyword, days in timeline_keywords.items():
            if keyword in project_lower:
                return days
                
        # Default timeline based on complexity
        complexity = self._estimate_project_complexity(project_description)
        return int(30 + (complexity * 60))  # 30-90 days based on complexity
        
    def _generate_gap_recommendations(self, domain: str, gap_size: float) -> List[str]:
        """Generate recommendations for addressing a knowledge gap"""
        recommendations = []
        
        if gap_size > 0.5:  # Large gap
            recommendations.extend([
                f"Start with fundamentals of {domain.replace('_', ' ')}",
                f"Take a comprehensive course on {domain.replace('_', ' ')}",
                f"Allocate 5-10 hours per week for {domain.replace('_', ' ')} learning"
            ])
        else:  # Smaller gap
            recommendations.extend([
                f"Review key concepts in {domain.replace('_', ' ')}",
                f"Practice hands-on projects in {domain.replace('_', ' ')}",
                f"Allocate 2-3 hours per week for {domain.replace('_', ' ')} improvement"
            ])
            
        # Add domain-specific recommendations
        domain_specific = {
            'data_science': ["Learn pandas and numpy", "Practice with real datasets"],
            'web_development': ["Build a portfolio project", "Learn modern frameworks"],
            'machine_learning': ["Implement algorithms from scratch", "Work on Kaggle competitions"],
            'design': ["Study design principles", "Create a design portfolio"],
            'project_management': ["Get familiar with Agile methodologies", "Practice with project management tools"]
        }
        
        if domain in domain_specific:
            recommendations.extend(domain_specific[domain])
            
        return recommendations[:4]  # Return top 4 recommendations
        
    def _generate_trend_recommendations(self, domain: str, trend_data: Dict) -> List[str]:
        """Generate recommendations for trending domain gaps"""
        growth_rate = trend_data['growth_rate']
        
        recommendations = [
            f"Stay updated with latest developments in {domain.replace('_', ' ')}",
            f"Follow industry leaders and publications in {domain.replace('_', ' ')}",
        ]
        
        if growth_rate > 0.3:  # Very high growth
            recommendations.extend([
                f"Consider specializing in {domain.replace('_', ' ')} for career advancement",
                f"Join professional communities focused on {domain.replace('_', ' ')}"
            ])
        else:
            recommendations.extend([
                f"Build foundational knowledge in {domain.replace('_', ' ')}",
                f"Explore practical applications of {domain.replace('_', ' ')}"
            ])
            
        return recommendations

class PredictiveAnalyticsEngine:
    """Main engine coordinating all predictive analytics components"""
    
    def __init__(self, temporal_graph):
        self.temporal_graph = temporal_graph
        self.feature_extractor = FeatureExtractor(temporal_graph)
        self.importance_predictor = ImportancePredictor(temporal_graph, self.feature_extractor)
        self.gap_predictor = KnowledgeGapPredictor(temporal_graph)
        
    def initialize_models(self) -> Dict:
        """Initialize and train all predictive models"""
        results = {}
        
        # Train importance prediction model
        training_results = self.importance_predictor.train_model([])
        results['importance_predictor'] = training_results
        
        return results
        
    def predict_concept_importance(self, concept_id: str, months_ahead: int = 6) -> PredictionResult:
        """Predict future importance of a concept"""
        return self.importance_predictor.predict_importance(concept_id, months_ahead)
        
    def predict_knowledge_gaps(self, user_id: int, current_projects: List[str],
                             time_horizon_days: int = 90) -> List[KnowledgeGapPrediction]:
        """Predict knowledge gaps for a user"""
        return self.gap_predictor.predict_knowledge_gaps(user_id, current_projects, time_horizon_days)
        
    def get_learning_recommendations(self, user_id: int, focus_areas: List[str]) -> List[Dict]:
        """Get personalized learning recommendations"""
        recommendations = []
        
        # Get user's current knowledge state
        user_knowledge = self.gap_predictor._get_user_knowledge_domains(user_id)
        
        # Analyze focus areas
        for area in focus_areas:
            current_strength = user_knowledge.get(area, 0.0)
            
            if current_strength < 0.7:  # Room for improvement
                recommendation = {
                    'area': area.replace('_', ' ').title(),
                    'current_strength': current_strength,
                    'target_strength': 0.8,
                    'priority': 1.0 - current_strength,
                    'estimated_time_weeks': int((0.8 - current_strength) * 20),
                    'learning_path': self._generate_learning_path(area, current_strength)
                }
                recommendations.append(recommendation)
                
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'], reverse=True)
        
        return recommendations[:5]  # Return top 5 recommendations
        
    def _generate_learning_path(self, area: str, current_strength: float) -> List[Dict]:
        """Generate a structured learning path for an area"""
        learning_paths = {
            'machine_learning': [
                {'step': 'Statistics and Probability', 'duration_weeks': 2, 'difficulty': 'beginner'},
                {'step': 'Python for Data Science', 'duration_weeks': 3, 'difficulty': 'beginner'},
                {'step': 'Supervised Learning Algorithms', 'duration_weeks': 4, 'difficulty': 'intermediate'},
                {'step': 'Unsupervised Learning', 'duration_weeks': 3, 'difficulty': 'intermediate'},
                {'step': 'Deep Learning Fundamentals', 'duration_weeks': 4, 'difficulty': 'advanced'},
                {'step': 'MLOps and Deployment', 'duration_weeks': 3, 'difficulty': 'advanced'}
            ],
            'web_development': [
                {'step': 'HTML/CSS Fundamentals', 'duration_weeks': 2, 'difficulty': 'beginner'},
                {'step': 'JavaScript Basics', 'duration_weeks': 3, 'difficulty': 'beginner'},
                {'step': 'React Framework', 'duration_weeks': 4, 'difficulty': 'intermediate'},
                {'step': 'Backend Development', 'duration_weeks': 4, 'difficulty': 'intermediate'},
                {'step': 'Database Design', 'duration_weeks': 2, 'difficulty': 'intermediate'},
                {'step': 'Full-Stack Projects', 'duration_weeks': 4, 'difficulty': 'advanced'}
            ]
        }
        
        default_path = [
            {'step': f'Fundamentals of {area.replace("_", " ").title()}', 'duration_weeks': 3, 'difficulty': 'beginner'},
            {'step': f'Intermediate {area.replace("_", " ").title()}', 'duration_weeks': 4, 'difficulty': 'intermediate'},
            {'step': f'Advanced {area.replace("_", " ").title()}', 'duration_weeks': 4, 'difficulty': 'advanced'},
            {'step': f'Practical Projects in {area.replace("_", " ").title()}', 'duration_weeks': 3, 'difficulty': 'advanced'}
        ]
        
        path = learning_paths.get(area, default_path)
        
        # Adjust path based on current strength
        if current_strength > 0.3:  # Skip beginner if already have some knowledge
            path = [step for step in path if step['difficulty'] != 'beginner']
        if current_strength > 0.6:  # Focus on advanced if already intermediate
            path = [step for step in path if step['difficulty'] == 'advanced']
            
        return path

