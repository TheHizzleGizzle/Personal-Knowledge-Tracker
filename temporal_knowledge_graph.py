"""
Temporal Knowledge Graph Implementation
======================================

This module implements a temporal knowledge graph system that tracks how knowledge
items, concepts, and their relationships evolve over time. It enables the AI system
to understand knowledge patterns, predict future relevance, and identify dormant
knowledge that becomes relevant again.

Key Features:
- Time-aware knowledge representation
- Concept relationship tracking
- Knowledge evolution analysis
- Temporal pattern recognition
"""

import networkx as nx
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from collections import defaultdict
import json
from dataclasses import dataclass, asdict
from enum import Enum

class RelationType(Enum):
    """Types of relationships between knowledge concepts"""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    CAUSAL_RELATIONSHIP = "causal_relationship"
    TEMPORAL_SEQUENCE = "temporal_sequence"
    CONCEPTUAL_HIERARCHY = "conceptual_hierarchy"
    APPLICATION_CONTEXT = "application_context"
    CROSS_DOMAIN = "cross_domain"

@dataclass
class TemporalNode:
    """Represents a knowledge concept with temporal information"""
    concept_id: str
    concept_name: str
    created_at: datetime
    last_accessed: datetime
    access_frequency: int
    importance_score: float
    domain_tags: List[str]
    content_summary: str
    metadata: Dict

@dataclass
class TemporalEdge:
    """Represents a relationship between concepts with temporal dynamics"""
    source_id: str
    target_id: str
    relation_type: RelationType
    strength: float
    created_at: datetime
    last_reinforced: datetime
    reinforcement_count: int
    context: str
    confidence: float

class TemporalKnowledgeGraph:
    """
    A temporal knowledge graph that tracks knowledge evolution over time.
    
    This graph maintains both the structural relationships between concepts
    and their temporal dynamics, enabling sophisticated analysis of knowledge
    patterns and predictions about future relevance.
    """
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.temporal_snapshots = {}  # timestamp -> graph state
        self.concept_timeline = defaultdict(list)  # concept_id -> [(timestamp, event)]
        self.relationship_timeline = defaultdict(list)  # (source, target) -> [(timestamp, event)]
        
    def add_concept(self, concept: TemporalNode) -> None:
        """Add a new concept to the temporal knowledge graph"""
        self.graph.add_node(
            concept.concept_id,
            **asdict(concept)
        )
        
        # Record temporal event
        event = {
            'type': 'concept_added',
            'timestamp': concept.created_at,
            'concept_id': concept.concept_id,
            'importance': concept.importance_score
        }
        self.concept_timeline[concept.concept_id].append(event)
        
    def add_relationship(self, edge: TemporalEdge) -> None:
        """Add a relationship between concepts"""
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            key=edge.relation_type.value,
            **asdict(edge)
        )
        
        # Record temporal event
        event = {
            'type': 'relationship_added',
            'timestamp': edge.created_at,
            'source_id': edge.source_id,
            'target_id': edge.target_id,
            'relation_type': edge.relation_type.value,
            'strength': edge.strength
        }
        self.relationship_timeline[(edge.source_id, edge.target_id)].append(event)
        
    def update_concept_access(self, concept_id: str, access_time: datetime) -> None:
        """Update concept access information"""
        if concept_id in self.graph.nodes:
            node_data = self.graph.nodes[concept_id]
            node_data['last_accessed'] = access_time
            node_data['access_frequency'] += 1
            
            # Record temporal event
            event = {
                'type': 'concept_accessed',
                'timestamp': access_time,
                'concept_id': concept_id,
                'access_count': node_data['access_frequency']
            }
            self.concept_timeline[concept_id].append(event)
            
    def reinforce_relationship(self, source_id: str, target_id: str, 
                             relation_type: RelationType, context: str) -> None:
        """Reinforce an existing relationship or create a new one"""
        edge_key = relation_type.value
        
        if self.graph.has_edge(source_id, target_id, key=edge_key):
            # Reinforce existing relationship
            edge_data = self.graph.edges[source_id, target_id, edge_key]
            edge_data['last_reinforced'] = datetime.utcnow()
            edge_data['reinforcement_count'] += 1
            edge_data['strength'] = min(1.0, edge_data['strength'] + 0.1)
            
            # Record temporal event
            event = {
                'type': 'relationship_reinforced',
                'timestamp': datetime.utcnow(),
                'source_id': source_id,
                'target_id': target_id,
                'relation_type': relation_type.value,
                'new_strength': edge_data['strength'],
                'context': context
            }
            self.relationship_timeline[(source_id, target_id)].append(event)
        else:
            # Create new relationship
            new_edge = TemporalEdge(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                strength=0.5,
                created_at=datetime.utcnow(),
                last_reinforced=datetime.utcnow(),
                reinforcement_count=1,
                context=context,
                confidence=0.7
            )
            self.add_relationship(new_edge)
            
    def get_concept_evolution(self, concept_id: str, 
                            time_window: timedelta = timedelta(days=365)) -> Dict:
        """Analyze how a concept has evolved over time"""
        if concept_id not in self.concept_timeline:
            return {}
            
        current_time = datetime.utcnow()
        start_time = current_time - time_window
        
        events = [
            event for event in self.concept_timeline[concept_id]
            if event['timestamp'] >= start_time
        ]
        
        # Calculate evolution metrics
        access_pattern = [e for e in events if e['type'] == 'concept_accessed']
        access_frequency = len(access_pattern)
        
        # Identify periods of high/low activity
        activity_periods = self._identify_activity_periods(access_pattern)
        
        # Calculate importance trajectory
        importance_trajectory = self._calculate_importance_trajectory(concept_id, events)
        
        return {
            'concept_id': concept_id,
            'total_events': len(events),
            'access_frequency': access_frequency,
            'activity_periods': activity_periods,
            'importance_trajectory': importance_trajectory,
            'current_importance': self.graph.nodes[concept_id]['importance_score'],
            'dormancy_risk': self._calculate_dormancy_risk(concept_id),
            'revival_potential': self._calculate_revival_potential(concept_id)
        }
        
    def predict_concept_importance(self, concept_id: str, 
                                 future_months: int = 6) -> Dict:
        """Predict future importance of a concept"""
        if concept_id not in self.graph.nodes:
            return {}
            
        # Get historical data
        evolution = self.get_concept_evolution(concept_id)
        
        # Analyze trends
        importance_trend = self._analyze_importance_trend(concept_id)
        relationship_strength_trend = self._analyze_relationship_trends(concept_id)
        domain_relevance_trend = self._analyze_domain_trends(concept_id)
        
        # Calculate prediction
        base_importance = self.graph.nodes[concept_id]['importance_score']
        trend_factor = importance_trend['slope']
        relationship_factor = relationship_strength_trend['average_growth']
        domain_factor = domain_relevance_trend['relevance_score']
        
        # Weighted prediction
        predicted_importance = base_importance + (
            0.4 * trend_factor +
            0.3 * relationship_factor +
            0.3 * domain_factor
        ) * (future_months / 6)  # Scale by time horizon
        
        predicted_importance = max(0.0, min(1.0, predicted_importance))
        
        # Calculate confidence based on data quality
        confidence = self._calculate_prediction_confidence(concept_id, evolution)
        
        return {
            'concept_id': concept_id,
            'current_importance': base_importance,
            'predicted_importance': predicted_importance,
            'change_magnitude': predicted_importance - base_importance,
            'confidence': confidence,
            'prediction_horizon_months': future_months,
            'key_factors': {
                'importance_trend': trend_factor,
                'relationship_strength': relationship_factor,
                'domain_relevance': domain_factor
            },
            'reasoning': self._generate_prediction_reasoning(
                concept_id, trend_factor, relationship_factor, domain_factor
            )
        }
        
    def discover_dormant_knowledge(self, current_context: List[str]) -> List[Dict]:
        """Identify dormant knowledge that's relevant to current context"""
        dormant_concepts = []
        current_time = datetime.utcnow()
        
        for concept_id in self.graph.nodes:
            node_data = self.graph.nodes[concept_id]
            
            # Check if concept is dormant (not accessed recently)
            last_access = node_data.get('last_accessed', node_data['created_at'])
            days_since_access = (current_time - last_access).days
            
            if days_since_access > 30:  # Consider dormant if not accessed in 30 days
                # Calculate relevance to current context
                relevance_score = self._calculate_context_relevance(
                    concept_id, current_context
                )
                
                if relevance_score > 0.6:  # High relevance threshold
                    dormant_concepts.append({
                        'concept_id': concept_id,
                        'concept_name': node_data['concept_name'],
                        'days_dormant': days_since_access,
                        'relevance_score': relevance_score,
                        'original_importance': node_data['importance_score'],
                        'context_connections': self._get_context_connections(
                            concept_id, current_context
                        ),
                        'revival_reasoning': self._generate_revival_reasoning(
                            concept_id, current_context, relevance_score
                        )
                    })
        
        # Sort by relevance score
        dormant_concepts.sort(key=lambda x: x['relevance_score'], reverse=True)
        return dormant_concepts[:10]  # Return top 10
        
    def identify_knowledge_gaps(self, current_projects: List[str]) -> List[Dict]:
        """Identify potential knowledge gaps based on current projects"""
        gaps = []
        
        # Analyze current project requirements
        for project in current_projects:
            project_concepts = self._extract_project_concepts(project)
            
            # Find missing connections
            for concept in project_concepts:
                if concept not in self.graph.nodes:
                    # This is a potential gap - concept mentioned but not in knowledge base
                    gap_info = {
                        'gap_type': 'missing_concept',
                        'concept': concept,
                        'project_context': project,
                        'urgency_score': 0.8,
                        'predicted_need_date': datetime.utcnow() + timedelta(days=14),
                        'confidence': 0.7,
                        'recommendations': [
                            f"Research fundamentals of {concept}",
                            f"Find authoritative sources on {concept}",
                            f"Connect {concept} to existing knowledge"
                        ]
                    }
                    gaps.append(gap_info)
                else:
                    # Check for weak connections
                    weak_connections = self._find_weak_connections(concept, project_concepts)
                    if weak_connections:
                        gap_info = {
                            'gap_type': 'weak_connections',
                            'concept': concept,
                            'weak_connections': weak_connections,
                            'project_context': project,
                            'urgency_score': 0.6,
                            'predicted_need_date': datetime.utcnow() + timedelta(days=30),
                            'confidence': 0.6,
                            'recommendations': [
                                f"Strengthen understanding of connections between {concept} and related concepts",
                                f"Review practical applications of {concept}",
                                f"Study case studies involving {concept}"
                            ]
                        }
                        gaps.append(gap_info)
        
        # Sort by urgency
        gaps.sort(key=lambda x: x['urgency_score'], reverse=True)
        return gaps
        
    def create_temporal_snapshot(self, timestamp: datetime = None) -> str:
        """Create a snapshot of the current graph state"""
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        snapshot_id = timestamp.isoformat()
        
        # Create a deep copy of current graph state
        snapshot = {
            'timestamp': timestamp,
            'nodes': dict(self.graph.nodes(data=True)),
            'edges': [
                {
                    'source': u,
                    'target': v,
                    'key': k,
                    'data': d
                }
                for u, v, k, d in self.graph.edges(keys=True, data=True)
            ],
            'metrics': {
                'total_concepts': self.graph.number_of_nodes(),
                'total_relationships': self.graph.number_of_edges(),
                'average_importance': np.mean([
                    data['importance_score'] 
                    for _, data in self.graph.nodes(data=True)
                ]) if self.graph.nodes else 0
            }
        }
        
        self.temporal_snapshots[snapshot_id] = snapshot
        return snapshot_id
        
    def analyze_knowledge_evolution(self, start_date: datetime, 
                                  end_date: datetime) -> Dict:
        """Analyze how knowledge has evolved between two time points"""
        # Find snapshots within the time range
        relevant_snapshots = [
            (timestamp, snapshot)
            for timestamp, snapshot in self.temporal_snapshots.items()
            if start_date <= snapshot['timestamp'] <= end_date
        ]
        
        if len(relevant_snapshots) < 2:
            return {'error': 'Insufficient temporal data for analysis'}
            
        # Sort by timestamp
        relevant_snapshots.sort(key=lambda x: x[1]['timestamp'])
        
        start_snapshot = relevant_snapshots[0][1]
        end_snapshot = relevant_snapshots[-1][1]
        
        # Calculate evolution metrics
        concept_growth = (
            end_snapshot['metrics']['total_concepts'] - 
            start_snapshot['metrics']['total_concepts']
        )
        
        relationship_growth = (
            end_snapshot['metrics']['total_relationships'] - 
            start_snapshot['metrics']['total_relationships']
        )
        
        importance_change = (
            end_snapshot['metrics']['average_importance'] - 
            start_snapshot['metrics']['average_importance']
        )
        
        return {
            'analysis_period': {
                'start_date': start_date,
                'end_date': end_date,
                'duration_days': (end_date - start_date).days
            },
            'growth_metrics': {
                'concept_growth': concept_growth,
                'relationship_growth': relationship_growth,
                'importance_change': importance_change
            },
            'evolution_patterns': self._identify_evolution_patterns(relevant_snapshots),
            'emerging_domains': self._identify_emerging_domains(relevant_snapshots),
            'declining_concepts': self._identify_declining_concepts(relevant_snapshots)
        }
    
    # Helper methods
    
    def _identify_activity_periods(self, access_events: List[Dict]) -> List[Dict]:
        """Identify periods of high and low activity"""
        if not access_events:
            return []
            
        # Group events by week
        weekly_activity = defaultdict(int)
        for event in access_events:
            week = event['timestamp'].isocalendar()[:2]  # (year, week)
            weekly_activity[week] += 1
            
        # Identify high/low activity periods
        activity_values = list(weekly_activity.values())
        if not activity_values:
            return []
            
        mean_activity = np.mean(activity_values)
        std_activity = np.std(activity_values) if len(activity_values) > 1 else 0
        
        periods = []
        for week, activity in weekly_activity.items():
            if activity > mean_activity + std_activity:
                periods.append({
                    'period': f"{week[0]}-W{week[1]}",
                    'type': 'high_activity',
                    'activity_level': activity
                })
            elif activity < mean_activity - std_activity:
                periods.append({
                    'period': f"{week[0]}-W{week[1]}",
                    'type': 'low_activity',
                    'activity_level': activity
                })
                
        return periods
        
    def _calculate_importance_trajectory(self, concept_id: str, events: List[Dict]) -> List[Dict]:
        """Calculate how importance has changed over time"""
        # This is a simplified implementation
        # In practice, you'd track importance changes more explicitly
        trajectory = []
        current_importance = self.graph.nodes[concept_id]['importance_score']
        
        # Simulate trajectory based on access patterns
        for i, event in enumerate(events[-10:]):  # Last 10 events
            trajectory.append({
                'timestamp': event['timestamp'],
                'importance': current_importance * (0.8 + 0.4 * (i / 10))
            })
            
        return trajectory
        
    def _calculate_dormancy_risk(self, concept_id: str) -> float:
        """Calculate risk of concept becoming dormant"""
        node_data = self.graph.nodes[concept_id]
        current_time = datetime.utcnow()
        
        # Time since last access
        last_access = node_data.get('last_accessed', node_data['created_at'])
        days_since_access = (current_time - last_access).days
        
        # Access frequency
        access_freq = node_data.get('access_frequency', 1)
        
        # Relationship strength
        avg_relationship_strength = np.mean([
            data.get('strength', 0.5)
            for _, _, data in self.graph.edges(concept_id, data=True)
        ]) if self.graph.edges(concept_id) else 0.3
        
        # Calculate risk (higher values = higher risk)
        time_risk = min(1.0, days_since_access / 90)  # 90 days = max risk
        frequency_risk = max(0.0, 1.0 - (access_freq / 10))  # 10+ accesses = low risk
        connection_risk = 1.0 - avg_relationship_strength
        
        return (time_risk + frequency_risk + connection_risk) / 3
        
    def _calculate_revival_potential(self, concept_id: str) -> float:
        """Calculate potential for concept to become relevant again"""
        node_data = self.graph.nodes[concept_id]
        
        # Original importance
        original_importance = node_data['importance_score']
        
        # Number of connections
        connection_count = len(list(self.graph.edges(concept_id)))
        
        # Domain diversity
        domain_tags = node_data.get('domain_tags', [])
        domain_diversity = len(set(domain_tags)) / max(1, len(domain_tags))
        
        # Calculate potential
        importance_factor = original_importance
        connection_factor = min(1.0, connection_count / 5)  # 5+ connections = high potential
        diversity_factor = domain_diversity
        
        return (importance_factor + connection_factor + diversity_factor) / 3
        
    def _analyze_importance_trend(self, concept_id: str) -> Dict:
        """Analyze the trend in concept importance over time"""
        events = self.concept_timeline.get(concept_id, [])
        
        # Simplified trend analysis
        if len(events) < 2:
            return {'slope': 0.0, 'confidence': 0.3}
            
        # Calculate access frequency trend
        recent_accesses = len([
            e for e in events 
            if e['type'] == 'concept_accessed' and 
            (datetime.utcnow() - e['timestamp']).days <= 30
        ])
        
        older_accesses = len([
            e for e in events 
            if e['type'] == 'concept_accessed' and 
            30 < (datetime.utcnow() - e['timestamp']).days <= 60
        ])
        
        if older_accesses == 0:
            slope = 0.1 if recent_accesses > 0 else 0.0
        else:
            slope = (recent_accesses - older_accesses) / older_accesses
            
        return {
            'slope': max(-1.0, min(1.0, slope)),
            'confidence': 0.7 if len(events) > 5 else 0.4
        }
        
    def _analyze_relationship_trends(self, concept_id: str) -> Dict:
        """Analyze trends in relationship strengths"""
        relationships = list(self.graph.edges(concept_id, data=True))
        
        if not relationships:
            return {'average_growth': 0.0, 'confidence': 0.2}
            
        # Calculate average relationship strength
        avg_strength = np.mean([data.get('strength', 0.5) for _, _, data in relationships])
        
        # Simplified growth calculation
        recent_reinforcements = sum(
            1 for _, _, data in relationships
            if (datetime.utcnow() - data.get('last_reinforced', datetime.utcnow())).days <= 30
        )
        
        growth_factor = recent_reinforcements / len(relationships)
        
        return {
            'average_growth': growth_factor - 0.5,  # Normalize around 0
            'confidence': 0.6 if len(relationships) > 3 else 0.3
        }
        
    def _analyze_domain_trends(self, concept_id: str) -> Dict:
        """Analyze trends in domain relevance"""
        node_data = self.graph.nodes[concept_id]
        domain_tags = node_data.get('domain_tags', [])
        
        # Simplified domain relevance calculation
        # In practice, this would analyze external trend data
        relevance_score = 0.5  # Neutral baseline
        
        # Boost score for certain trending domains
        trending_domains = ['ai', 'machine_learning', 'sustainability', 'remote_work']
        for tag in domain_tags:
            if any(trend in tag.lower() for trend in trending_domains):
                relevance_score += 0.2
                
        return {
            'relevance_score': min(1.0, relevance_score),
            'confidence': 0.5
        }
        
    def _calculate_prediction_confidence(self, concept_id: str, evolution: Dict) -> float:
        """Calculate confidence in prediction based on data quality"""
        factors = []
        
        # Data volume
        event_count = evolution.get('total_events', 0)
        factors.append(min(1.0, event_count / 20))  # 20+ events = high confidence
        
        # Time span
        node_data = self.graph.nodes[concept_id]
        age_days = (datetime.utcnow() - node_data['created_at']).days
        factors.append(min(1.0, age_days / 180))  # 6+ months = high confidence
        
        # Relationship count
        relationship_count = len(list(self.graph.edges(concept_id)))
        factors.append(min(1.0, relationship_count / 5))  # 5+ relationships = high confidence
        
        return np.mean(factors)
        
    def _generate_prediction_reasoning(self, concept_id: str, trend_factor: float,
                                     relationship_factor: float, domain_factor: float) -> str:
        """Generate human-readable reasoning for prediction"""
        node_data = self.graph.nodes[concept_id]
        concept_name = node_data['concept_name']
        
        reasoning_parts = []
        
        if trend_factor > 0.1:
            reasoning_parts.append(f"'{concept_name}' shows increasing access patterns")
        elif trend_factor < -0.1:
            reasoning_parts.append(f"'{concept_name}' shows declining access patterns")
        else:
            reasoning_parts.append(f"'{concept_name}' shows stable access patterns")
            
        if relationship_factor > 0.1:
            reasoning_parts.append("strengthening connections to other concepts")
        elif relationship_factor < -0.1:
            reasoning_parts.append("weakening connections to other concepts")
            
        if domain_factor > 0.6:
            reasoning_parts.append("high relevance in trending domains")
        elif domain_factor < 0.4:
            reasoning_parts.append("lower relevance in current domain trends")
            
        return ". ".join(reasoning_parts) + "."
        
    def _calculate_context_relevance(self, concept_id: str, context: List[str]) -> float:
        """Calculate how relevant a concept is to current context"""
        node_data = self.graph.nodes[concept_id]
        
        # Check domain overlap
        domain_tags = set(node_data.get('domain_tags', []))
        context_set = set(tag.lower() for tag in context)
        
        domain_overlap = len(domain_tags.intersection(context_set)) / max(1, len(context_set))
        
        # Check content similarity (simplified)
        content_summary = node_data.get('content_summary', '').lower()
        content_overlap = sum(
            1 for term in context
            if term.lower() in content_summary
        ) / max(1, len(context))
        
        # Check relationship connections
        relationship_relevance = 0.0
        for neighbor in self.graph.neighbors(concept_id):
            neighbor_data = self.graph.nodes[neighbor]
            neighbor_tags = set(neighbor_data.get('domain_tags', []))
            if neighbor_tags.intersection(context_set):
                relationship_relevance += 0.1
                
        relationship_relevance = min(1.0, relationship_relevance)
        
        # Weighted combination
        return (0.4 * domain_overlap + 0.4 * content_overlap + 0.2 * relationship_relevance)
        
    def _get_context_connections(self, concept_id: str, context: List[str]) -> List[str]:
        """Get connections between concept and context"""
        connections = []
        
        # Direct domain connections
        node_data = self.graph.nodes[concept_id]
        domain_tags = node_data.get('domain_tags', [])
        
        for tag in domain_tags:
            if any(ctx.lower() in tag.lower() or tag.lower() in ctx.lower() for ctx in context):
                connections.append(f"Domain connection: {tag}")
                
        # Relationship connections
        for neighbor in self.graph.neighbors(concept_id):
            neighbor_data = self.graph.nodes[neighbor]
            neighbor_tags = neighbor_data.get('domain_tags', [])
            
            for tag in neighbor_tags:
                if any(ctx.lower() in tag.lower() for ctx in context):
                    connections.append(f"Connected via: {neighbor_data['concept_name']}")
                    break
                    
        return connections[:3]  # Return top 3 connections
        
    def _generate_revival_reasoning(self, concept_id: str, context: List[str], 
                                  relevance_score: float) -> str:
        """Generate reasoning for why dormant knowledge is relevant"""
        node_data = self.graph.nodes[concept_id]
        concept_name = node_data['concept_name']
        
        if relevance_score > 0.8:
            return f"'{concept_name}' has strong connections to your current focus areas and could provide valuable insights."
        elif relevance_score > 0.6:
            return f"'{concept_name}' shares important concepts with your current work and may offer useful perspectives."
        else:
            return f"'{concept_name}' has some relevance to your current context and might be worth revisiting."
            
    def _extract_project_concepts(self, project_description: str) -> List[str]:
        """Extract key concepts from project description"""
        # Simplified concept extraction
        # In practice, this would use NLP techniques
        common_concepts = [
            'machine learning', 'data analysis', 'visualization', 'statistics',
            'user experience', 'design', 'development', 'testing', 'deployment',
            'agile', 'scrum', 'project management', 'communication', 'leadership'
        ]
        
        project_lower = project_description.lower()
        found_concepts = [
            concept for concept in common_concepts
            if concept in project_lower
        ]
        
        return found_concepts
        
    def _find_weak_connections(self, concept: str, project_concepts: List[str]) -> List[str]:
        """Find weak connections between concept and project concepts"""
        if concept not in self.graph.nodes:
            return []
            
        weak_connections = []
        
        for other_concept in project_concepts:
            if other_concept != concept and other_concept in self.graph.nodes:
                # Check if there's a direct connection
                if not self.graph.has_edge(concept, other_concept):
                    weak_connections.append(other_concept)
                else:
                    # Check connection strength
                    edges = self.graph.edges[concept, other_concept]
                    avg_strength = np.mean([data.get('strength', 0.5) for data in edges.values()])
                    if avg_strength < 0.4:
                        weak_connections.append(other_concept)
                        
        return weak_connections
        
    def _identify_evolution_patterns(self, snapshots: List[Tuple]) -> List[Dict]:
        """Identify patterns in knowledge evolution"""
        patterns = []
        
        # Simplified pattern identification
        if len(snapshots) >= 3:
            # Check for accelerating growth
            growth_rates = []
            for i in range(1, len(snapshots)):
                prev_concepts = snapshots[i-1][1]['metrics']['total_concepts']
                curr_concepts = snapshots[i][1]['metrics']['total_concepts']
                growth_rate = (curr_concepts - prev_concepts) / max(1, prev_concepts)
                growth_rates.append(growth_rate)
                
            if len(growth_rates) >= 2 and growth_rates[-1] > growth_rates[-2]:
                patterns.append({
                    'type': 'accelerating_growth',
                    'description': 'Knowledge acquisition is accelerating',
                    'confidence': 0.7
                })
                
        return patterns
        
    def _identify_emerging_domains(self, snapshots: List[Tuple]) -> List[str]:
        """Identify emerging knowledge domains"""
        # Simplified implementation
        # In practice, this would analyze domain tag frequencies over time
        return ['artificial_intelligence', 'sustainability', 'remote_collaboration']
        
    def _identify_declining_concepts(self, snapshots: List[Tuple]) -> List[str]:
        """Identify concepts that are declining in importance"""
        # Simplified implementation
        # In practice, this would track importance scores over time
        return ['legacy_systems', 'traditional_marketing']

