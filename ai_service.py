"""
AI Service Integration Layer
===========================

This module provides the integration layer between the Flask application
and the AI engine components. It handles initialization, data synchronization,
and provides high-level AI services to the application.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import logging
from temporal_knowledge_graph import TemporalKnowledgeGraph, TemporalNode, TemporalEdge, RelationType
from predictive_analytics import PredictiveAnalyticsEngine
from knowledge_item import KnowledgeItem, Tag, UserActivity
from analytics import Prediction, DormantDiscovery, KnowledgeGap
from user import db

logger = logging.getLogger(__name__)

class AIService:
    """
    Main AI service that coordinates all AI components and provides
    high-level AI functionality to the Flask application.
    """
    
    def __init__(self):
        self.temporal_graph = TemporalKnowledgeGraph()
        self.predictive_engine = None
        self.is_initialized = False
        
    def initialize(self) -> Dict:
        """Initialize the AI service with existing data"""
        try:
            logger.info("Initializing AI Service...")
            
            # Load existing knowledge items into temporal graph
            self._load_knowledge_items()
            
            # Initialize predictive analytics engine
            self.predictive_engine = PredictiveAnalyticsEngine(self.temporal_graph)
            model_results = self.predictive_engine.initialize_models()
            
            # Create initial temporal snapshot
            snapshot_id = self.temporal_graph.create_temporal_snapshot()
            
            self.is_initialized = True
            
            logger.info("AI Service initialized successfully")
            
            return {
                'success': True,
                'nodes_loaded': self.temporal_graph.graph.number_of_nodes(),
                'edges_loaded': self.temporal_graph.graph.number_of_edges(),
                'model_results': model_results,
                'snapshot_id': snapshot_id
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize AI Service: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
    def _load_knowledge_items(self):
        """Load existing knowledge items into the temporal graph"""
        try:
            # Get all knowledge items from database
            knowledge_items = KnowledgeItem.query.filter_by(is_archived=False).all()
            
            logger.info(f"Loading {len(knowledge_items)} knowledge items into temporal graph")
            
            for item in knowledge_items:
                # Create temporal node
                temporal_node = TemporalNode(
                    concept_id=item.id,
                    concept_name=item.title,
                    created_at=item.created_at,
                    last_accessed=item.accessed_at or item.created_at,
                    access_frequency=item.access_count,
                    importance_score=item.importance_score,
                    domain_tags=self._extract_domain_tags(item),
                    content_summary=item.content[:500] if item.content else "",
                    metadata={
                        'user_id': item.user_id,
                        'content_type': item.content_type,
                        'source_url': item.source_url,
                        'source_type': item.source_type
                    }
                )
                
                # Add to temporal graph
                self.temporal_graph.add_concept(temporal_node)
                
            # Build relationships between concepts
            self._build_concept_relationships()
            
        except Exception as e:
            logger.error(f"Error loading knowledge items: {str(e)}")
            raise
            
    def _extract_domain_tags(self, knowledge_item: KnowledgeItem) -> List[str]:
        """Extract domain tags from knowledge item"""
        domain_tags = []
        
        # Get explicit tags from database
        from src.models.knowledge_item import KnowledgeItemTag
        item_tags = db.session.query(Tag).join(KnowledgeItemTag).filter(
            KnowledgeItemTag.knowledge_item_id == knowledge_item.id
        ).all()
        
        domain_tags.extend([tag.name for tag in item_tags])
        
        # Extract implicit tags from content and title
        content_text = f"{knowledge_item.title} {knowledge_item.content or ''}".lower()
        
        # Domain keyword mapping
        domain_keywords = {
            'machine_learning': ['machine learning', 'ml', 'neural network', 'deep learning'],
            'data_science': ['data science', 'analytics', 'statistics', 'data analysis'],
            'web_development': ['web development', 'javascript', 'react', 'frontend', 'backend'],
            'artificial_intelligence': ['artificial intelligence', 'ai', 'nlp', 'computer vision'],
            'project_management': ['project management', 'agile', 'scrum', 'planning'],
            'design': ['design', 'ui', 'ux', 'user experience', 'interface'],
            'cybersecurity': ['security', 'cybersecurity', 'encryption', 'privacy'],
            'cloud_computing': ['cloud', 'aws', 'azure', 'deployment', 'infrastructure'],
            'mobile_development': ['mobile', 'ios', 'android', 'app development'],
            'devops': ['devops', 'ci/cd', 'docker', 'kubernetes', 'automation']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in content_text for keyword in keywords):
                domain_tags.append(domain)
                
        return list(set(domain_tags))  # Remove duplicates
        
    def _build_concept_relationships(self):
        """Build relationships between concepts based on various factors"""
        concepts = list(self.temporal_graph.graph.nodes.keys())
        
        logger.info(f"Building relationships between {len(concepts)} concepts")
        
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                # Calculate relationship strength
                relationship_info = self._calculate_concept_relationship(concept1, concept2)
                
                if relationship_info and relationship_info['strength'] > 0.3:
                    # Create temporal edge
                    edge = TemporalEdge(
                        source_id=concept1,
                        target_id=concept2,
                        relation_type=relationship_info['type'],
                        strength=relationship_info['strength'],
                        created_at=datetime.utcnow(),
                        last_reinforced=datetime.utcnow(),
                        reinforcement_count=1,
                        context=relationship_info['context'],
                        confidence=relationship_info['confidence']
                    )
                    
                    self.temporal_graph.add_relationship(edge)
                    
    def _calculate_concept_relationship(self, concept1_id: str, concept2_id: str) -> Optional[Dict]:
        """Calculate relationship between two concepts"""
        try:
            node1 = self.temporal_graph.graph.nodes[concept1_id]
            node2 = self.temporal_graph.graph.nodes[concept2_id]
            
            # Check domain similarity
            tags1 = set(node1['domain_tags'])
            tags2 = set(node2['domain_tags'])
            
            if not tags1 or not tags2:
                return None
                
            domain_overlap = len(tags1.intersection(tags2)) / len(tags1.union(tags2))
            
            if domain_overlap > 0.2:  # Significant domain overlap
                # Check temporal proximity
                time_diff = abs((node1['created_at'] - node2['created_at']).days)
                temporal_factor = max(0.0, 1.0 - (time_diff / 365))  # Closer in time = stronger relationship
                
                # Check user similarity
                user_factor = 1.0 if node1['metadata']['user_id'] == node2['metadata']['user_id'] else 0.5
                
                # Calculate overall strength
                strength = (domain_overlap * 0.6 + temporal_factor * 0.2 + user_factor * 0.2)
                
                if strength > 0.3:
                    return {
                        'type': RelationType.SEMANTIC_SIMILARITY,
                        'strength': min(1.0, strength),
                        'context': f"Domain overlap: {domain_overlap:.2f}",
                        'confidence': 0.7
                    }
                    
            return None
            
        except Exception as e:
            logger.error(f"Error calculating relationship between {concept1_id} and {concept2_id}: {str(e)}")
            return None
            
    def update_concept_access(self, knowledge_item_id: str, user_id: int) -> Dict:
        """Update concept access information in the temporal graph"""
        try:
            if not self.is_initialized:
                return {'success': False, 'error': 'AI Service not initialized'}
                
            # Update in temporal graph
            self.temporal_graph.update_concept_access(knowledge_item_id, datetime.utcnow())
            
            # Check for new relationships that might have emerged
            self._check_for_new_relationships(knowledge_item_id, user_id)
            
            return {'success': True}
            
        except Exception as e:
            logger.error(f"Error updating concept access: {str(e)}")
            return {'success': False, 'error': str(e)}
            
    def _check_for_new_relationships(self, concept_id: str, user_id: int):
        """Check if accessing this concept creates new relationships"""
        # Get recent user activities to find patterns
        recent_activities = UserActivity.query.filter(
            UserActivity.user_id == user_id,
            UserActivity.activity_type == 'knowledge_view',
            UserActivity.created_at >= datetime.utcnow() - timedelta(hours=24)
        ).all()
        
        # If user accessed multiple concepts recently, strengthen relationships
        recent_concept_ids = [
            activity.entity_id for activity in recent_activities 
            if activity.entity_id and activity.entity_id != concept_id
        ]
        
        for related_concept_id in recent_concept_ids:
            if related_concept_id in self.temporal_graph.graph.nodes:
                self.temporal_graph.reinforce_relationship(
                    concept_id, 
                    related_concept_id,
                    RelationType.TEMPORAL_SEQUENCE,
                    "Sequential access pattern"
                )
                
    def predict_idea_importance(self, concept_id: str, months_ahead: int = 6) -> Dict:
        """Predict future importance of an idea/concept"""
        try:
            if not self.is_initialized or not self.predictive_engine:
                return {'success': False, 'error': 'AI Service not initialized'}
                
            # Get prediction from predictive engine
            prediction = self.predictive_engine.predict_concept_importance(concept_id, months_ahead)
            
            # Store prediction in database
            db_prediction = Prediction(
                user_id=self.temporal_graph.graph.nodes[concept_id]['metadata']['user_id'],
                prediction_type='importance',
                target_entity_id=concept_id,
                target_entity_type='knowledge_item',
                predicted_value=prediction.predicted_value,
                confidence_score=prediction.confidence,
                prediction_date=prediction.prediction_date.date(),
                target_date=prediction.target_date.date(),
                prediction_metadata={
                    'reasoning': prediction.reasoning,
                    'contributing_factors': prediction.contributing_factors,
                    'months_ahead': months_ahead
                }
            )
            
            db.session.add(db_prediction)
            db.session.commit()
            
            return {
                'success': True,
                'prediction': {
                    'concept_id': concept_id,
                    'predicted_importance': prediction.predicted_value,
                    'confidence': prediction.confidence,
                    'reasoning': prediction.reasoning,
                    'contributing_factors': prediction.contributing_factors,
                    'prediction_date': prediction.prediction_date.isoformat(),
                    'target_date': prediction.target_date.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error predicting idea importance: {str(e)}")
            return {'success': False, 'error': str(e)}
            
    def discover_dormant_knowledge(self, user_id: int, current_context: List[str]) -> Dict:
        """Discover dormant knowledge relevant to current context"""
        try:
            if not self.is_initialized:
                return {'success': False, 'error': 'AI Service not initialized'}
                
            # Get dormant knowledge from temporal graph
            dormant_concepts = self.temporal_graph.discover_dormant_knowledge(current_context)
            
            # Store discoveries in database
            discoveries = []
            for concept_info in dormant_concepts:
                # Check if discovery already exists
                existing = DormantDiscovery.query.filter_by(
                    user_id=user_id,
                    knowledge_item_id=concept_info['concept_id'],
                    status='new'
                ).first()
                
                if not existing:
                    discovery = DormantDiscovery(
                        user_id=user_id,
                        knowledge_item_id=concept_info['concept_id'],
                        relevance_score=concept_info['relevance_score'],
                        context_description=concept_info['revival_reasoning']
                    )
                    
                    db.session.add(discovery)
                    discoveries.append(concept_info)
                    
            db.session.commit()
            
            return {
                'success': True,
                'discoveries': discoveries,
                'count': len(discoveries)
            }
            
        except Exception as e:
            logger.error(f"Error discovering dormant knowledge: {str(e)}")
            return {'success': False, 'error': str(e)}
            
    def forecast_knowledge_gaps(self, user_id: int, current_projects: List[str]) -> Dict:
        """Forecast knowledge gaps based on current projects"""
        try:
            if not self.is_initialized or not self.predictive_engine:
                return {'success': False, 'error': 'AI Service not initialized'}
                
            # Get gap predictions from predictive engine
            gap_predictions = self.predictive_engine.predict_knowledge_gaps(
                user_id, current_projects
            )
            
            # Store gaps in database
            gaps = []
            for gap_pred in gap_predictions:
                # Check if gap already exists
                existing = KnowledgeGap.query.filter_by(
                    user_id=user_id,
                    gap_topic=gap_pred.gap_topic,
                    status='identified'
                ).first()
                
                if not existing:
                    gap = KnowledgeGap(
                        user_id=user_id,
                        gap_topic=gap_pred.gap_topic,
                        gap_description=gap_pred.context,
                        urgency_score=gap_pred.urgency_score,
                        predicted_need_date=gap_pred.predicted_need_date.date(),
                        confidence_score=gap_pred.confidence
                    )
                    
                    db.session.add(gap)
                    gaps.append({
                        'topic': gap_pred.gap_topic,
                        'urgency_score': gap_pred.urgency_score,
                        'predicted_need_date': gap_pred.predicted_need_date.isoformat(),
                        'confidence': gap_pred.confidence,
                        'context': gap_pred.context,
                        'recommendations': gap_pred.recommendations
                    })
                    
            db.session.commit()
            
            return {
                'success': True,
                'gaps': gaps,
                'count': len(gaps)
            }
            
        except Exception as e:
            logger.error(f"Error forecasting knowledge gaps: {str(e)}")
            return {'success': False, 'error': str(e)}
            
    def get_learning_recommendations(self, user_id: int, focus_areas: List[str]) -> Dict:
        """Get personalized learning recommendations"""
        try:
            if not self.is_initialized or not self.predictive_engine:
                return {'success': False, 'error': 'AI Service not initialized'}
                
            recommendations = self.predictive_engine.get_learning_recommendations(
                user_id, focus_areas
            )
            
            return {
                'success': True,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error getting learning recommendations: {str(e)}")
            return {'success': False, 'error': str(e)}
            
    def analyze_knowledge_evolution(self, user_id: int, days_back: int = 90) -> Dict:
        """Analyze how user's knowledge has evolved over time"""
        try:
            if not self.is_initialized:
                return {'success': False, 'error': 'AI Service not initialized'}
                
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)
            
            evolution_analysis = self.temporal_graph.analyze_knowledge_evolution(
                start_date, end_date
            )
            
            return {
                'success': True,
                'evolution': evolution_analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing knowledge evolution: {str(e)}")
            return {'success': False, 'error': str(e)}
            
    def add_new_concept(self, knowledge_item: KnowledgeItem) -> Dict:
        """Add a new concept to the temporal graph"""
        try:
            if not self.is_initialized:
                return {'success': False, 'error': 'AI Service not initialized'}
                
            # Create temporal node
            temporal_node = TemporalNode(
                concept_id=knowledge_item.id,
                concept_name=knowledge_item.title,
                created_at=knowledge_item.created_at,
                last_accessed=knowledge_item.accessed_at or knowledge_item.created_at,
                access_frequency=knowledge_item.access_count,
                importance_score=knowledge_item.importance_score,
                domain_tags=self._extract_domain_tags(knowledge_item),
                content_summary=knowledge_item.content[:500] if knowledge_item.content else "",
                metadata={
                    'user_id': knowledge_item.user_id,
                    'content_type': knowledge_item.content_type,
                    'source_url': knowledge_item.source_url,
                    'source_type': knowledge_item.source_type
                }
            )
            
            # Add to temporal graph
            self.temporal_graph.add_concept(temporal_node)
            
            # Look for relationships with existing concepts
            self._find_relationships_for_new_concept(knowledge_item.id, knowledge_item.user_id)
            
            return {'success': True}
            
        except Exception as e:
            logger.error(f"Error adding new concept: {str(e)}")
            return {'success': False, 'error': str(e)}
            
    def _find_relationships_for_new_concept(self, concept_id: str, user_id: int):
        """Find relationships for a newly added concept"""
        # Get other concepts from the same user
        user_concepts = [
            node_id for node_id, data in self.temporal_graph.graph.nodes(data=True)
            if data['metadata']['user_id'] == user_id and node_id != concept_id
        ]
        
        # Calculate relationships with existing concepts
        for other_concept_id in user_concepts:
            relationship_info = self._calculate_concept_relationship(concept_id, other_concept_id)
            
            if relationship_info and relationship_info['strength'] > 0.3:
                edge = TemporalEdge(
                    source_id=concept_id,
                    target_id=other_concept_id,
                    relation_type=relationship_info['type'],
                    strength=relationship_info['strength'],
                    created_at=datetime.utcnow(),
                    last_reinforced=datetime.utcnow(),
                    reinforcement_count=1,
                    context=relationship_info['context'],
                    confidence=relationship_info['confidence']
                )
                
                self.temporal_graph.add_relationship(edge)
                
    def get_concept_insights(self, concept_id: str) -> Dict:
        """Get detailed insights about a specific concept"""
        try:
            if not self.is_initialized:
                return {'success': False, 'error': 'AI Service not initialized'}
                
            if concept_id not in self.temporal_graph.graph.nodes:
                return {'success': False, 'error': 'Concept not found'}
                
            # Get concept evolution
            evolution = self.temporal_graph.get_concept_evolution(concept_id)
            
            # Get related concepts
            related_concepts = []
            for neighbor in self.temporal_graph.graph.neighbors(concept_id):
                neighbor_data = self.temporal_graph.graph.nodes[neighbor]
                edge_data = self.temporal_graph.graph.edges[concept_id, neighbor]
                
                related_concepts.append({
                    'concept_id': neighbor,
                    'concept_name': neighbor_data['concept_name'],
                    'relationship_strength': edge_data.get('strength', 0.5),
                    'relationship_type': edge_data.get('relation_type', 'unknown')
                })
                
            # Sort by relationship strength
            related_concepts.sort(key=lambda x: x['relationship_strength'], reverse=True)
            
            return {
                'success': True,
                'insights': {
                    'evolution': evolution,
                    'related_concepts': related_concepts[:10],  # Top 10 related concepts
                    'graph_metrics': {
                        'centrality': len(related_concepts),
                        'importance_rank': self._calculate_importance_rank(concept_id)
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting concept insights: {str(e)}")
            return {'success': False, 'error': str(e)}
            
    def _calculate_importance_rank(self, concept_id: str) -> int:
        """Calculate the importance rank of a concept among all concepts"""
        all_concepts = [
            (node_id, data['importance_score'])
            for node_id, data in self.temporal_graph.graph.nodes(data=True)
        ]
        
        # Sort by importance score
        all_concepts.sort(key=lambda x: x[1], reverse=True)
        
        # Find rank
        for rank, (node_id, _) in enumerate(all_concepts, 1):
            if node_id == concept_id:
                return rank
                
        return len(all_concepts)  # Default to last rank
        
    def create_knowledge_snapshot(self) -> Dict:
        """Create a snapshot of current knowledge state"""
        try:
            if not self.is_initialized:
                return {'success': False, 'error': 'AI Service not initialized'}
                
            snapshot_id = self.temporal_graph.create_temporal_snapshot()
            
            return {
                'success': True,
                'snapshot_id': snapshot_id,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating knowledge snapshot: {str(e)}")
            return {'success': False, 'error': str(e)}

# Global AI service instance
ai_service = AIService()

