"""
Enhanced AI Insights Routes
===========================

This module provides Flask routes for advanced AI-powered insights including
temporal knowledge modeling, predictive analytics, and personalized recommendations.
"""

from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
from ai_service import ai_service
from user import db
from knowledge_item import KnowledgeItem, UserActivity
import logging

logger = logging.getLogger(__name__)

ai_insights_bp = Blueprint('ai_insights', __name__)

@ai_insights_bp.route('/ai/initialize', methods=['POST'])
def initialize_ai_service():
    """Initialize the AI service with current data"""
    try:
        result = ai_service.initialize()
        
        if result['success']:
            return jsonify({
                'success': True,
                'message': 'AI service initialized successfully',
                'data': result
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error')
            }), 500
            
    except Exception as e:
        logger.error(f"Error initializing AI service: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@ai_insights_bp.route('/ai/predict-importance', methods=['POST'])
def predict_idea_importance():
    """Predict future importance of an idea/concept"""
    try:
        data = request.get_json()
        concept_id = data.get('concept_id')
        months_ahead = data.get('months_ahead', 6)
        
        if not concept_id:
            return jsonify({'success': False, 'error': 'concept_id is required'}), 400
            
        # Ensure AI service is initialized
        if not ai_service.is_initialized:
            ai_service.initialize()
            
        result = ai_service.predict_idea_importance(concept_id, months_ahead)
        
        if result['success']:
            return jsonify({
                'success': True,
                'data': result['prediction']
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Prediction failed')
            }), 500
            
    except Exception as e:
        logger.error(f"Error predicting idea importance: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@ai_insights_bp.route('/ai/dormant-knowledge', methods=['POST'])
def discover_dormant_knowledge():
    """Discover dormant knowledge relevant to current context"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 1)
        current_context = data.get('context', [])
        
        if not current_context:
            return jsonify({'success': False, 'error': 'context is required'}), 400
            
        # Ensure AI service is initialized
        if not ai_service.is_initialized:
            ai_service.initialize()
            
        result = ai_service.discover_dormant_knowledge(user_id, current_context)
        
        if result['success']:
            return jsonify({
                'success': True,
                'data': result['discoveries'],
                'meta': {
                    'count': result['count'],
                    'context': current_context
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Discovery failed')
            }), 500
            
    except Exception as e:
        logger.error(f"Error discovering dormant knowledge: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@ai_insights_bp.route('/ai/knowledge-gaps', methods=['POST'])
def forecast_knowledge_gaps():
    """Forecast knowledge gaps based on current projects"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 1)
        current_projects = data.get('projects', [])
        
        if not current_projects:
            return jsonify({'success': False, 'error': 'projects list is required'}), 400
            
        # Ensure AI service is initialized
        if not ai_service.is_initialized:
            ai_service.initialize()
            
        result = ai_service.forecast_knowledge_gaps(user_id, current_projects)
        
        if result['success']:
            return jsonify({
                'success': True,
                'data': result['gaps'],
                'meta': {
                    'count': result['count'],
                    'projects_analyzed': len(current_projects)
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Gap forecasting failed')
            }), 500
            
    except Exception as e:
        logger.error(f"Error forecasting knowledge gaps: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@ai_insights_bp.route('/ai/learning-recommendations', methods=['POST'])
def get_learning_recommendations():
    """Get personalized learning recommendations"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 1)
        focus_areas = data.get('focus_areas', [])
        
        # Ensure AI service is initialized
        if not ai_service.is_initialized:
            ai_service.initialize()
            
        result = ai_service.get_learning_recommendations(user_id, focus_areas)
        
        if result['success']:
            return jsonify({
                'success': True,
                'data': result['recommendations'],
                'meta': {
                    'focus_areas': focus_areas,
                    'recommendation_count': len(result['recommendations'])
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Recommendation generation failed')
            }), 500
            
    except Exception as e:
        logger.error(f"Error getting learning recommendations: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@ai_insights_bp.route('/ai/knowledge-evolution', methods=['GET'])
def analyze_knowledge_evolution():
    """Analyze knowledge evolution over time"""
    try:
        user_id = request.args.get('user_id', 1, type=int)
        days_back = request.args.get('days_back', 90, type=int)
        
        # Ensure AI service is initialized
        if not ai_service.is_initialized:
            ai_service.initialize()
            
        result = ai_service.analyze_knowledge_evolution(user_id, days_back)
        
        if result['success']:
            return jsonify({
                'success': True,
                'data': result['evolution'],
                'meta': {
                    'user_id': user_id,
                    'analysis_period_days': days_back
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Evolution analysis failed')
            }), 500
            
    except Exception as e:
        logger.error(f"Error analyzing knowledge evolution: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@ai_insights_bp.route('/ai/concept-insights/<concept_id>', methods=['GET'])
def get_concept_insights(concept_id):
    """Get detailed insights about a specific concept"""
    try:
        # Ensure AI service is initialized
        if not ai_service.is_initialized:
            ai_service.initialize()
            
        result = ai_service.get_concept_insights(concept_id)
        
        if result['success']:
            return jsonify({
                'success': True,
                'data': result['insights'],
                'meta': {
                    'concept_id': concept_id
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Concept insights failed')
            }), 500
            
    except Exception as e:
        logger.error(f"Error getting concept insights: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@ai_insights_bp.route('/ai/update-access', methods=['POST'])
def update_concept_access():
    """Update concept access information for AI tracking"""
    try:
        data = request.get_json()
        concept_id = data.get('concept_id')
        user_id = data.get('user_id', 1)
        
        if not concept_id:
            return jsonify({'success': False, 'error': 'concept_id is required'}), 400
            
        # Ensure AI service is initialized
        if not ai_service.is_initialized:
            ai_service.initialize()
            
        result = ai_service.update_concept_access(concept_id, user_id)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error updating concept access: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@ai_insights_bp.route('/ai/temporal-patterns', methods=['GET'])
def get_temporal_patterns():
    """Get temporal patterns in knowledge consumption"""
    try:
        user_id = request.args.get('user_id', 1, type=int)
        days_back = request.args.get('days_back', 30, type=int)
        
        # Get user activities for pattern analysis
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        
        activities = UserActivity.query.filter(
            UserActivity.user_id == user_id,
            UserActivity.created_at >= start_date,
            UserActivity.activity_type.in_(['knowledge_view', 'knowledge_create'])
        ).order_by(UserActivity.created_at).all()
        
        # Analyze patterns
        patterns = analyze_temporal_patterns(activities)
        
        return jsonify({
            'success': True,
            'data': patterns,
            'meta': {
                'user_id': user_id,
                'analysis_period_days': days_back,
                'total_activities': len(activities)
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting temporal patterns: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@ai_insights_bp.route('/ai/knowledge-network', methods=['GET'])
def get_knowledge_network():
    """Get knowledge network visualization data"""
    try:
        user_id = request.args.get('user_id', 1, type=int)
        
        # Ensure AI service is initialized
        if not ai_service.is_initialized:
            ai_service.initialize()
            
        # Get network data from temporal graph
        network_data = extract_network_data(ai_service.temporal_graph, user_id)
        
        return jsonify({
            'success': True,
            'data': network_data,
            'meta': {
                'user_id': user_id,
                'node_count': len(network_data['nodes']),
                'edge_count': len(network_data['edges'])
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting knowledge network: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@ai_insights_bp.route('/ai/predictive-dashboard', methods=['GET'])
def get_predictive_dashboard():
    """Get comprehensive predictive dashboard data"""
    try:
        user_id = request.args.get('user_id', 1, type=int)
        
        # Ensure AI service is initialized
        if not ai_service.is_initialized:
            ai_service.initialize()
            
        # Get recent knowledge items for prediction
        recent_items = KnowledgeItem.query.filter_by(
            user_id=user_id, is_archived=False
        ).order_by(KnowledgeItem.importance_score.desc()).limit(5).all()
        
        # Generate predictions for top concepts
        importance_predictions = []
        for item in recent_items:
            pred_result = ai_service.predict_idea_importance(item.id, 6)
            if pred_result['success']:
                importance_predictions.append(pred_result['prediction'])
                
        # Get dormant knowledge with general context
        dormant_result = ai_service.discover_dormant_knowledge(
            user_id, ['current_work', 'projects', 'research']
        )
        dormant_knowledge = dormant_result['discoveries'] if dormant_result['success'] else []
        
        # Get knowledge gaps with sample projects
        gaps_result = ai_service.forecast_knowledge_gaps(
            user_id, ['machine learning project', 'web development', 'data analysis']
        )
        knowledge_gaps = gaps_result['gaps'] if gaps_result['success'] else []
        
        # Get learning recommendations
        learning_result = ai_service.get_learning_recommendations(
            user_id, ['machine_learning', 'web_development', 'data_science']
        )
        learning_recommendations = learning_result['recommendations'] if learning_result['success'] else []
        
        return jsonify({
            'success': True,
            'data': {
                'importance_predictions': importance_predictions,
                'dormant_knowledge': dormant_knowledge[:5],  # Top 5
                'knowledge_gaps': knowledge_gaps[:5],  # Top 5
                'learning_recommendations': learning_recommendations[:3]  # Top 3
            },
            'meta': {
                'user_id': user_id,
                'generated_at': datetime.utcnow().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting predictive dashboard: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@ai_insights_bp.route('/ai/create-snapshot', methods=['POST'])
def create_knowledge_snapshot():
    """Create a snapshot of current knowledge state"""
    try:
        # Ensure AI service is initialized
        if not ai_service.is_initialized:
            ai_service.initialize()
            
        result = ai_service.create_knowledge_snapshot()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error creating knowledge snapshot: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# Helper functions

def analyze_temporal_patterns(activities):
    """Analyze temporal patterns in user activities"""
    if not activities:
        return {
            'hourly_distribution': {},
            'daily_distribution': {},
            'activity_streaks': [],
            'peak_hours': [],
            'learning_velocity': 0.0
        }
        
    # Hourly distribution
    hourly_counts = {}
    daily_counts = {}
    
    for activity in activities:
        hour = activity.created_at.hour
        day = activity.created_at.strftime('%A')
        
        hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
        daily_counts[day] = daily_counts.get(day, 0) + 1
        
    # Find peak hours (top 3)
    peak_hours = sorted(hourly_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    peak_hours = [f"{hour}:00" for hour, _ in peak_hours]
    
    # Calculate learning velocity (activities per day)
    total_days = (activities[-1].created_at - activities[0].created_at).days + 1
    learning_velocity = len(activities) / max(1, total_days)
    
    # Identify activity streaks (consecutive days with activity)
    activity_dates = set(activity.created_at.date() for activity in activities)
    streaks = find_consecutive_streaks(sorted(activity_dates))
    
    return {
        'hourly_distribution': hourly_counts,
        'daily_distribution': daily_counts,
        'activity_streaks': streaks,
        'peak_hours': peak_hours,
        'learning_velocity': round(learning_velocity, 2)
    }


def find_consecutive_streaks(dates):
    """Find consecutive date streaks"""
    if not dates:
        return []
        
    streaks = []
    current_streak = [dates[0]]
    
    for i in range(1, len(dates)):
        if (dates[i] - dates[i-1]).days == 1:
            current_streak.append(dates[i])
        else:
            if len(current_streak) > 1:
                streaks.append({
                    'start_date': current_streak[0].isoformat(),
                    'end_date': current_streak[-1].isoformat(),
                    'length_days': len(current_streak)
                })
            current_streak = [dates[i]]
            
    # Don't forget the last streak
    if len(current_streak) > 1:
        streaks.append({
            'start_date': current_streak[0].isoformat(),
            'end_date': current_streak[-1].isoformat(),
            'length_days': len(current_streak)
        })
        
    return streaks


def extract_network_data(temporal_graph, user_id):
    """Extract network visualization data from temporal graph"""
    nodes = []
    edges = []
    
    # Get user's concepts
    user_concepts = [
        (node_id, data) for node_id, data in temporal_graph.graph.nodes(data=True)
        if data['metadata']['user_id'] == user_id
    ]
    
    # Create nodes
    for concept_id, data in user_concepts:
        nodes.append({
            'id': concept_id,
            'label': data['concept_name'],
            'size': data['importance_score'] * 100,  # Scale for visualization
            'color': get_domain_color(data['domain_tags']),
            'importance': data['importance_score'],
            'access_count': data['access_frequency'],
            'domains': data['domain_tags']
        })
        
    # Create edges
    concept_ids = [concept_id for concept_id, _ in user_concepts]
    for concept_id in concept_ids:
        for neighbor in temporal_graph.graph.neighbors(concept_id):
            if neighbor in concept_ids:  # Only include edges between user's concepts
                edge_data = temporal_graph.graph.edges[concept_id, neighbor]
                edges.append({
                    'source': concept_id,
                    'target': neighbor,
                    'weight': edge_data.get('strength', 0.5),
                    'type': edge_data.get('relation_type', 'unknown'),
                    'context': edge_data.get('context', '')
                })
                
    return {
        'nodes': nodes,
        'edges': edges
    }


def get_domain_color(domain_tags):
    """Get color for visualization based on domain tags"""
    domain_colors = {
        'machine_learning': '#FF6B6B',
        'data_science': '#4ECDC4',
        'web_development': '#45B7D1',
        'artificial_intelligence': '#96CEB4',
        'design': '#FFEAA7',
        'project_management': '#DDA0DD',
        'cybersecurity': '#FF7675',
        'cloud_computing': '#74B9FF'
    }
    
    if not domain_tags:
        return '#95A5A6'  # Default gray
        
    # Return color of first matching domain
    for tag in domain_tags:
        if tag in domain_colors:
            return domain_colors[tag]
            
    return '#95A5A6'  # Default gray

