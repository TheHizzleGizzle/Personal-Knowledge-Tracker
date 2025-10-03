from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta, date
from sqlalchemy import desc, func
from user import db
from knowledge_item import KnowledgeItem
from analytics import Prediction, DormantDiscovery, KnowledgeGap
import random

insights_bp = Blueprint('insights', __name__)

@insights_bp.route('/insights/predictions', methods=['GET'])
def get_predictions():
    """Get AI-powered predictions for idea importance"""
    try:
        user_id = request.args.get('user_id', 1, type=int)
        prediction_type = request.args.get('type', 'importance')
        limit = request.args.get('limit', 10, type=int)
        
        # Get existing predictions
        predictions = Prediction.query.filter_by(
            user_id=user_id,
            prediction_type=prediction_type,
            status='active'
        ).order_by(desc(Prediction.confidence_score)).limit(limit).all()
        
        # If no predictions exist, generate some demo predictions
        if not predictions:
            predictions = generate_demo_predictions(user_id, prediction_type, limit)
        
        # Enhance predictions with knowledge item details
        enhanced_predictions = []
        for pred in predictions:
            pred_dict = pred.to_dict()
            
            # Get associated knowledge item if exists
            if pred.target_entity_type == 'knowledge_item' and pred.target_entity_id:
                item = KnowledgeItem.query.get(pred.target_entity_id)
                if item:
                    pred_dict['knowledge_item'] = {
                        'id': item.id,
                        'title': item.title,
                        'content_type': item.content_type,
                        'current_importance': item.importance_score
                    }
            
            # Add reasoning based on prediction type
            if prediction_type == 'importance':
                pred_dict['reasoning'] = generate_importance_reasoning(pred)
            
            enhanced_predictions.append(pred_dict)
        
        return jsonify({
            'success': True,
            'data': enhanced_predictions,
            'meta': {
                'prediction_type': prediction_type,
                'count': len(enhanced_predictions)
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@insights_bp.route('/insights/gaps', methods=['GET'])
def get_knowledge_gaps():
    """Get knowledge gap forecasts"""
    try:
        user_id = request.args.get('user_id', 1, type=int)
        status = request.args.get('status', 'identified')
        
        # Get existing knowledge gaps
        gaps = KnowledgeGap.query.filter_by(
            user_id=user_id,
            status=status
        ).order_by(desc(KnowledgeGap.urgency_score)).all()
        
        # If no gaps exist, generate some demo gaps
        if not gaps:
            gaps = generate_demo_knowledge_gaps(user_id)
        
        # Enhance gaps with recommendations
        enhanced_gaps = []
        for gap in gaps:
            gap_dict = gap.to_dict()
            gap_dict['recommendations'] = generate_gap_recommendations(gap)
            enhanced_gaps.append(gap_dict)
        
        return jsonify({
            'success': True,
            'data': enhanced_gaps
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@insights_bp.route('/insights/dormant', methods=['GET'])
def get_dormant_knowledge():
    """Get dormant knowledge discoveries"""
    try:
        user_id = request.args.get('user_id', 1, type=int)
        status = request.args.get('status', 'new')
        limit = request.args.get('limit', 10, type=int)
        
        # Get existing dormant discoveries
        discoveries = DormantDiscovery.query.filter_by(
            user_id=user_id,
            status=status
        ).order_by(desc(DormantDiscovery.relevance_score)).limit(limit).all()
        
        # If no discoveries exist, generate some demo discoveries
        if not discoveries:
            discoveries = generate_demo_dormant_discoveries(user_id, limit)
        
        # Enhance discoveries with knowledge item details
        enhanced_discoveries = []
        for discovery in discoveries:
            discovery_dict = discovery.to_dict()
            
            # Get associated knowledge item
            if discovery.knowledge_item_id:
                item = KnowledgeItem.query.get(discovery.knowledge_item_id)
                if item:
                    discovery_dict['knowledge_item'] = {
                        'id': item.id,
                        'title': item.title,
                        'content': item.content[:200] + '...' if len(item.content) > 200 else item.content,
                        'created_at': item.created_at.isoformat() if item.created_at else None,
                        'source_url': item.source_url
                    }
            
            enhanced_discoveries.append(discovery_dict)
        
        return jsonify({
            'success': True,
            'data': enhanced_discoveries
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@insights_bp.route('/insights/recommendations', methods=['GET'])
def get_recommendations():
    """Get personalized recommendations"""
    try:
        user_id = request.args.get('user_id', 1, type=int)
        rec_type = request.args.get('type', 'content')  # content, learning, connections
        
        if rec_type == 'content':
            recommendations = generate_content_recommendations(user_id)
        elif rec_type == 'learning':
            recommendations = generate_learning_recommendations(user_id)
        elif rec_type == 'connections':
            recommendations = generate_connection_recommendations(user_id)
        else:
            return jsonify({'success': False, 'error': 'Invalid recommendation type'}), 400
        
        return jsonify({
            'success': True,
            'data': recommendations,
            'meta': {
                'type': rec_type,
                'count': len(recommendations)
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@insights_bp.route('/insights/feedback', methods=['POST'])
def provide_feedback():
    """Provide feedback on AI insights"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 1)
        
        insight_type = data['insight_type']  # prediction, gap, dormant, recommendation
        insight_id = data['insight_id']
        feedback_type = data['feedback_type']  # helpful, not_helpful, applied, dismissed
        feedback_text = data.get('feedback_text', '')
        
        # Update the insight based on feedback
        if insight_type == 'prediction':
            insight = Prediction.query.get(insight_id)
            if insight and feedback_type == 'applied':
                insight.status = 'applied'
        elif insight_type == 'dormant':
            insight = DormantDiscovery.query.get(insight_id)
            if insight:
                if feedback_type == 'applied':
                    insight.applied_at = datetime.utcnow()
                    insight.status = 'applied'
                elif feedback_type == 'viewed':
                    insight.viewed_at = datetime.utcnow()
                    insight.status = 'viewed'
        elif insight_type == 'gap':
            insight = KnowledgeGap.query.get(insight_id)
            if insight and feedback_type == 'addressed':
                insight.addressed_at = datetime.utcnow()
                insight.status = 'addressed'
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Feedback recorded successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


# Helper functions for generating demo data

def generate_demo_predictions(user_id, prediction_type, limit):
    """Generate demo predictions for testing"""
    demo_predictions = []
    
    # Get some knowledge items to predict on
    items = KnowledgeItem.query.filter_by(user_id=user_id, is_archived=False).limit(limit).all()
    
    prediction_topics = [
        "Quantum Computing Applications",
        "Sustainable Design Principles", 
        "AI Ethics Framework",
        "Remote Work Methodologies",
        "Data Privacy Regulations",
        "Blockchain Technology",
        "Mental Health in Tech",
        "Climate Change Solutions"
    ]
    
    for i in range(min(limit, len(prediction_topics))):
        target_item = items[i] if i < len(items) else None
        
        prediction = Prediction(
            user_id=user_id,
            prediction_type=prediction_type,
            target_entity_id=target_item.id if target_item else None,
            target_entity_type='knowledge_item' if target_item else 'topic',
            predicted_value=round(random.uniform(0.4, 0.95), 2),
            confidence_score=round(random.uniform(0.6, 0.9), 2),
            prediction_date=date.today(),
            target_date=date.today() + timedelta(days=random.randint(30, 180)),
            prediction_metadata={'topic': prediction_topics[i] if not target_item else target_item.title}
        )
        
        db.session.add(prediction)
        demo_predictions.append(prediction)
    
    db.session.commit()
    return demo_predictions


def generate_demo_knowledge_gaps(user_id):
    """Generate demo knowledge gaps"""
    demo_gaps = []
    
    gap_topics = [
        {
            'topic': 'Data Visualization',
            'description': 'Advanced statistical visualization techniques for complex datasets',
            'urgency': 0.8,
            'days_ahead': 14
        },
        {
            'topic': 'Agile Methodologies',
            'description': 'Modern agile frameworks and best practices for team management',
            'urgency': 0.6,
            'days_ahead': 30
        },
        {
            'topic': 'Cloud Architecture',
            'description': 'Scalable cloud infrastructure design and implementation',
            'urgency': 0.7,
            'days_ahead': 45
        }
    ]
    
    for gap_data in gap_topics:
        gap = KnowledgeGap(
            user_id=user_id,
            gap_topic=gap_data['topic'],
            gap_description=gap_data['description'],
            urgency_score=gap_data['urgency'],
            predicted_need_date=date.today() + timedelta(days=gap_data['days_ahead']),
            confidence_score=round(random.uniform(0.65, 0.85), 2)
        )
        
        db.session.add(gap)
        demo_gaps.append(gap)
    
    db.session.commit()
    return demo_gaps


def generate_demo_dormant_discoveries(user_id, limit):
    """Generate demo dormant knowledge discoveries"""
    demo_discoveries = []
    
    # Get some older knowledge items
    items = KnowledgeItem.query.filter_by(user_id=user_id, is_archived=False).order_by(
        KnowledgeItem.created_at
    ).limit(limit).all()
    
    contexts = [
        "Your notes on network theory are highly applicable to your current social media analytics project",
        "Concepts from your economics research could enhance your current user behavior analysis",
        "Your previous work on design patterns is relevant to your current software architecture decisions",
        "Historical data analysis techniques you learned could improve your current forecasting models"
    ]
    
    for i, item in enumerate(items[:limit]):
        discovery = DormantDiscovery(
            user_id=user_id,
            knowledge_item_id=item.id,
            relevance_score=round(random.uniform(0.7, 0.95), 2),
            context_description=contexts[i % len(contexts)]
        )
        
        db.session.add(discovery)
        demo_discoveries.append(discovery)
    
    db.session.commit()
    return demo_discoveries


def generate_importance_reasoning(prediction):
    """Generate reasoning for importance predictions"""
    confidence = prediction.confidence_score
    
    if confidence > 0.8:
        return f"Based on industry trends and your research patterns, this topic shows strong indicators of growing relevance. Cross-referencing with current projects suggests high future importance."
    elif confidence > 0.6:
        return f"Moderate confidence based on emerging patterns in your knowledge consumption and external trend analysis."
    else:
        return f"Initial indicators suggest potential relevance, but requires more data for higher confidence."


def generate_gap_recommendations(gap):
    """Generate recommendations for addressing knowledge gaps"""
    return [
        f"Explore foundational concepts in {gap.gap_topic}",
        f"Review recent developments and best practices",
        f"Allocate 2-3 hours weekly for focused learning",
        f"Consider practical projects to apply new knowledge"
    ]


def generate_content_recommendations(user_id):
    """Generate content recommendations"""
    return [
        {
            'type': 'article',
            'title': 'Advanced Machine Learning Techniques',
            'source': 'MIT Technology Review',
            'relevance_score': 0.89,
            'reason': 'Matches your interest in AI and current project requirements'
        },
        {
            'type': 'course',
            'title': 'Data Visualization Masterclass',
            'source': 'Coursera',
            'relevance_score': 0.76,
            'reason': 'Addresses identified knowledge gap in visualization'
        }
    ]


def generate_learning_recommendations(user_id):
    """Generate learning path recommendations"""
    return [
        {
            'path': 'AI Ethics and Governance',
            'duration': '4 weeks',
            'difficulty': 'intermediate',
            'relevance_score': 0.85,
            'modules': ['Ethical Frameworks', 'Bias Detection', 'Governance Models']
        },
        {
            'path': 'Systems Thinking Fundamentals',
            'duration': '3 weeks', 
            'difficulty': 'beginner',
            'relevance_score': 0.72,
            'modules': ['Complex Systems', 'Feedback Loops', 'Emergence']
        }
    ]


def generate_connection_recommendations(user_id):
    """Generate knowledge connection recommendations"""
    return [
        {
            'connection_type': 'concept_bridge',
            'from_topic': 'Machine Learning',
            'to_topic': 'Cognitive Psychology',
            'strength': 0.78,
            'insight': 'Understanding cognitive biases can improve ML model design'
        },
        {
            'connection_type': 'application',
            'from_topic': 'Network Theory',
            'to_topic': 'Social Media Analysis',
            'strength': 0.85,
            'insight': 'Graph algorithms can reveal hidden patterns in social networks'
        }
    ]

