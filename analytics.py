from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, date
import uuid
from user import db

class LearningMetric(db.Model):
    __tablename__ = 'learning_metrics'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    metric_type = db.Column(db.String(50), nullable=False)
    metric_value = db.Column(db.Float, nullable=False)
    time_period = db.Column(db.String(20), nullable=False)  # 'daily', 'weekly', 'monthly'
    date = db.Column(db.Date, nullable=False)
    metric_metadata = db.Column('metadata', db.JSON, default=dict)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship with user
    user = db.relationship('User', backref=db.backref('learning_metrics', lazy=True))
    
    def __repr__(self):
        return f'<LearningMetric {self.metric_type}: {self.metric_value}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'metric_type': self.metric_type,
            'metric_value': self.metric_value,
            'time_period': self.time_period,
            'date': self.date.isoformat() if self.date else None,
            'metadata': self.metric_metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Prediction(db.Model):
    __tablename__ = 'predictions'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prediction_type = db.Column(db.String(50), nullable=False)
    target_entity_id = db.Column(db.String(36))
    target_entity_type = db.Column(db.String(50))
    predicted_value = db.Column(db.Float)
    confidence_score = db.Column(db.Float)
    prediction_date = db.Column(db.Date, nullable=False)
    target_date = db.Column(db.Date)
    status = db.Column(db.String(20), default='active')
    prediction_metadata = db.Column('metadata', db.JSON, default=dict)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship with user
    user = db.relationship('User', backref=db.backref('predictions', lazy=True))
    
    def __repr__(self):
        return f'<Prediction {self.prediction_type}: {self.predicted_value}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'prediction_type': self.prediction_type,
            'target_entity_id': self.target_entity_id,
            'target_entity_type': self.target_entity_type,
            'predicted_value': self.predicted_value,
            'confidence_score': self.confidence_score,
            'prediction_date': self.prediction_date.isoformat() if self.prediction_date else None,
            'target_date': self.target_date.isoformat() if self.target_date else None,
            'status': self.status,
            'metadata': self.prediction_metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class DormantDiscovery(db.Model):
    __tablename__ = 'dormant_discoveries'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    knowledge_item_id = db.Column(db.String(36), db.ForeignKey('knowledge_items.id'), nullable=False)
    relevance_score = db.Column(db.Float, nullable=False)
    context_description = db.Column(db.Text)
    discovered_at = db.Column(db.DateTime, default=datetime.utcnow)
    viewed_at = db.Column(db.DateTime)
    applied_at = db.Column(db.DateTime)
    status = db.Column(db.String(20), default='new')
    
    # Relationships
    user = db.relationship('User', backref=db.backref('dormant_discoveries', lazy=True))
    knowledge_item = db.relationship('KnowledgeItem', backref=db.backref('dormant_discoveries', lazy=True))
    
    def __repr__(self):
        return f'<DormantDiscovery {self.relevance_score}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'knowledge_item_id': self.knowledge_item_id,
            'relevance_score': self.relevance_score,
            'context_description': self.context_description,
            'discovered_at': self.discovered_at.isoformat() if self.discovered_at else None,
            'viewed_at': self.viewed_at.isoformat() if self.viewed_at else None,
            'applied_at': self.applied_at.isoformat() if self.applied_at else None,
            'status': self.status
        }


class KnowledgeGap(db.Model):
    __tablename__ = 'knowledge_gaps'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    gap_topic = db.Column(db.String(200), nullable=False)
    gap_description = db.Column(db.Text)
    urgency_score = db.Column(db.Float, default=0.0)
    predicted_need_date = db.Column(db.Date)
    confidence_score = db.Column(db.Float)
    status = db.Column(db.String(20), default='identified')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    addressed_at = db.Column(db.DateTime)
    
    # Relationship with user
    user = db.relationship('User', backref=db.backref('knowledge_gaps', lazy=True))
    
    def __repr__(self):
        return f'<KnowledgeGap {self.gap_topic}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'gap_topic': self.gap_topic,
            'gap_description': self.gap_description,
            'urgency_score': self.urgency_score,
            'predicted_need_date': self.predicted_need_date.isoformat() if self.predicted_need_date else None,
            'confidence_score': self.confidence_score,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'addressed_at': self.addressed_at.isoformat() if self.addressed_at else None
        }


class ReadingSession(db.Model):
    __tablename__ = 'reading_sessions'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    knowledge_item_id = db.Column(db.String(36), db.ForeignKey('knowledge_items.id'), nullable=False)
    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime)
    duration_seconds = db.Column(db.Integer)
    completion_percentage = db.Column(db.Float, default=0.0)
    reading_speed_wpm = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', backref=db.backref('reading_sessions', lazy=True))
    knowledge_item = db.relationship('KnowledgeItem', backref=db.backref('reading_sessions', lazy=True))
    
    def __repr__(self):
        return f'<ReadingSession {self.duration_seconds}s>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'knowledge_item_id': self.knowledge_item_id,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'completion_percentage': self.completion_percentage,
            'reading_speed_wpm': self.reading_speed_wpm,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

