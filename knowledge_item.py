from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import uuid
from user import db

class KnowledgeItem(db.Model):
    __tablename__ = 'knowledge_items'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(500), nullable=False)
    content = db.Column(db.Text)
    content_type = db.Column(db.String(50), default='text')
    source_url = db.Column(db.Text)
    source_type = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    accessed_at = db.Column(db.DateTime, default=datetime.utcnow)
    access_count = db.Column(db.Integer, default=0)
    importance_score = db.Column(db.Float, default=0.0)
    item_metadata = db.Column('metadata', db.JSON, default=dict)
    is_archived = db.Column(db.Boolean, default=False)
    
    # Relationship with user
    user = db.relationship('User', backref=db.backref('knowledge_items', lazy=True))
    
    def __repr__(self):
        return f'<KnowledgeItem {self.title}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'content': self.content,
            'content_type': self.content_type,
            'source_url': self.source_url,
            'source_type': self.source_type,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'accessed_at': self.accessed_at.isoformat() if self.accessed_at else None,
            'access_count': self.access_count,
            'importance_score': self.importance_score,
            'metadata': self.item_metadata,
            'is_archived': self.is_archived
        }
    
    def update_access(self):
        """Update access timestamp and count"""
        self.accessed_at = datetime.utcnow()
        self.access_count += 1
        db.session.commit()


class Tag(db.Model):
    __tablename__ = 'tags'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    color = db.Column(db.String(7), default='#6366f1')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship with user
    user = db.relationship('User', backref=db.backref('tags', lazy=True))
    
    # Unique constraint for user_id and name combination
    __table_args__ = (db.UniqueConstraint('user_id', 'name', name='unique_user_tag'),)
    
    def __repr__(self):
        return f'<Tag {self.name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'name': self.name,
            'color': self.color,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class KnowledgeItemTag(db.Model):
    __tablename__ = 'knowledge_item_tags'
    
    knowledge_item_id = db.Column(db.String(36), db.ForeignKey('knowledge_items.id'), primary_key=True)
    tag_id = db.Column(db.String(36), db.ForeignKey('tags.id'), primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    knowledge_item = db.relationship('KnowledgeItem', backref=db.backref('item_tags', lazy=True))
    tag = db.relationship('Tag', backref=db.backref('tag_items', lazy=True))


class UserActivity(db.Model):
    __tablename__ = 'user_activities'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    activity_type = db.Column(db.String(50), nullable=False)
    entity_type = db.Column(db.String(50))
    entity_id = db.Column(db.String(36))
    duration_seconds = db.Column(db.Integer)
    activity_metadata = db.Column('metadata', db.JSON, default=dict)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship with user
    user = db.relationship('User', backref=db.backref('activities', lazy=True))
    
    def __repr__(self):
        return f'<UserActivity {self.activity_type}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'activity_type': self.activity_type,
            'entity_type': self.entity_type,
            'entity_id': self.entity_id,
            'duration_seconds': self.duration_seconds,
            'metadata': self.activity_metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

