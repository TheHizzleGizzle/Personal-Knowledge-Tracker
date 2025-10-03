from flask import Blueprint, request, jsonify
from datetime import datetime
import re

from user import db
from knowledge_item import KnowledgeItem, Tag, KnowledgeItemTag, UserActivity

knowledge_bp = Blueprint('knowledge', __name__)

# Supported content types for knowledge items
ALLOWED_CONTENT_TYPES = {
    'text', 'link', 'markdown', 'pdf', 'video', 'audio', 'image', 'code'
}


def normalize_and_validate_tags(raw_tags):
    """Validate tags array and return a cleaned list.
    - Must be a list of strings
    - Max 20 tags
    - Each tag non-empty after trim, <=100 chars
    - Allowed chars: letters, numbers, spaces, underscores, hyphens
    - Case-insensitive uniqueness
    """
    if raw_tags is None:
        return []
    if not isinstance(raw_tags, list):
        raise ValueError('Tags must be an array of strings')

    cleaned = []
    seen = set()
    for t in raw_tags:
        if not isinstance(t, str):
            raise ValueError('Tags must be strings')
        name = t.strip()
        if not name:
            continue
        if len(name) > 100:
            raise ValueError('Each tag must be 100 characters or fewer')
        if not re.match(r'^[\w\-\s]+$', name):
            raise ValueError('Tags may contain letters, numbers, spaces, underscores, and hyphens only')
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(name)

    if len(cleaned) > 20:
        raise ValueError('A maximum of 20 tags are allowed')

    return cleaned


@knowledge_bp.route('/knowledge', methods=['GET'])
def get_knowledge_items():
    """Get all knowledge items for a user"""
    try:
        # For demo purposes, using user_id = 1
        # In production, this would come from JWT token
        user_id = request.args.get('user_id', 1, type=int)

        # Query parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        search = request.args.get('search', '')
        tag_filter = request.args.get('tag', '')
        type_filter = request.args.get('type', '')

        # Base query
        query = KnowledgeItem.query.filter_by(user_id=user_id, is_archived=False)

        # Apply search filter
        if search:
            query = query.filter(
                KnowledgeItem.title.contains(search) |
                KnowledgeItem.content.contains(search)
            )

        # Apply tag filter
        if tag_filter:
            query = query.join(KnowledgeItemTag).join(Tag).filter(Tag.name == tag_filter)

        # Apply content type filter
        if type_filter:
            if type_filter not in ALLOWED_CONTENT_TYPES:
                return jsonify({'success': False, 'error': f"Unsupported type filter. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}"}), 400
            query = query.filter(KnowledgeItem.content_type == type_filter)

        # Order by importance score and creation date
        query = query.order_by(KnowledgeItem.importance_score.desc(), KnowledgeItem.created_at.desc())

        # Paginate
        items = query.paginate(page=page, per_page=per_page, error_out=False)

        # Track activity
        activity = UserActivity(
            user_id=user_id,
            activity_type='knowledge_list',
            activity_metadata={'search': search, 'tag_filter': tag_filter, 'type_filter': type_filter}
        )
        db.session.add(activity)
        db.session.commit()

        # Build response including tags for each item
        data = []
        for item in items.items:
            d = item.to_dict()
            d['tags'] = [it.tag.to_dict() for it in (item.item_tags or [])]
            data.append(d)

        return jsonify({
            'success': True,
            'data': data,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': items.total,
                'total_pages': items.pages
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@knowledge_bp.route('/knowledge', methods=['POST'])
def create_knowledge_item():
    """Create a new knowledge item"""
    try:
        data = request.get_json()

        # For demo purposes, using user_id = 1
        user_id = data.get('user_id', 1)

        # Validate required fields
        if not data.get('title'):
            return jsonify({'success': False, 'error': 'Title is required'}), 400

        # Validate content_type
        content_type = data.get('content_type', 'text')
        if content_type not in ALLOWED_CONTENT_TYPES:
            return jsonify({
                'success': False,
                'error': f"Unsupported content_type. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}"
            }), 400

        # Validate and normalize tags
        try:
            cleaned_tags = normalize_and_validate_tags(data.get('tags', []))
        except ValueError as ve:
            return jsonify({'success': False, 'error': str(ve)}), 400

        # Create knowledge item
        item = KnowledgeItem(
            user_id=user_id,
            title=data['title'],
            content=data.get('content', ''),
            content_type=content_type,
            source_url=data.get('source_url'),
            source_type=data.get('source_type'),
            item_metadata=data.get('metadata', {})
        )

        db.session.add(item)
        db.session.flush()  # Get the ID

        # Handle tags
        for tag_name in cleaned_tags:
            # Find or create tag
            tag = Tag.query.filter_by(user_id=user_id, name=tag_name).first()
            if not tag:
                tag = Tag(user_id=user_id, name=tag_name)
                db.session.add(tag)
                db.session.flush()

            # Create tag association
            item_tag = KnowledgeItemTag(knowledge_item_id=item.id, tag_id=tag.id)
            db.session.add(item_tag)

        # Track activity
        activity = UserActivity(
            user_id=user_id,
            activity_type='knowledge_create',
            entity_type='knowledge_item',
            entity_id=item.id,
            activity_metadata={'title': item.title, 'content_type': item.content_type}
        )
        db.session.add(activity)

        db.session.commit()

        return jsonify({
            'success': True,
            'data': item.to_dict()
        }), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@knowledge_bp.route('/knowledge/<item_id>', methods=['GET'])
def get_knowledge_item(item_id):
    """Get a specific knowledge item"""
    try:
        # For demo purposes, using user_id = 1
        user_id = request.args.get('user_id', 1, type=int)

        item = KnowledgeItem.query.filter_by(id=item_id, user_id=user_id).first()
        if not item:
            return jsonify({'success': False, 'error': 'Knowledge item not found'}), 404

        # Update access tracking
        item.update_access()

        # Track activity
        activity = UserActivity(
            user_id=user_id,
            activity_type='knowledge_view',
            entity_type='knowledge_item',
            entity_id=item.id,
            activity_metadata={'title': item.title}
        )
        db.session.add(activity)
        db.session.commit()

        # Get associated tags
        tags = db.session.query(Tag).join(KnowledgeItemTag).filter(
            KnowledgeItemTag.knowledge_item_id == item.id
        ).all()

        item_dict = item.to_dict()
        item_dict['tags'] = [tag.to_dict() for tag in tags]

        return jsonify({
            'success': True,
            'data': item_dict
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@knowledge_bp.route('/knowledge/<item_id>', methods=['PUT'])
def update_knowledge_item(item_id):
    """Update a knowledge item"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 1)

        item = KnowledgeItem.query.filter_by(id=item_id, user_id=user_id).first()
        if not item:
            return jsonify({'success': False, 'error': 'Knowledge item not found'}), 404

        # Update fields
        if 'title' in data:
            item.title = data['title']
        if 'content' in data:
            item.content = data['content']
        if 'content_type' in data:
            new_ct = data['content_type']
            if new_ct not in ALLOWED_CONTENT_TYPES:
                return jsonify({
                    'success': False,
                    'error': f"Unsupported content_type. Allowed: {sorted(ALLOWED_CONTENT_TYPES)}"
                }), 400
            item.content_type = new_ct
        if 'source_url' in data:
            item.source_url = data['source_url']
        if 'source_type' in data:
            item.source_type = data['source_type']
        if 'metadata' in data:
            item.item_metadata = data['metadata']

        item.updated_at = datetime.utcnow()

        # Handle tags update
        if 'tags' in data:
            try:
                cleaned_tags = normalize_and_validate_tags(data['tags'])
            except ValueError as ve:
                return jsonify({'success': False, 'error': str(ve)}), 400

            # Remove existing tag associations
            KnowledgeItemTag.query.filter_by(knowledge_item_id=item.id).delete()

            # Add new tags
            for tag_name in cleaned_tags:
                tag = Tag.query.filter_by(user_id=user_id, name=tag_name).first()
                if not tag:
                    tag = Tag(user_id=user_id, name=tag_name)
                    db.session.add(tag)
                    db.session.flush()

                item_tag = KnowledgeItemTag(knowledge_item_id=item.id, tag_id=tag.id)
                db.session.add(item_tag)

        # Track activity
        activity = UserActivity(
            user_id=user_id,
            activity_type='knowledge_update',
            entity_type='knowledge_item',
            entity_id=item.id,
            activity_metadata={'title': item.title}
        )
        db.session.add(activity)

        db.session.commit()

        return jsonify({
            'success': True,
            'data': item.to_dict()
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@knowledge_bp.route('/knowledge/<item_id>', methods=['DELETE'])
def delete_knowledge_item(item_id):
    """Delete a knowledge item (soft delete)"""
    try:
        user_id = request.args.get('user_id', 1, type=int)

        item = KnowledgeItem.query.filter_by(id=item_id, user_id=user_id).first()
        if not item:
            return jsonify({'success': False, 'error': 'Knowledge item not found'}), 404

        # Soft delete
        item.is_archived = True
        item.updated_at = datetime.utcnow()

        # Track activity
        activity = UserActivity(
            user_id=user_id,
            activity_type='knowledge_delete',
            entity_type='knowledge_item',
            entity_id=item.id,
            activity_metadata={'title': item.title}
        )
        db.session.add(activity)

        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Knowledge item deleted successfully'
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@knowledge_bp.route('/knowledge/search', methods=['POST'])
def search_knowledge():
    """Advanced search for knowledge items"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 1)

        query_text = data.get('query', '')
        filters = data.get('filters', {})

        # Base query
        query = KnowledgeItem.query.filter_by(user_id=user_id, is_archived=False)

        # Text search
        if query_text:
            query = query.filter(
                KnowledgeItem.title.contains(query_text) |
                KnowledgeItem.content.contains(query_text)
            )

        # Apply filters
        if filters.get('content_type'):
            query = query.filter(KnowledgeItem.content_type == filters['content_type'])

        if filters.get('date_from'):
            query = query.filter(KnowledgeItem.created_at >= filters['date_from'])

        if filters.get('date_to'):
            query = query.filter(KnowledgeItem.created_at <= filters['date_to'])

        if filters.get('min_importance'):
            query = query.filter(KnowledgeItem.importance_score >= filters['min_importance'])

        # Order by relevance (importance score and recency)
        query = query.order_by(
            KnowledgeItem.importance_score.desc(),
            KnowledgeItem.accessed_at.desc()
        )

        items = query.limit(50).all()

        # Track search activity
        activity = UserActivity(
            user_id=user_id,
            activity_type='knowledge_search',
            activity_metadata={'query': query_text, 'filters': filters, 'results_count': len(items)}
        )
        db.session.add(activity)
        db.session.commit()

        return jsonify({
            'success': True,
            'data': [item.to_dict() for item in items],
            'meta': {
                'query': query_text,
                'results_count': len(items)
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@knowledge_bp.route('/tags', methods=['GET'])
def get_tags():
    """Get all tags for a user"""
    try:
        user_id = request.args.get('user_id', 1, type=int)

        tags = Tag.query.filter_by(user_id=user_id).order_by(Tag.name).all()

        return jsonify({
            'success': True,
            'data': [tag.to_dict() for tag in tags]
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@knowledge_bp.route('/tags', methods=['POST'])
def create_tag():
    """Create a new tag"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 1)

        if not data.get('name'):
            return jsonify({'success': False, 'error': 'Tag name is required'}), 400

        # Check if tag already exists
        existing_tag = Tag.query.filter_by(user_id=user_id, name=data['name']).first()
        if existing_tag:
            return jsonify({'success': False, 'error': 'Tag already exists'}), 400

        tag = Tag(
            user_id=user_id,
            name=data['name'],
            color=data.get('color', '#6366f1')
        )

        db.session.add(tag)
        db.session.commit()

        return jsonify({
            'success': True,
            'data': tag.to_dict()
        }), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

