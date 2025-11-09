"""
Review Database Implementation

This module implements the ReviewDatabase class with SQLite schema and CRUD operations
for the AI Agent Review System, following the existing database patterns in the project.
"""

import sqlite3
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict

from review_system import (
    AgentReview, QualityMetrics, ReviewStatistics, 
    ReviewType, TrendDirection, NotificationStatus,
    create_empty_review_statistics
)

# Configure logging
logger = logging.getLogger(__name__)


class ReviewDatabase:
    """
    Database manager for review system with SQLite backend
    
    Handles persistent storage of reviews, quality metrics, and notifications
    following the existing database patterns in the project.
    """
    
    def __init__(self, db_path: str = "review_system.sqlite"):
        """
        Initialize review database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with review system schema"""
        logger.info(f"Initializing review database: {self.db_path}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")
                cursor = conn.cursor()
                
                # Create agent_reviews table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS agent_reviews (
                        review_id TEXT PRIMARY KEY,
                        reviewer_id TEXT NOT NULL,
                        reviewed_agent_id TEXT NOT NULL,
                        interaction_id TEXT NOT NULL,
                        review_type TEXT NOT NULL,
                        communication_quality REAL,
                        response_time_rating REAL,
                        accuracy_rating REAL,
                        helpfulness_rating REAL,
                        text_feedback TEXT,
                        review_timestamp REAL NOT NULL,
                        interaction_timestamp REAL NOT NULL,
                        review_signature TEXT NOT NULL,
                        metadata TEXT,
                        created_at REAL DEFAULT (julianday('now')),
                        UNIQUE(reviewer_id, interaction_id)
                    )
                ''')
                
                # Create quality_metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS quality_metrics (
                        metric_id TEXT PRIMARY KEY,
                        agent_id TEXT NOT NULL,
                        interaction_id TEXT NOT NULL,
                        response_time_ms REAL NOT NULL,
                        accuracy_score REAL NOT NULL,
                        protocol_compliance_score REAL NOT NULL,
                        message_clarity_score REAL NOT NULL,
                        error_rate REAL NOT NULL,
                        success_rate REAL NOT NULL,
                        timestamp REAL NOT NULL,
                        created_at REAL DEFAULT (julianday('now'))
                    )
                ''')
                
                # Create review_notifications table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS review_notifications (
                        notification_id TEXT PRIMARY KEY,
                        recipient_id TEXT NOT NULL,
                        review_id TEXT NOT NULL,
                        notification_type TEXT NOT NULL,
                        status TEXT DEFAULT 'pending',
                        created_at REAL DEFAULT (julianday('now')),
                        delivered_at REAL,
                        read_at REAL,
                        FOREIGN KEY (review_id) REFERENCES agent_reviews (review_id)
                    )
                ''')
                
                # Create indexes for better query performance
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_reviews_reviewed_agent 
                    ON agent_reviews(reviewed_agent_id)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_reviews_reviewer 
                    ON agent_reviews(reviewer_id)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_reviews_timestamp 
                    ON agent_reviews(review_timestamp)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_metrics_agent 
                    ON quality_metrics(agent_id)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_notifications_recipient 
                    ON review_notifications(recipient_id)
                ''')
                
                conn.commit()
                logger.info(f"Review database initialized successfully: {self.db_path}")
                
        except sqlite3.Error as e:
            logger.error(f"Review database initialization error: {e}")
            raise
    
    def migrate_database(self):
        """
        Perform database migrations for schema updates
        
        This method handles schema changes and data migrations
        for future versions of the review system.
        """
        logger.info("Checking for database migrations")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check current schema version
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS schema_version (
                        version INTEGER PRIMARY KEY,
                        applied_at REAL DEFAULT (julianday('now'))
                    )
                ''')
                
                cursor.execute('SELECT MAX(version) FROM schema_version')
                current_version = cursor.fetchone()[0] or 0
                
                # Apply migrations based on current version
                if current_version < 1:
                    logger.info("Applying migration v1: Adding review indexes")
                    # Indexes are already created in _init_database
                    cursor.execute('INSERT INTO schema_version (version) VALUES (1)')
                
                conn.commit()
                logger.info(f"Database migrations completed. Current version: {current_version}")
                
        except sqlite3.Error as e:
            logger.error(f"Database migration error: {e}")
            raise
    
    # Review CRUD Operations
    
    def add_review(self, review: AgentReview) -> bool:
        """
        Add a new review to the database
        
        Args:
            review: AgentReview object to store
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO agent_reviews (
                        review_id, reviewer_id, reviewed_agent_id, interaction_id,
                        review_type, communication_quality, response_time_rating,
                        accuracy_rating, helpfulness_rating, text_feedback,
                        review_timestamp, interaction_timestamp, review_signature, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    review.review_id,
                    review.reviewer_id,
                    review.reviewed_agent_id,
                    review.interaction_id,
                    review.review_type.value,
                    review.ratings.get('communication_quality'),
                    review.ratings.get('response_time'),
                    review.ratings.get('accuracy'),
                    review.ratings.get('helpfulness'),
                    review.text_feedback,
                    review.review_timestamp,
                    review.interaction_timestamp,
                    review.review_signature,
                    json.dumps(review.metadata) if review.metadata else None
                ))
                
                conn.commit()
                logger.info(f"Review added successfully: {review.review_id}")
                return True
                
        except sqlite3.IntegrityError as e:
            logger.warning(f"Review already exists or constraint violation: {e}")
            return False
        except sqlite3.Error as e:
            logger.error(f"Error adding review {review.review_id}: {e}")
            return False
    
    def get_review(self, review_id: str) -> Optional[AgentReview]:
        """
        Retrieve a specific review by ID
        
        Args:
            review_id: Unique review identifier
            
        Returns:
            AgentReview object or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT review_id, reviewer_id, reviewed_agent_id, interaction_id,
                           review_type, communication_quality, response_time_rating,
                           accuracy_rating, helpfulness_rating, text_feedback,
                           review_timestamp, interaction_timestamp, review_signature, metadata
                    FROM agent_reviews WHERE review_id = ?
                ''', (review_id,))
                
                row = cursor.fetchone()
                if row:
                    return self._row_to_review(row)
                return None
                
        except sqlite3.Error as e:
            logger.error(f"Error retrieving review {review_id}: {e}")
            return None
    
    def get_reviews_for_agent(self, agent_id: str, limit: int = 100) -> List[AgentReview]:
        """
        Get all reviews for a specific agent
        
        Args:
            agent_id: ID of the agent to get reviews for
            limit: Maximum number of reviews to return
            
        Returns:
            List of AgentReview objects
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT review_id, reviewer_id, reviewed_agent_id, interaction_id,
                           review_type, communication_quality, response_time_rating,
                           accuracy_rating, helpfulness_rating, text_feedback,
                           review_timestamp, interaction_timestamp, review_signature, metadata
                    FROM agent_reviews 
                    WHERE reviewed_agent_id = ?
                    ORDER BY review_timestamp DESC
                    LIMIT ?
                ''', (agent_id, limit))
                
                rows = cursor.fetchall()
                return [self._row_to_review(row) for row in rows]
                
        except sqlite3.Error as e:
            logger.error(f"Error retrieving reviews for agent {agent_id}: {e}")
            return []
    
    def get_reviews_by_reviewer(self, reviewer_id: str, limit: int = 100) -> List[AgentReview]:
        """
        Get all reviews submitted by a specific reviewer
        
        Args:
            reviewer_id: ID of the reviewer
            limit: Maximum number of reviews to return
            
        Returns:
            List of AgentReview objects
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT review_id, reviewer_id, reviewed_agent_id, interaction_id,
                           review_type, communication_quality, response_time_rating,
                           accuracy_rating, helpfulness_rating, text_feedback,
                           review_timestamp, interaction_timestamp, review_signature, metadata
                    FROM agent_reviews 
                    WHERE reviewer_id = ?
                    ORDER BY review_timestamp DESC
                    LIMIT ?
                ''', (reviewer_id, limit))
                
                rows = cursor.fetchall()
                return [self._row_to_review(row) for row in rows]
                
        except sqlite3.Error as e:
            logger.error(f"Error retrieving reviews by reviewer {reviewer_id}: {e}")
            return []
    
    def search_reviews(
        self, 
        agent_id: Optional[str] = None,
        reviewer_id: Optional[str] = None,
        review_type: Optional[ReviewType] = None,
        min_rating: Optional[float] = None,
        max_rating: Optional[float] = None,
        start_date: Optional[float] = None,
        end_date: Optional[float] = None,
        text_search: Optional[str] = None,
        limit: int = 100
    ) -> List[AgentReview]:
        """
        Search reviews with multiple filter criteria
        
        Args:
            agent_id: Filter by reviewed agent ID
            reviewer_id: Filter by reviewer ID
            review_type: Filter by review type
            min_rating: Minimum average rating
            max_rating: Maximum average rating
            start_date: Start timestamp for date range
            end_date: End timestamp for date range
            text_search: Search text in feedback
            limit: Maximum number of results
            
        Returns:
            List of matching AgentReview objects
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build dynamic query
                conditions = []
                params = []
                
                if agent_id:
                    conditions.append("reviewed_agent_id = ?")
                    params.append(agent_id)
                
                if reviewer_id:
                    conditions.append("reviewer_id = ?")
                    params.append(reviewer_id)
                
                if review_type:
                    conditions.append("review_type = ?")
                    params.append(review_type.value)
                
                if start_date:
                    conditions.append("review_timestamp >= ?")
                    params.append(start_date)
                
                if end_date:
                    conditions.append("review_timestamp <= ?")
                    params.append(end_date)
                
                if text_search:
                    conditions.append("text_feedback LIKE ?")
                    params.append(f"%{text_search}%")
                
                # Add rating filter using calculated average
                rating_filter = ""
                if min_rating is not None or max_rating is not None:
                    rating_calc = '''
                        (COALESCE(communication_quality, 0) + 
                         COALESCE(response_time_rating, 0) + 
                         COALESCE(accuracy_rating, 0) + 
                         COALESCE(helpfulness_rating, 0)) / 4.0
                    '''
                    
                    if min_rating is not None:
                        conditions.append(f"({rating_calc}) >= ?")
                        params.append(min_rating)
                    
                    if max_rating is not None:
                        conditions.append(f"({rating_calc}) <= ?")
                        params.append(max_rating)
                
                where_clause = " AND ".join(conditions) if conditions else "1=1"
                
                query = f'''
                    SELECT review_id, reviewer_id, reviewed_agent_id, interaction_id,
                           review_type, communication_quality, response_time_rating,
                           accuracy_rating, helpfulness_rating, text_feedback,
                           review_timestamp, interaction_timestamp, review_signature, metadata
                    FROM agent_reviews 
                    WHERE {where_clause}
                    ORDER BY review_timestamp DESC
                    LIMIT ?
                '''
                
                params.append(limit)
                cursor.execute(query, params)
                
                rows = cursor.fetchall()
                return [self._row_to_review(row) for row in rows]
                
        except sqlite3.Error as e:
            logger.error(f"Error searching reviews: {e}")
            return []
    
    def update_review(self, review: AgentReview) -> bool:
        """
        Update an existing review
        
        Args:
            review: Updated AgentReview object
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE agent_reviews SET
                        communication_quality = ?,
                        response_time_rating = ?,
                        accuracy_rating = ?,
                        helpfulness_rating = ?,
                        text_feedback = ?,
                        review_signature = ?,
                        metadata = ?
                    WHERE review_id = ?
                ''', (
                    review.ratings.get('communication_quality'),
                    review.ratings.get('response_time'),
                    review.ratings.get('accuracy'),
                    review.ratings.get('helpfulness'),
                    review.text_feedback,
                    review.review_signature,
                    json.dumps(review.metadata) if review.metadata else None,
                    review.review_id
                ))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    logger.info(f"Review updated successfully: {review.review_id}")
                    return True
                else:
                    logger.warning(f"Review not found for update: {review.review_id}")
                    return False
                
        except sqlite3.Error as e:
            logger.error(f"Error updating review {review.review_id}: {e}")
            return False
    
    def delete_review(self, review_id: str) -> bool:
        """
        Delete a review from the database
        
        Args:
            review_id: ID of the review to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete associated notifications first
                cursor.execute('DELETE FROM review_notifications WHERE review_id = ?', (review_id,))
                
                # Delete the review
                cursor.execute('DELETE FROM agent_reviews WHERE review_id = ?', (review_id,))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    logger.info(f"Review deleted successfully: {review_id}")
                    return True
                else:
                    logger.warning(f"Review not found for deletion: {review_id}")
                    return False
                
        except sqlite3.Error as e:
            logger.error(f"Error deleting review {review_id}: {e}")
            return False
    
    # Quality Metrics CRUD Operations
    
    def add_quality_metrics(self, metric_id: str, agent_id: str, interaction_id: str, metrics: QualityMetrics) -> bool:
        """
        Add quality metrics to the database
        
        Args:
            metric_id: Unique identifier for the metrics
            agent_id: ID of the agent being measured
            interaction_id: ID of the interaction
            metrics: QualityMetrics object
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO quality_metrics (
                        metric_id, agent_id, interaction_id, response_time_ms,
                        accuracy_score, protocol_compliance_score, message_clarity_score,
                        error_rate, success_rate, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metric_id,
                    agent_id,
                    interaction_id,
                    metrics.response_time_ms,
                    metrics.accuracy_score,
                    metrics.protocol_compliance_score,
                    metrics.message_clarity_score,
                    metrics.error_rate,
                    metrics.success_rate,
                    metrics.timestamp
                ))
                
                conn.commit()
                logger.info(f"Quality metrics added successfully: {metric_id}")
                return True
                
        except sqlite3.IntegrityError as e:
            logger.warning(f"Quality metrics already exist or constraint violation: {e}")
            return False
        except sqlite3.Error as e:
            logger.error(f"Error adding quality metrics {metric_id}: {e}")
            return False
    
    def get_quality_metrics(self, agent_id: str, limit: int = 100) -> List[Tuple[str, str, QualityMetrics]]:
        """
        Get quality metrics for an agent
        
        Args:
            agent_id: ID of the agent
            limit: Maximum number of metrics to return
            
        Returns:
            List of tuples (metric_id, interaction_id, QualityMetrics)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT metric_id, interaction_id, response_time_ms, accuracy_score,
                           protocol_compliance_score, message_clarity_score,
                           error_rate, success_rate, timestamp
                    FROM quality_metrics 
                    WHERE agent_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (agent_id, limit))
                
                rows = cursor.fetchall()
                results = []
                
                for row in rows:
                    metric_id, interaction_id = row[0], row[1]
                    metrics = QualityMetrics(
                        response_time_ms=row[2],
                        accuracy_score=row[3],
                        protocol_compliance_score=row[4],
                        message_clarity_score=row[5],
                        error_rate=row[6],
                        success_rate=row[7],
                        timestamp=row[8]
                    )
                    results.append((metric_id, interaction_id, metrics))
                
                return results
                
        except sqlite3.Error as e:
            logger.error(f"Error retrieving quality metrics for agent {agent_id}: {e}")
            return []
    
    # Review Statistics Operations
    
    def calculate_review_statistics(self, agent_id: str) -> ReviewStatistics:
        """
        Calculate aggregated review statistics for an agent
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            ReviewStatistics object
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get basic statistics
                cursor.execute('''
                    SELECT COUNT(*) as total_reviews,
                           AVG(communication_quality) as avg_comm,
                           AVG(response_time_rating) as avg_response,
                           AVG(accuracy_rating) as avg_accuracy,
                           AVG(helpfulness_rating) as avg_helpful
                    FROM agent_reviews 
                    WHERE reviewed_agent_id = ?
                ''', (agent_id,))
                
                stats_row = cursor.fetchone()
                
                if not stats_row or stats_row[0] == 0:
                    return create_empty_review_statistics(agent_id)
                
                total_reviews = stats_row[0]
                average_ratings = {
                    'communication_quality': stats_row[1] or 3.0,
                    'response_time': stats_row[2] or 3.0,
                    'accuracy': stats_row[3] or 3.0,
                    'helpfulness': stats_row[4] or 3.0
                }
                
                # Calculate rating distribution
                cursor.execute('''
                    SELECT 
                        SUM(CASE WHEN (communication_quality + response_time_rating + 
                                      accuracy_rating + helpfulness_rating) / 4.0 BETWEEN 1.0 AND 2.0 
                                THEN 1 ELSE 0 END) as range_1_2,
                        SUM(CASE WHEN (communication_quality + response_time_rating + 
                                      accuracy_rating + helpfulness_rating) / 4.0 BETWEEN 2.0 AND 3.0 
                                THEN 1 ELSE 0 END) as range_2_3,
                        SUM(CASE WHEN (communication_quality + response_time_rating + 
                                      accuracy_rating + helpfulness_rating) / 4.0 BETWEEN 3.0 AND 4.0 
                                THEN 1 ELSE 0 END) as range_3_4,
                        SUM(CASE WHEN (communication_quality + response_time_rating + 
                                      accuracy_rating + helpfulness_rating) / 4.0 BETWEEN 4.0 AND 5.0 
                                THEN 1 ELSE 0 END) as range_4_5
                    FROM agent_reviews 
                    WHERE reviewed_agent_id = ?
                ''', (agent_id,))
                
                dist_row = cursor.fetchone()
                review_distribution = {
                    '1-2': dist_row[0] or 0,
                    '2-3': dist_row[1] or 0,
                    '3-4': dist_row[2] or 0,
                    '4-5': dist_row[3] or 0
                }
                
                # Calculate trend (simplified - compare recent vs older reviews)
                trend = self._calculate_trend(cursor, agent_id)
                
                # Calculate performance percentile (simplified)
                overall_avg = sum(average_ratings.values()) / len(average_ratings)
                performance_percentile = min(100.0, max(0.0, (overall_avg - 1.0) / 4.0 * 100.0))
                
                return ReviewStatistics(
                    agent_id=agent_id,
                    total_reviews=total_reviews,
                    average_ratings=average_ratings,
                    review_trend=trend,
                    performance_percentile=performance_percentile,
                    last_updated=time.time(),
                    review_distribution=review_distribution
                )
                
        except sqlite3.Error as e:
            logger.error(f"Error calculating review statistics for agent {agent_id}: {e}")
            return create_empty_review_statistics(agent_id)
    
    def _calculate_trend(self, cursor, agent_id: str) -> TrendDirection:
        """Calculate performance trend for an agent"""
        try:
            # Get recent reviews (last 30 days) vs older reviews
            recent_cutoff = time.time() - (30 * 24 * 3600)  # 30 days ago
            
            cursor.execute('''
                SELECT AVG((communication_quality + response_time_rating + 
                           accuracy_rating + helpfulness_rating) / 4.0) as recent_avg
                FROM agent_reviews 
                WHERE reviewed_agent_id = ? AND review_timestamp >= ?
            ''', (agent_id, recent_cutoff))
            
            recent_avg = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT AVG((communication_quality + response_time_rating + 
                           accuracy_rating + helpfulness_rating) / 4.0) as older_avg
                FROM agent_reviews 
                WHERE reviewed_agent_id = ? AND review_timestamp < ?
            ''', (agent_id, recent_cutoff))
            
            older_avg = cursor.fetchone()[0]
            
            if recent_avg is None or older_avg is None:
                return TrendDirection.INSUFFICIENT_DATA
            
            diff = recent_avg - older_avg
            
            if diff > 0.2:
                return TrendDirection.IMPROVING
            elif diff < -0.2:
                return TrendDirection.DECLINING
            else:
                return TrendDirection.STABLE
                
        except sqlite3.Error:
            return TrendDirection.INSUFFICIENT_DATA
    
    # Notification Operations
    
    def add_notification(self, notification_id: str, recipient_id: str, review_id: str, 
                        notification_type: str) -> bool:
        """
        Add a review notification
        
        Args:
            notification_id: Unique notification ID
            recipient_id: ID of the recipient agent
            review_id: ID of the related review
            notification_type: Type of notification
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO review_notifications (
                        notification_id, recipient_id, review_id, notification_type
                    ) VALUES (?, ?, ?, ?)
                ''', (notification_id, recipient_id, review_id, notification_type))
                
                conn.commit()
                logger.info(f"Notification added successfully: {notification_id}")
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Error adding notification {notification_id}: {e}")
            return False
    
    def get_pending_notifications(self, recipient_id: str) -> List[Dict[str, Any]]:
        """
        Get pending notifications for a recipient
        
        Args:
            recipient_id: ID of the recipient agent
            
        Returns:
            List of notification dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT n.notification_id, n.review_id, n.notification_type,
                           n.created_at, r.reviewer_id, r.reviewed_agent_id
                    FROM review_notifications n
                    JOIN agent_reviews r ON n.review_id = r.review_id
                    WHERE n.recipient_id = ? AND n.status = 'pending'
                    ORDER BY n.created_at DESC
                ''', (recipient_id,))
                
                rows = cursor.fetchall()
                notifications = []
                
                for row in rows:
                    notifications.append({
                        'notification_id': row[0],
                        'review_id': row[1],
                        'notification_type': row[2],
                        'created_at': row[3],
                        'reviewer_id': row[4],
                        'reviewed_agent_id': row[5]
                    })
                
                return notifications
                
        except sqlite3.Error as e:
            logger.error(f"Error retrieving notifications for {recipient_id}: {e}")
            return []
    
    def mark_notification_delivered(self, notification_id: str) -> bool:
        """
        Mark a notification as delivered
        
        Args:
            notification_id: ID of the notification
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE review_notifications 
                    SET status = 'delivered', delivered_at = julianday('now')
                    WHERE notification_id = ?
                ''', (notification_id,))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    return True
                return False
                
        except sqlite3.Error as e:
            logger.error(f"Error marking notification delivered {notification_id}: {e}")
            return False
    
    # Utility Methods
    
    def _row_to_review(self, row) -> AgentReview:
        """Convert database row to AgentReview object"""
        ratings = {}
        if row[5] is not None:  # communication_quality
            ratings['communication_quality'] = row[5]
        if row[6] is not None:  # response_time_rating
            ratings['response_time'] = row[6]
        if row[7] is not None:  # accuracy_rating
            ratings['accuracy'] = row[7]
        if row[8] is not None:  # helpfulness_rating
            ratings['helpfulness'] = row[8]
        
        metadata = json.loads(row[13]) if row[13] else {}
        
        return AgentReview(
            review_id=row[0],
            reviewer_id=row[1],
            reviewed_agent_id=row[2],
            interaction_id=row[3],
            review_type=ReviewType(row[4]),
            ratings=ratings,
            text_feedback=row[9],
            review_timestamp=row[10],
            interaction_timestamp=row[11],
            review_signature=row[12],
            metadata=metadata
        )
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Returns:
            Dictionary with database statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Count records in each table
                cursor.execute('SELECT COUNT(*) FROM agent_reviews')
                review_count = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM quality_metrics')
                metrics_count = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM review_notifications')
                notification_count = cursor.fetchone()[0]
                
                # Get database file size
                db_size = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
                
                return {
                    'database_path': str(self.db_path),
                    'database_size_mb': round(db_size, 2),
                    'total_reviews': review_count,
                    'total_quality_metrics': metrics_count,
                    'total_notifications': notification_count,
                    'last_updated': time.time()
                }
                
        except sqlite3.Error as e:
            logger.error(f"Error getting database statistics: {e}")
            return {
                'database_path': str(self.db_path),
                'error': str(e)
            }
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> int:
        """
        Clean up old review data
        
        Args:
            days_to_keep: Number of days of data to retain
            
        Returns:
            Number of records deleted
        """
        try:
            cutoff_time = time.time() - (days_to_keep * 24 * 3600)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old notifications first (foreign key constraint)
                cursor.execute('''
                    DELETE FROM review_notifications 
                    WHERE created_at < julianday(?, 'unixepoch')
                ''', (cutoff_time,))
                
                notifications_deleted = cursor.rowcount
                
                # Delete old reviews
                cursor.execute('''
                    DELETE FROM agent_reviews 
                    WHERE review_timestamp < ?
                ''', (cutoff_time,))
                
                reviews_deleted = cursor.rowcount
                
                # Delete old quality metrics
                cursor.execute('''
                    DELETE FROM quality_metrics 
                    WHERE timestamp < ?
                ''', (cutoff_time,))
                
                metrics_deleted = cursor.rowcount
                
                conn.commit()
                
                total_deleted = reviews_deleted + metrics_deleted + notifications_deleted
                logger.info(f"Cleaned up {total_deleted} old records (older than {days_to_keep} days)")
                
                return total_deleted
                
        except sqlite3.Error as e:
            logger.error(f"Error cleaning up old data: {e}")
            return 0