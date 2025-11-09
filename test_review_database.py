"""
Unit Tests for Review Database

Comprehensive tests for ReviewDatabase class operations and error handling.
"""

import unittest
import tempfile
import os
import time
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from review_database import ReviewDatabase
from review_system import (
    AgentReview, QualityMetrics, ReviewStatistics,
    ReviewType, TrendDirection, NotificationStatus,
    create_review_id, create_automatic_review
)


class TestReviewDatabase(unittest.TestCase):
    """Test cases for ReviewDatabase class"""
    
    def setUp(self):
        """Set up test database for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_review.sqlite")
        self.db = ReviewDatabase(self.db_path)
        
        # Create sample data
        self.sample_review = AgentReview(
            review_id="test_review_001",
            reviewer_id="agent_alice",
            reviewed_agent_id="agent_bob",
            interaction_id="interaction_001",
            review_type=ReviewType.MANUAL,
            ratings={
                'communication_quality': 4.5,
                'response_time': 4.0,
                'accuracy': 4.2,
                'helpfulness': 4.8
            },
            text_feedback="Excellent collaboration and quick responses",
            review_timestamp=time.time(),
            interaction_timestamp=time.time() - 3600,  # 1 hour ago
            review_signature="test_signature_123",
            metadata={"test_key": "test_value"}
        )
        
        self.sample_metrics = QualityMetrics(
            response_time_ms=250.0,
            accuracy_score=0.95,
            protocol_compliance_score=0.98,
            message_clarity_score=0.92,
            error_rate=0.02,
            success_rate=0.98,
            timestamp=time.time()
        )
    
    def tearDown(self):
        """Clean up test database after each test"""
        # Close database connection properly
        if hasattr(self, 'db'):
            del self.db
        
        # Try to remove file with retry for Windows file locking
        import time
        for _ in range(3):
            try:
                if os.path.exists(self.db_path):
                    os.remove(self.db_path)
                break
            except PermissionError:
                time.sleep(0.1)
        
        try:
            os.rmdir(self.temp_dir)
        except OSError:
            pass  # Directory might not be empty, ignore
    
    def test_database_initialization(self):
        """Test database initialization and schema creation"""
        # Database should be created and initialized
        self.assertTrue(os.path.exists(self.db_path))
        
        # Check that tables exist
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check agent_reviews table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='agent_reviews'")
            self.assertIsNotNone(cursor.fetchone())
            
            # Check quality_metrics table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='quality_metrics'")
            self.assertIsNotNone(cursor.fetchone())
            
            # Check review_notifications table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='review_notifications'")
            self.assertIsNotNone(cursor.fetchone())
            
            # Check indexes exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_reviews_reviewed_agent'")
            self.assertIsNotNone(cursor.fetchone())
    
    def test_database_migration(self):
        """Test database migration functionality"""
        # Migration should run without errors
        self.db.migrate_database()
        
        # Check schema_version table exists
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'")
            self.assertIsNotNone(cursor.fetchone())
    
    def test_add_review_success(self):
        """Test successful review addition"""
        result = self.db.add_review(self.sample_review)
        self.assertTrue(result)
        
        # Verify review was added
        retrieved_review = self.db.get_review(self.sample_review.review_id)
        self.assertIsNotNone(retrieved_review)
        self.assertEqual(retrieved_review.review_id, self.sample_review.review_id)
        self.assertEqual(retrieved_review.reviewer_id, self.sample_review.reviewer_id)
        self.assertEqual(retrieved_review.reviewed_agent_id, self.sample_review.reviewed_agent_id)
    
    def test_add_review_duplicate(self):
        """Test adding duplicate review (should fail)"""
        # Add review first time
        result1 = self.db.add_review(self.sample_review)
        self.assertTrue(result1)
        
        # Try to add same review again (should fail due to unique constraint)
        result2 = self.db.add_review(self.sample_review)
        self.assertFalse(result2)
    
    def test_get_review_not_found(self):
        """Test retrieving non-existent review"""
        result = self.db.get_review("non_existent_review")
        self.assertIsNone(result)
    
    def test_get_reviews_for_agent(self):
        """Test retrieving reviews for a specific agent"""
        # Add multiple reviews for the same agent
        review1 = self.sample_review
        review2 = AgentReview(
            review_id="test_review_002",
            reviewer_id="agent_charlie",
            reviewed_agent_id="agent_bob",  # Same agent as review1
            interaction_id="interaction_002",
            review_type=ReviewType.AUTOMATIC,
            ratings={'communication_quality': 3.5, 'response_time': 3.0, 'accuracy': 3.8, 'helpfulness': 3.2},
            text_feedback=None,
            review_timestamp=time.time() + 100,
            interaction_timestamp=time.time() - 1800,
            review_signature="test_signature_456",
            metadata={}
        )
        
        self.db.add_review(review1)
        self.db.add_review(review2)
        
        # Retrieve reviews for agent_bob
        reviews = self.db.get_reviews_for_agent("agent_bob")
        self.assertEqual(len(reviews), 2)
        
        # Should be ordered by timestamp (newest first)
        self.assertEqual(reviews[0].review_id, "test_review_002")
        self.assertEqual(reviews[1].review_id, "test_review_001")
    
    def test_get_reviews_by_reviewer(self):
        """Test retrieving reviews by reviewer"""
        self.db.add_review(self.sample_review)
        
        reviews = self.db.get_reviews_by_reviewer("agent_alice")
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0].review_id, self.sample_review.review_id)
        
        # Test with non-existent reviewer
        reviews = self.db.get_reviews_by_reviewer("non_existent_agent")
        self.assertEqual(len(reviews), 0)
    
    def test_search_reviews_basic(self):
        """Test basic review search functionality"""
        self.db.add_review(self.sample_review)
        
        # Search by agent_id
        reviews = self.db.search_reviews(agent_id="agent_bob")
        self.assertEqual(len(reviews), 1)
        
        # Search by reviewer_id
        reviews = self.db.search_reviews(reviewer_id="agent_alice")
        self.assertEqual(len(reviews), 1)
        
        # Search by review_type
        reviews = self.db.search_reviews(review_type=ReviewType.MANUAL)
        self.assertEqual(len(reviews), 1)
        
        reviews = self.db.search_reviews(review_type=ReviewType.AUTOMATIC)
        self.assertEqual(len(reviews), 0)
    
    def test_search_reviews_rating_filter(self):
        """Test review search with rating filters"""
        self.db.add_review(self.sample_review)
        
        # Search with min_rating (should find review with avg ~4.4)
        reviews = self.db.search_reviews(min_rating=4.0)
        self.assertEqual(len(reviews), 1)
        
        reviews = self.db.search_reviews(min_rating=5.0)
        self.assertEqual(len(reviews), 0)
        
        # Search with max_rating
        reviews = self.db.search_reviews(max_rating=5.0)
        self.assertEqual(len(reviews), 1)
        
        reviews = self.db.search_reviews(max_rating=3.0)
        self.assertEqual(len(reviews), 0)
    
    def test_search_reviews_date_filter(self):
        """Test review search with date filters"""
        self.db.add_review(self.sample_review)
        
        # Search with start_date (should find recent review)
        start_date = time.time() - 7200  # 2 hours ago
        reviews = self.db.search_reviews(start_date=start_date)
        self.assertEqual(len(reviews), 1)
        
        # Search with start_date in future (should find nothing)
        start_date = time.time() + 3600  # 1 hour in future
        reviews = self.db.search_reviews(start_date=start_date)
        self.assertEqual(len(reviews), 0)
    
    def test_search_reviews_text_search(self):
        """Test review search with text search"""
        self.db.add_review(self.sample_review)
        
        # Search for text in feedback
        reviews = self.db.search_reviews(text_search="Excellent")
        self.assertEqual(len(reviews), 1)
        
        reviews = self.db.search_reviews(text_search="terrible")
        self.assertEqual(len(reviews), 0)
    
    def test_update_review(self):
        """Test review update functionality"""
        # Add original review
        self.db.add_review(self.sample_review)
        
        # Update review
        updated_review = self.sample_review
        updated_review.text_feedback = "Updated feedback"
        updated_review.ratings['communication_quality'] = 5.0
        
        result = self.db.update_review(updated_review)
        self.assertTrue(result)
        
        # Verify update
        retrieved_review = self.db.get_review(self.sample_review.review_id)
        self.assertEqual(retrieved_review.text_feedback, "Updated feedback")
        self.assertEqual(retrieved_review.ratings['communication_quality'], 5.0)
    
    def test_update_review_not_found(self):
        """Test updating non-existent review"""
        result = self.db.update_review(self.sample_review)
        self.assertFalse(result)
    
    def test_delete_review(self):
        """Test review deletion"""
        # Add review
        self.db.add_review(self.sample_review)
        
        # Verify it exists
        review = self.db.get_review(self.sample_review.review_id)
        self.assertIsNotNone(review)
        
        # Delete review
        result = self.db.delete_review(self.sample_review.review_id)
        self.assertTrue(result)
        
        # Verify it's gone
        review = self.db.get_review(self.sample_review.review_id)
        self.assertIsNone(review)
    
    def test_delete_review_not_found(self):
        """Test deleting non-existent review"""
        result = self.db.delete_review("non_existent_review")
        self.assertFalse(result)
    
    def test_add_quality_metrics(self):
        """Test adding quality metrics"""
        result = self.db.add_quality_metrics(
            "metric_001", 
            "agent_bob", 
            "interaction_001", 
            self.sample_metrics
        )
        self.assertTrue(result)
        
        # Verify metrics were added
        metrics_list = self.db.get_quality_metrics("agent_bob")
        self.assertEqual(len(metrics_list), 1)
        
        metric_id, interaction_id, metrics = metrics_list[0]
        self.assertEqual(metric_id, "metric_001")
        self.assertEqual(interaction_id, "interaction_001")
        self.assertEqual(metrics.response_time_ms, self.sample_metrics.response_time_ms)
    
    def test_get_quality_metrics_empty(self):
        """Test getting quality metrics for agent with no metrics"""
        metrics_list = self.db.get_quality_metrics("non_existent_agent")
        self.assertEqual(len(metrics_list), 0)
    
    def test_calculate_review_statistics_empty(self):
        """Test calculating statistics for agent with no reviews"""
        stats = self.db.calculate_review_statistics("agent_bob")
        
        self.assertEqual(stats.agent_id, "agent_bob")
        self.assertEqual(stats.total_reviews, 0)
        self.assertEqual(stats.review_trend, TrendDirection.INSUFFICIENT_DATA)
        self.assertEqual(stats.performance_percentile, 0.0)
    
    def test_calculate_review_statistics_with_data(self):
        """Test calculating statistics with review data"""
        # Add sample review
        self.db.add_review(self.sample_review)
        
        stats = self.db.calculate_review_statistics("agent_bob")
        
        self.assertEqual(stats.agent_id, "agent_bob")
        self.assertEqual(stats.total_reviews, 1)
        self.assertGreater(stats.performance_percentile, 0.0)
        self.assertIn('communication_quality', stats.average_ratings)
        self.assertIn('1-2', stats.review_distribution)
    
    def test_add_notification(self):
        """Test adding review notification"""
        # First add a review
        self.db.add_review(self.sample_review)
        
        result = self.db.add_notification(
            "notification_001",
            "agent_bob",
            self.sample_review.review_id,
            "new_review"
        )
        self.assertTrue(result)
        
        # Verify notification was added
        notifications = self.db.get_pending_notifications("agent_bob")
        self.assertEqual(len(notifications), 1)
        self.assertEqual(notifications[0]['notification_id'], "notification_001")
    
    def test_get_pending_notifications_empty(self):
        """Test getting notifications for agent with no notifications"""
        notifications = self.db.get_pending_notifications("agent_bob")
        self.assertEqual(len(notifications), 0)
    
    def test_mark_notification_delivered(self):
        """Test marking notification as delivered"""
        # Add review and notification
        self.db.add_review(self.sample_review)
        self.db.add_notification(
            "notification_001",
            "agent_bob", 
            self.sample_review.review_id,
            "new_review"
        )
        
        # Mark as delivered
        result = self.db.mark_notification_delivered("notification_001")
        self.assertTrue(result)
        
        # Should no longer appear in pending notifications
        notifications = self.db.get_pending_notifications("agent_bob")
        self.assertEqual(len(notifications), 0)
    
    def test_mark_notification_delivered_not_found(self):
        """Test marking non-existent notification as delivered"""
        result = self.db.mark_notification_delivered("non_existent_notification")
        self.assertFalse(result)
    
    def test_get_database_stats(self):
        """Test database statistics retrieval"""
        # Add some data
        self.db.add_review(self.sample_review)
        self.db.add_quality_metrics("metric_001", "agent_bob", "interaction_001", self.sample_metrics)
        
        stats = self.db.get_database_stats()
        
        self.assertIn('database_path', stats)
        self.assertIn('database_size_mb', stats)
        self.assertIn('total_reviews', stats)
        self.assertIn('total_quality_metrics', stats)
        self.assertIn('total_notifications', stats)
        
        self.assertEqual(stats['total_reviews'], 1)
        self.assertEqual(stats['total_quality_metrics'], 1)
        self.assertGreaterEqual(stats['database_size_mb'], 0)
    
    def test_cleanup_old_data(self):
        """Test cleanup of old data"""
        # Add review with old timestamp
        old_review = self.sample_review
        old_review.review_timestamp = time.time() - (100 * 24 * 3600)  # 100 days ago
        old_review.review_id = "old_review_001"
        
        self.db.add_review(old_review)
        
        # Add recent review
        recent_review = self.sample_review
        recent_review.review_timestamp = time.time()
        recent_review.review_id = "recent_review_001"
        
        self.db.add_review(recent_review)
        
        # Cleanup data older than 90 days
        deleted_count = self.db.cleanup_old_data(days_to_keep=90)
        self.assertGreater(deleted_count, 0)
        
        # Verify old review is gone, recent review remains
        old_retrieved = self.db.get_review("old_review_001")
        recent_retrieved = self.db.get_review("recent_review_001")
        
        self.assertIsNone(old_retrieved)
        self.assertIsNotNone(recent_retrieved)
    
    def test_database_error_handling(self):
        """Test database error handling"""
        # Test with invalid database path
        with patch('sqlite3.connect') as mock_connect:
            mock_connect.side_effect = Exception("Database connection failed")
            
            # Should handle error gracefully
            result = self.db.add_review(self.sample_review)
            self.assertFalse(result)
    
    def test_invalid_review_data(self):
        """Test handling of invalid review data"""
        # Test with invalid review - create new instance to avoid modifying sample
        try:
            invalid_review = AgentReview(
                review_id="invalid_review_001",
                reviewer_id="agent_alice",
                reviewed_agent_id="agent_bob",
                interaction_id="interaction_001",
                review_type=ReviewType.MANUAL,
                ratings={},  # Empty ratings should cause validation error
                text_feedback="Test feedback",
                review_timestamp=time.time(),
                interaction_timestamp=time.time() - 3600,
                review_signature="test_signature",
                metadata={}
            )
            self.fail("Should have raised ValueError for empty ratings")
        except ValueError:
            pass  # Expected behavior
    
    def test_concurrent_access(self):
        """Test concurrent database access"""
        import threading
        import time
        
        results = []
        
        def add_review_worker(review_id):
            review = self.sample_review
            review.review_id = f"concurrent_review_{review_id}"
            review.interaction_id = f"interaction_{review_id}"
            result = self.db.add_review(review)
            results.append(result)
        
        # Create multiple threads to test concurrent access
        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_review_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All operations should succeed
        self.assertEqual(len(results), 5)
        self.assertTrue(all(results))
        
        # Verify all reviews were added
        reviews = self.db.get_reviews_for_agent("agent_bob")
        self.assertEqual(len(reviews), 5)


class TestReviewDatabaseIntegration(unittest.TestCase):
    """Integration tests for ReviewDatabase with review system components"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "integration_test.sqlite")
        self.db = ReviewDatabase(self.db_path)
    
    def tearDown(self):
        """Clean up integration test environment"""
        # Close database connection properly
        if hasattr(self, 'db'):
            del self.db
        
        # Try to remove file with retry for Windows file locking
        import time
        for _ in range(3):
            try:
                if os.path.exists(self.db_path):
                    os.remove(self.db_path)
                break
            except PermissionError:
                time.sleep(0.1)
        
        try:
            os.rmdir(self.temp_dir)
        except OSError:
            pass  # Directory might not be empty, ignore
    
    def test_automatic_review_creation_and_storage(self):
        """Test creating and storing automatic reviews"""
        # Create quality metrics
        metrics = QualityMetrics(
            response_time_ms=150.0,
            accuracy_score=0.92,
            protocol_compliance_score=0.95,
            message_clarity_score=0.88,
            error_rate=0.03,
            success_rate=0.97,
            timestamp=time.time()
        )
        
        # Create automatic review from metrics
        auto_review = create_automatic_review(
            reviewer_id="system",
            reviewed_agent_id="agent_bob",
            interaction_id="interaction_auto_001",
            quality_metrics=metrics,
            interaction_timestamp=time.time() - 1800,
            signature="auto_signature_123"
        )
        
        # Store review and metrics
        review_result = self.db.add_review(auto_review)
        metrics_result = self.db.add_quality_metrics(
            "metric_auto_001",
            "agent_bob",
            "interaction_auto_001",
            metrics
        )
        
        self.assertTrue(review_result)
        self.assertTrue(metrics_result)
        
        # Verify automatic review properties
        retrieved_review = self.db.get_review(auto_review.review_id)
        self.assertEqual(retrieved_review.review_type, ReviewType.AUTOMATIC)
        self.assertEqual(retrieved_review.reviewer_id, "system")
        self.assertIsNone(retrieved_review.text_feedback)
        self.assertIn('quality_metrics', retrieved_review.metadata)
    
    def test_review_workflow_with_notifications(self):
        """Test complete review workflow with notifications"""
        # Create and add review
        review = AgentReview(
            review_id=create_review_id(),
            reviewer_id="agent_alice",
            reviewed_agent_id="agent_bob",
            interaction_id="interaction_workflow_001",
            review_type=ReviewType.MANUAL,
            ratings={
                'communication_quality': 4.0,
                'response_time': 3.5,
                'accuracy': 4.2,
                'helpfulness': 4.1
            },
            text_feedback="Good collaboration overall",
            review_timestamp=time.time(),
            interaction_timestamp=time.time() - 3600,
            review_signature="workflow_signature_123",
            metadata={"workflow_test": True}
        )
        
        # Add review
        self.assertTrue(self.db.add_review(review))
        
        # Add notification
        notification_id = f"notification_{int(time.time())}"
        self.assertTrue(self.db.add_notification(
            notification_id,
            "agent_bob",
            review.review_id,
            "new_review"
        ))
        
        # Check pending notifications
        notifications = self.db.get_pending_notifications("agent_bob")
        self.assertEqual(len(notifications), 1)
        self.assertEqual(notifications[0]['review_id'], review.review_id)
        
        # Mark notification as delivered
        self.assertTrue(self.db.mark_notification_delivered(notification_id))
        
        # Should no longer be pending
        notifications = self.db.get_pending_notifications("agent_bob")
        self.assertEqual(len(notifications), 0)
        
        # Calculate statistics
        stats = self.db.calculate_review_statistics("agent_bob")
        self.assertEqual(stats.total_reviews, 1)
        self.assertGreater(stats.performance_percentile, 0)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
    
    # Run tests
    unittest.main(verbosity=2)