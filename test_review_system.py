"""
Unit tests for the Review System Core Data Structures

This module contains comprehensive unit tests for AgentReview, QualityMetrics,
ReviewStatistics dataclasses and related enums, including validation and serialization tests.
"""

import json
import time
import unittest
from unittest.mock import patch
import uuid

from review_system import (
    AgentReview, QualityMetrics, ReviewStatistics,
    ReviewType, TrendDirection, NotificationStatus,
    create_review_id, create_automatic_review, create_empty_review_statistics,
    _response_time_to_rating
)


class TestReviewSystemEnums(unittest.TestCase):
    """Test cases for review system enums"""
    
    def test_review_type_enum(self):
        """Test ReviewType enum values"""
        self.assertEqual(ReviewType.MANUAL.value, "manual")
        self.assertEqual(ReviewType.AUTOMATIC.value, "automatic")
        
        # Test enum creation from string
        self.assertEqual(ReviewType("manual"), ReviewType.MANUAL)
        self.assertEqual(ReviewType("automatic"), ReviewType.AUTOMATIC)
    
    def test_trend_direction_enum(self):
        """Test TrendDirection enum values"""
        self.assertEqual(TrendDirection.IMPROVING.value, "improving")
        self.assertEqual(TrendDirection.DECLINING.value, "declining")
        self.assertEqual(TrendDirection.STABLE.value, "stable")
        self.assertEqual(TrendDirection.INSUFFICIENT_DATA.value, "insufficient_data")
    
    def test_notification_status_enum(self):
        """Test NotificationStatus enum values"""
        self.assertEqual(NotificationStatus.PENDING.value, "pending")
        self.assertEqual(NotificationStatus.DELIVERED.value, "delivered")
        self.assertEqual(NotificationStatus.READ.value, "read")
        self.assertEqual(NotificationStatus.FAILED.value, "failed")


class TestAgentReview(unittest.TestCase):
    """Test cases for AgentReview dataclass"""
    
    def setUp(self):
        """Set up test data"""
        self.valid_review_data = {
            'review_id': 'review_123',
            'reviewer_id': 'agent_alice',
            'reviewed_agent_id': 'agent_bob',
            'interaction_id': 'interaction_456',
            'review_type': ReviewType.MANUAL,
            'ratings': {
                'communication_quality': 4.5,
                'response_time': 3.8,
                'accuracy': 4.2,
                'helpfulness': 4.0
            },
            'text_feedback': 'Great response quality',
            'review_timestamp': time.time(),
            'interaction_timestamp': time.time() - 100,
            'review_signature': 'signature_123',
            'metadata': {'test': 'data'}
        }
    
    def test_valid_agent_review_creation(self):
        """Test creating a valid AgentReview"""
        review = AgentReview(**self.valid_review_data)
        
        self.assertEqual(review.review_id, 'review_123')
        self.assertEqual(review.reviewer_id, 'agent_alice')
        self.assertEqual(review.reviewed_agent_id, 'agent_bob')
        self.assertEqual(review.review_type, ReviewType.MANUAL)
        self.assertEqual(len(review.ratings), 4)
        self.assertEqual(review.text_feedback, 'Great response quality')
    
    def test_rating_validation_valid_ratings(self):
        """Test that valid ratings pass validation"""
        # This should not raise any exception
        review = AgentReview(**self.valid_review_data)
        self.assertIsInstance(review, AgentReview)
    
    def test_rating_validation_invalid_criterion(self):
        """Test that invalid rating criteria raise ValueError"""
        invalid_data = self.valid_review_data.copy()
        invalid_data['ratings'] = {'invalid_criterion': 4.0}
        
        with self.assertRaises(ValueError) as context:
            AgentReview(**invalid_data)
        
        self.assertIn("Invalid rating criterion", str(context.exception))
    
    def test_rating_validation_out_of_range(self):
        """Test that ratings outside 1.0-5.0 range raise ValueError"""
        invalid_data = self.valid_review_data.copy()
        invalid_data['ratings'] = {'communication_quality': 6.0}
        
        with self.assertRaises(ValueError) as context:
            AgentReview(**invalid_data)
        
        self.assertIn("must be between 1.0 and 5.0", str(context.exception))
    
    def test_rating_validation_non_numeric(self):
        """Test that non-numeric ratings raise ValueError"""
        invalid_data = self.valid_review_data.copy()
        invalid_data['ratings'] = {'communication_quality': 'excellent'}
        
        with self.assertRaises(ValueError) as context:
            AgentReview(**invalid_data)
        
        self.assertIn("must be numeric", str(context.exception))
    
    def test_rating_validation_empty_ratings(self):
        """Test that empty ratings dictionary raises ValueError"""
        invalid_data = self.valid_review_data.copy()
        invalid_data['ratings'] = {}
        
        with self.assertRaises(ValueError) as context:
            AgentReview(**invalid_data)
        
        self.assertIn("cannot be empty", str(context.exception))
    
    def test_timestamp_validation_invalid_review_timestamp(self):
        """Test that invalid review timestamp raises ValueError"""
        invalid_data = self.valid_review_data.copy()
        invalid_data['review_timestamp'] = -1.0
        
        with self.assertRaises(ValueError) as context:
            AgentReview(**invalid_data)
        
        self.assertIn("Review timestamp must be positive", str(context.exception))
    
    def test_timestamp_validation_invalid_interaction_timestamp(self):
        """Test that invalid interaction timestamp raises ValueError"""
        invalid_data = self.valid_review_data.copy()
        invalid_data['interaction_timestamp'] = 0.0
        
        with self.assertRaises(ValueError) as context:
            AgentReview(**invalid_data)
        
        self.assertIn("Interaction timestamp must be positive", str(context.exception))
    
    def test_timestamp_validation_review_before_interaction(self):
        """Test that review timestamp before interaction timestamp raises ValueError"""
        invalid_data = self.valid_review_data.copy()
        invalid_data['review_timestamp'] = 100.0
        invalid_data['interaction_timestamp'] = 200.0
        
        with self.assertRaises(ValueError) as context:
            AgentReview(**invalid_data)
        
        self.assertIn("cannot be before interaction timestamp", str(context.exception))
    
    def test_get_average_rating(self):
        """Test average rating calculation"""
        review = AgentReview(**self.valid_review_data)
        expected_average = (4.5 + 3.8 + 4.2 + 4.0) / 4
        self.assertAlmostEqual(review.get_average_rating(), expected_average, places=2)
    
    def test_to_dict_serialization(self):
        """Test conversion to dictionary"""
        review = AgentReview(**self.valid_review_data)
        review_dict = review.to_dict()
        
        self.assertEqual(review_dict['review_id'], 'review_123')
        self.assertEqual(review_dict['review_type'], 'manual')
        self.assertIsInstance(review_dict['ratings'], dict)
    
    def test_from_dict_deserialization(self):
        """Test creation from dictionary"""
        review = AgentReview(**self.valid_review_data)
        review_dict = review.to_dict()
        
        restored_review = AgentReview.from_dict(review_dict)
        
        self.assertEqual(restored_review.review_id, review.review_id)
        self.assertEqual(restored_review.review_type, review.review_type)
        self.assertEqual(restored_review.ratings, review.ratings)
    
    def test_json_serialization_roundtrip(self):
        """Test JSON serialization and deserialization"""
        review = AgentReview(**self.valid_review_data)
        
        # Serialize to JSON
        json_str = review.to_json()
        self.assertIsInstance(json_str, str)
        
        # Deserialize from JSON
        restored_review = AgentReview.from_json(json_str)
        
        self.assertEqual(restored_review.review_id, review.review_id)
        self.assertEqual(restored_review.review_type, review.review_type)
        self.assertEqual(restored_review.ratings, review.ratings)


class TestQualityMetrics(unittest.TestCase):
    """Test cases for QualityMetrics dataclass"""
    
    def setUp(self):
        """Set up test data"""
        self.valid_metrics_data = {
            'response_time_ms': 250.5,
            'accuracy_score': 0.95,
            'protocol_compliance_score': 0.98,
            'message_clarity_score': 0.87,
            'error_rate': 0.02,
            'success_rate': 0.96,
            'timestamp': time.time()
        }
    
    def test_valid_quality_metrics_creation(self):
        """Test creating valid QualityMetrics"""
        metrics = QualityMetrics(**self.valid_metrics_data)
        
        self.assertEqual(metrics.response_time_ms, 250.5)
        self.assertEqual(metrics.accuracy_score, 0.95)
        self.assertEqual(metrics.protocol_compliance_score, 0.98)
    
    def test_score_validation_valid_scores(self):
        """Test that valid scores pass validation"""
        # This should not raise any exception
        metrics = QualityMetrics(**self.valid_metrics_data)
        self.assertIsInstance(metrics, QualityMetrics)
    
    def test_score_validation_out_of_range_high(self):
        """Test that scores above 1.0 raise ValueError"""
        invalid_data = self.valid_metrics_data.copy()
        invalid_data['accuracy_score'] = 1.5
        
        with self.assertRaises(ValueError) as context:
            QualityMetrics(**invalid_data)
        
        self.assertIn("must be between 0.0 and 1.0", str(context.exception))
    
    def test_score_validation_out_of_range_low(self):
        """Test that scores below 0.0 raise ValueError"""
        invalid_data = self.valid_metrics_data.copy()
        invalid_data['error_rate'] = -0.1
        
        with self.assertRaises(ValueError) as context:
            QualityMetrics(**invalid_data)
        
        self.assertIn("must be between 0.0 and 1.0", str(context.exception))
    
    def test_score_validation_non_numeric(self):
        """Test that non-numeric scores raise ValueError"""
        invalid_data = self.valid_metrics_data.copy()
        invalid_data['success_rate'] = 'high'
        
        with self.assertRaises(ValueError) as context:
            QualityMetrics(**invalid_data)
        
        self.assertIn("must be numeric", str(context.exception))
    
    def test_response_time_validation_negative(self):
        """Test that negative response time raises ValueError"""
        invalid_data = self.valid_metrics_data.copy()
        invalid_data['response_time_ms'] = -100.0
        
        with self.assertRaises(ValueError) as context:
            QualityMetrics(**invalid_data)
        
        self.assertIn("cannot be negative", str(context.exception))
    
    def test_response_time_validation_non_numeric(self):
        """Test that non-numeric response time raises ValueError"""
        invalid_data = self.valid_metrics_data.copy()
        invalid_data['response_time_ms'] = 'fast'
        
        with self.assertRaises(ValueError) as context:
            QualityMetrics(**invalid_data)
        
        self.assertIn("must be numeric", str(context.exception))
    
    def test_timestamp_validation_invalid(self):
        """Test that invalid timestamp raises ValueError"""
        invalid_data = self.valid_metrics_data.copy()
        invalid_data['timestamp'] = -1.0
        
        with self.assertRaises(ValueError) as context:
            QualityMetrics(**invalid_data)
        
        self.assertIn("must be positive", str(context.exception))
    
    def test_get_overall_quality_score(self):
        """Test overall quality score calculation"""
        metrics = QualityMetrics(**self.valid_metrics_data)
        score = metrics.get_overall_quality_score()
        
        # Should be a weighted average minus error penalty
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_json_serialization_roundtrip(self):
        """Test JSON serialization and deserialization"""
        metrics = QualityMetrics(**self.valid_metrics_data)
        
        # Serialize to JSON
        json_str = metrics.to_json()
        self.assertIsInstance(json_str, str)
        
        # Deserialize from JSON
        restored_metrics = QualityMetrics.from_json(json_str)
        
        self.assertEqual(restored_metrics.response_time_ms, metrics.response_time_ms)
        self.assertEqual(restored_metrics.accuracy_score, metrics.accuracy_score)


class TestReviewStatistics(unittest.TestCase):
    """Test cases for ReviewStatistics dataclass"""
    
    def setUp(self):
        """Set up test data"""
        self.valid_stats_data = {
            'agent_id': 'agent_bob',
            'total_reviews': 25,
            'average_ratings': {
                'communication_quality': 4.2,
                'response_time': 3.8,
                'accuracy': 4.5,
                'helpfulness': 4.1
            },
            'review_trend': TrendDirection.IMPROVING,
            'performance_percentile': 78.5,
            'last_updated': time.time(),
            'review_distribution': {
                '1-2': 2,
                '2-3': 3,
                '3-4': 8,
                '4-5': 12
            }
        }
    
    def test_valid_review_statistics_creation(self):
        """Test creating valid ReviewStatistics"""
        stats = ReviewStatistics(**self.valid_stats_data)
        
        self.assertEqual(stats.agent_id, 'agent_bob')
        self.assertEqual(stats.total_reviews, 25)
        self.assertEqual(stats.review_trend, TrendDirection.IMPROVING)
        self.assertEqual(stats.performance_percentile, 78.5)
    
    def test_total_reviews_validation_negative(self):
        """Test that negative total reviews raises ValueError"""
        invalid_data = self.valid_stats_data.copy()
        invalid_data['total_reviews'] = -5
        
        with self.assertRaises(ValueError) as context:
            ReviewStatistics(**invalid_data)
        
        self.assertIn("cannot be negative", str(context.exception))
    
    def test_total_reviews_validation_non_integer(self):
        """Test that non-integer total reviews raises ValueError"""
        invalid_data = self.valid_stats_data.copy()
        invalid_data['total_reviews'] = 25.5
        
        with self.assertRaises(ValueError) as context:
            ReviewStatistics(**invalid_data)
        
        self.assertIn("must be an integer", str(context.exception))
    
    def test_average_ratings_validation_invalid_criterion(self):
        """Test that invalid rating criteria raise ValueError"""
        invalid_data = self.valid_stats_data.copy()
        invalid_data['average_ratings'] = {'invalid_criterion': 4.0}
        
        with self.assertRaises(ValueError) as context:
            ReviewStatistics(**invalid_data)
        
        self.assertIn("Invalid rating criterion", str(context.exception))
    
    def test_average_ratings_validation_out_of_range(self):
        """Test that average ratings outside 1.0-5.0 range raise ValueError"""
        invalid_data = self.valid_stats_data.copy()
        invalid_data['average_ratings'] = {'communication_quality': 0.5}
        
        with self.assertRaises(ValueError) as context:
            ReviewStatistics(**invalid_data)
        
        self.assertIn("must be between 1.0 and 5.0", str(context.exception))
    
    def test_performance_percentile_validation_out_of_range(self):
        """Test that performance percentile outside 0.0-100.0 range raises ValueError"""
        invalid_data = self.valid_stats_data.copy()
        invalid_data['performance_percentile'] = 150.0
        
        with self.assertRaises(ValueError) as context:
            ReviewStatistics(**invalid_data)
        
        self.assertIn("must be between 0.0 and 100.0", str(context.exception))
    
    def test_get_overall_average_rating(self):
        """Test overall average rating calculation"""
        stats = ReviewStatistics(**self.valid_stats_data)
        expected_average = (4.2 + 3.8 + 4.5 + 4.1) / 4
        self.assertAlmostEqual(stats.get_overall_average_rating(), expected_average, places=2)
    
    def test_json_serialization_roundtrip(self):
        """Test JSON serialization and deserialization"""
        stats = ReviewStatistics(**self.valid_stats_data)
        
        # Serialize to JSON
        json_str = stats.to_json()
        self.assertIsInstance(json_str, str)
        
        # Deserialize from JSON
        restored_stats = ReviewStatistics.from_json(json_str)
        
        self.assertEqual(restored_stats.agent_id, stats.agent_id)
        self.assertEqual(restored_stats.total_reviews, stats.total_reviews)
        self.assertEqual(restored_stats.review_trend, stats.review_trend)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions"""
    
    def test_create_review_id(self):
        """Test review ID generation"""
        review_id = create_review_id()
        
        self.assertIsInstance(review_id, str)
        self.assertTrue(review_id.startswith('review_'))
        self.assertEqual(len(review_id), 19)  # 'review_' + 12 hex chars
        
        # Test uniqueness
        review_id2 = create_review_id()
        self.assertNotEqual(review_id, review_id2)
    
    def test_response_time_to_rating(self):
        """Test response time to rating conversion"""
        # Test excellent rating (≤100ms)
        self.assertEqual(_response_time_to_rating(50), 5.0)
        self.assertEqual(_response_time_to_rating(100), 5.0)
        
        # Test good rating (≤500ms)
        self.assertEqual(_response_time_to_rating(300), 4.0)
        self.assertEqual(_response_time_to_rating(500), 4.0)
        
        # Test average rating (≤1000ms)
        self.assertEqual(_response_time_to_rating(750), 3.0)
        self.assertEqual(_response_time_to_rating(1000), 3.0)
        
        # Test poor rating (≤2000ms)
        self.assertEqual(_response_time_to_rating(1500), 2.0)
        self.assertEqual(_response_time_to_rating(2000), 2.0)
        
        # Test very poor rating (>2000ms)
        self.assertEqual(_response_time_to_rating(3000), 1.0)
    
    def test_create_automatic_review(self):
        """Test automatic review creation from quality metrics"""
        quality_metrics = QualityMetrics(
            response_time_ms=300.0,
            accuracy_score=0.9,
            protocol_compliance_score=0.95,
            message_clarity_score=0.85,
            error_rate=0.05,
            success_rate=0.92,
            timestamp=time.time()
        )
        
        review = create_automatic_review(
            reviewer_id='system',
            reviewed_agent_id='agent_bob',
            interaction_id='interaction_123',
            quality_metrics=quality_metrics,
            interaction_timestamp=time.time() - 100,
            signature='auto_signature'
        )
        
        self.assertEqual(review.review_type, ReviewType.AUTOMATIC)
        self.assertEqual(review.reviewer_id, 'system')
        self.assertEqual(review.reviewed_agent_id, 'agent_bob')
        self.assertIsNone(review.text_feedback)
        self.assertTrue(review.metadata['auto_generated'])
        
        # Check rating conversions
        self.assertEqual(review.ratings['response_time'], 4.0)  # 300ms -> good
        self.assertAlmostEqual(review.ratings['accuracy'], 4.6, places=1)  # 0.9 -> 4.6
    
    def test_create_empty_review_statistics(self):
        """Test empty review statistics creation"""
        stats = create_empty_review_statistics('agent_alice')
        
        self.assertEqual(stats.agent_id, 'agent_alice')
        self.assertEqual(stats.total_reviews, 0)
        self.assertEqual(stats.review_trend, TrendDirection.INSUFFICIENT_DATA)
        self.assertEqual(stats.performance_percentile, 0.0)
        
        # Check all rating criteria are present with default neutral values
        expected_criteria = {'communication_quality', 'response_time', 'accuracy', 'helpfulness'}
        self.assertEqual(set(stats.average_ratings.keys()), expected_criteria)
        self.assertTrue(all(rating == 3.0 for rating in stats.average_ratings.values()))


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)