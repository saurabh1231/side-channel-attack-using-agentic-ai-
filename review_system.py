"""
Review System Core Data Structures and Enums

This module contains the core data structures and enums for the AI Agent Review System,
including AgentReview, QualityMetrics, ReviewStatistics dataclasses and related enums.
"""

import json
import time
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any


class ReviewType(Enum):
    """Type of review - manual or automatic"""
    MANUAL = "manual"
    AUTOMATIC = "automatic"


class TrendDirection(Enum):
    """Direction of performance trend"""
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    INSUFFICIENT_DATA = "insufficient_data"


class NotificationStatus(Enum):
    """Status of review notifications"""
    PENDING = "pending"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"


@dataclass
class AgentReview:
    """
    Core review data structure for agent performance reviews
    
    Attributes:
        review_id: Unique identifier for the review
        reviewer_id: ID of the agent submitting the review
        reviewed_agent_id: ID of the agent being reviewed
        interaction_id: ID of the interaction being reviewed
        review_type: Type of review (manual or automatic)
        ratings: Dictionary of rating criteria and scores
        text_feedback: Optional text feedback
        review_timestamp: When the review was submitted
        interaction_timestamp: When the interaction occurred
        review_signature: Cryptographic signature for authenticity
        metadata: Additional metadata for the review
    """
    review_id: str
    reviewer_id: str
    reviewed_agent_id: str
    interaction_id: str
    review_type: ReviewType
    ratings: Dict[str, float]  # communication_quality, response_time, accuracy, helpfulness
    text_feedback: Optional[str]
    review_timestamp: float
    interaction_timestamp: float
    review_signature: str
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate review data after initialization"""
        self._validate_ratings()
        self._validate_timestamps()
    
    def _validate_ratings(self):
        """Validate that ratings are within acceptable ranges"""
        valid_criteria = {'communication_quality', 'response_time', 'accuracy', 'helpfulness'}
        
        if not self.ratings:
            raise ValueError("Ratings dictionary cannot be empty")
        
        for criterion, rating in self.ratings.items():
            if criterion not in valid_criteria:
                raise ValueError(f"Invalid rating criterion: {criterion}")
            
            if not isinstance(rating, (int, float)):
                raise ValueError(f"Rating for {criterion} must be numeric")
            
            if not (1.0 <= rating <= 5.0):
                raise ValueError(f"Rating for {criterion} must be between 1.0 and 5.0")
    
    def _validate_timestamps(self):
        """Validate timestamp values"""
        if self.review_timestamp <= 0:
            raise ValueError("Review timestamp must be positive")
        
        if self.interaction_timestamp <= 0:
            raise ValueError("Interaction timestamp must be positive")
        
        if self.review_timestamp < self.interaction_timestamp:
            raise ValueError("Review timestamp cannot be before interaction timestamp")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert review to dictionary for serialization"""
        data = asdict(self)
        data['review_type'] = self.review_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentReview':
        """Create AgentReview from dictionary"""
        data = data.copy()
        data['review_type'] = ReviewType(data['review_type'])
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert review to JSON string"""
        return json.dumps(self.to_dict(), sort_keys=True)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AgentReview':
        """Create AgentReview from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_average_rating(self) -> float:
        """Calculate average rating across all criteria"""
        if not self.ratings:
            return 0.0
        return sum(self.ratings.values()) / len(self.ratings)


@dataclass
class QualityMetrics:
    """
    Quality metrics for automatic assessment of agent interactions
    
    Attributes:
        response_time_ms: Response time in milliseconds
        accuracy_score: Accuracy score (0.0 - 1.0)
        protocol_compliance_score: Protocol compliance score (0.0 - 1.0)
        message_clarity_score: Message clarity score (0.0 - 1.0)
        error_rate: Error rate (0.0 - 1.0)
        success_rate: Success rate (0.0 - 1.0)
        timestamp: When metrics were calculated
    """
    response_time_ms: float
    accuracy_score: float
    protocol_compliance_score: float
    message_clarity_score: float
    error_rate: float
    success_rate: float
    timestamp: float
    
    def __post_init__(self):
        """Validate quality metrics after initialization"""
        self._validate_scores()
        self._validate_response_time()
        self._validate_timestamp()
    
    def _validate_scores(self):
        """Validate that all scores are within 0.0 - 1.0 range"""
        score_fields = [
            'accuracy_score', 'protocol_compliance_score', 
            'message_clarity_score', 'error_rate', 'success_rate'
        ]
        
        for field in score_fields:
            value = getattr(self, field)
            if not isinstance(value, (int, float)):
                raise ValueError(f"{field} must be numeric")
            
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{field} must be between 0.0 and 1.0")
    
    def _validate_response_time(self):
        """Validate response time"""
        if not isinstance(self.response_time_ms, (int, float)):
            raise ValueError("Response time must be numeric")
        
        if self.response_time_ms < 0:
            raise ValueError("Response time cannot be negative")
    
    def _validate_timestamp(self):
        """Validate timestamp"""
        if self.timestamp <= 0:
            raise ValueError("Timestamp must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QualityMetrics':
        """Create QualityMetrics from dictionary"""
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert metrics to JSON string"""
        return json.dumps(self.to_dict(), sort_keys=True)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'QualityMetrics':
        """Create QualityMetrics from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_overall_quality_score(self) -> float:
        """Calculate overall quality score from all metrics"""
        # Weighted average of quality metrics
        weights = {
            'accuracy_score': 0.3,
            'protocol_compliance_score': 0.2,
            'message_clarity_score': 0.2,
            'success_rate': 0.3
        }
        
        weighted_sum = (
            self.accuracy_score * weights['accuracy_score'] +
            self.protocol_compliance_score * weights['protocol_compliance_score'] +
            self.message_clarity_score * weights['message_clarity_score'] +
            self.success_rate * weights['success_rate']
        )
        
        # Penalize high error rates
        error_penalty = self.error_rate * 0.1
        
        return max(0.0, weighted_sum - error_penalty)


@dataclass
class ReviewStatistics:
    """
    Aggregated review statistics for an agent
    
    Attributes:
        agent_id: ID of the agent these statistics are for
        total_reviews: Total number of reviews received
        average_ratings: Average ratings for each criterion
        review_trend: Overall trend direction
        performance_percentile: Performance percentile (0.0 - 100.0)
        last_updated: When statistics were last calculated
        review_distribution: Distribution of ratings by range
    """
    agent_id: str
    total_reviews: int
    average_ratings: Dict[str, float]
    review_trend: TrendDirection
    performance_percentile: float
    last_updated: float
    review_distribution: Dict[str, int]  # rating ranges like "1-2", "2-3", etc.
    
    def __post_init__(self):
        """Validate review statistics after initialization"""
        self._validate_total_reviews()
        self._validate_average_ratings()
        self._validate_performance_percentile()
        self._validate_timestamp()
    
    def _validate_total_reviews(self):
        """Validate total reviews count"""
        if not isinstance(self.total_reviews, int):
            raise ValueError("Total reviews must be an integer")
        
        if self.total_reviews < 0:
            raise ValueError("Total reviews cannot be negative")
    
    def _validate_average_ratings(self):
        """Validate average ratings"""
        if not isinstance(self.average_ratings, dict):
            raise ValueError("Average ratings must be a dictionary")
        
        valid_criteria = {'communication_quality', 'response_time', 'accuracy', 'helpfulness'}
        
        for criterion, rating in self.average_ratings.items():
            if criterion not in valid_criteria:
                raise ValueError(f"Invalid rating criterion: {criterion}")
            
            if not isinstance(rating, (int, float)):
                raise ValueError(f"Average rating for {criterion} must be numeric")
            
            if not (1.0 <= rating <= 5.0):
                raise ValueError(f"Average rating for {criterion} must be between 1.0 and 5.0")
    
    def _validate_performance_percentile(self):
        """Validate performance percentile"""
        if not isinstance(self.performance_percentile, (int, float)):
            raise ValueError("Performance percentile must be numeric")
        
        if not (0.0 <= self.performance_percentile <= 100.0):
            raise ValueError("Performance percentile must be between 0.0 and 100.0")
    
    def _validate_timestamp(self):
        """Validate timestamp"""
        if self.last_updated <= 0:
            raise ValueError("Last updated timestamp must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary for serialization"""
        data = asdict(self)
        data['review_trend'] = self.review_trend.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReviewStatistics':
        """Create ReviewStatistics from dictionary"""
        data = data.copy()
        data['review_trend'] = TrendDirection(data['review_trend'])
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert statistics to JSON string"""
        return json.dumps(self.to_dict(), sort_keys=True)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ReviewStatistics':
        """Create ReviewStatistics from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_overall_average_rating(self) -> float:
        """Calculate overall average rating across all criteria"""
        if not self.average_ratings:
            return 0.0
        return sum(self.average_ratings.values()) / len(self.average_ratings)


# Utility functions for creating review data structures

def create_review_id() -> str:
    """Generate a unique review ID"""
    return f"review_{uuid.uuid4().hex[:12]}"


def create_automatic_review(
    reviewer_id: str,
    reviewed_agent_id: str,
    interaction_id: str,
    quality_metrics: QualityMetrics,
    interaction_timestamp: float,
    signature: str
) -> AgentReview:
    """
    Create an automatic review from quality metrics
    
    Args:
        reviewer_id: ID of the reviewing agent/system
        reviewed_agent_id: ID of the agent being reviewed
        interaction_id: ID of the interaction
        quality_metrics: Quality metrics for the interaction
        interaction_timestamp: When the interaction occurred
        signature: Cryptographic signature
    
    Returns:
        AgentReview: Automatically generated review
    """
    # Convert quality metrics to ratings (0.0-1.0 to 1.0-5.0 scale)
    ratings = {
        'communication_quality': 1.0 + (quality_metrics.message_clarity_score * 4.0),
        'response_time': _response_time_to_rating(quality_metrics.response_time_ms),
        'accuracy': 1.0 + (quality_metrics.accuracy_score * 4.0),
        'helpfulness': 1.0 + (quality_metrics.success_rate * 4.0)
    }
    
    return AgentReview(
        review_id=create_review_id(),
        reviewer_id=reviewer_id,
        reviewed_agent_id=reviewed_agent_id,
        interaction_id=interaction_id,
        review_type=ReviewType.AUTOMATIC,
        ratings=ratings,
        text_feedback=None,
        review_timestamp=time.time(),
        interaction_timestamp=interaction_timestamp,
        review_signature=signature,
        metadata={
            'quality_metrics': quality_metrics.to_dict(),
            'auto_generated': True
        }
    )


def _response_time_to_rating(response_time_ms: float) -> float:
    """
    Convert response time to rating scale (1.0-5.0)
    
    Args:
        response_time_ms: Response time in milliseconds
    
    Returns:
        float: Rating between 1.0 and 5.0
    """
    # Rating scale based on response time thresholds
    if response_time_ms <= 100:
        return 5.0  # Excellent
    elif response_time_ms <= 500:
        return 4.0  # Good
    elif response_time_ms <= 1000:
        return 3.0  # Average
    elif response_time_ms <= 2000:
        return 2.0  # Poor
    else:
        return 1.0  # Very poor


def create_empty_review_statistics(agent_id: str) -> ReviewStatistics:
    """
    Create empty review statistics for a new agent
    
    Args:
        agent_id: ID of the agent
    
    Returns:
        ReviewStatistics: Empty statistics structure
    """
    return ReviewStatistics(
        agent_id=agent_id,
        total_reviews=0,
        average_ratings={
            'communication_quality': 3.0,  # Default neutral rating
            'response_time': 3.0,
            'accuracy': 3.0,
            'helpfulness': 3.0
        },
        review_trend=TrendDirection.INSUFFICIENT_DATA,
        performance_percentile=0.0,
        last_updated=time.time(),
        review_distribution={
            '1-2': 0,
            '2-3': 0,
            '3-4': 0,
            '4-5': 0
        }
    )