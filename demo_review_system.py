#!/usr/bin/env python3
"""
Demonstration script for the Review System Core Data Structures

This script demonstrates the usage of AgentReview, QualityMetrics, and ReviewStatistics
dataclasses along with the utility functions.
"""

import time
from review_system import (
    AgentReview, QualityMetrics, ReviewStatistics,
    ReviewType, TrendDirection, NotificationStatus,
    create_review_id, create_automatic_review, create_empty_review_statistics
)
from base_agent import MCPMessageType


def demonstrate_review_system():
    """Demonstrate the review system functionality"""
    
    print("=" * 60)
    print("AI Agent Review System - Core Data Structures Demo")
    print("=" * 60)
    
    # 1. Demonstrate Enums
    print("\n1. Review System Enums:")
    print(f"   Review Types: {[rt.value for rt in ReviewType]}")
    print(f"   Trend Directions: {[td.value for td in TrendDirection]}")
    print(f"   Notification Status: {[ns.value for ns in NotificationStatus]}")
    print(f"   New MCP Message Types: {[msg.value for msg in MCPMessageType if 'review' in msg.value]}")
    
    # 2. Create Quality Metrics
    print("\n2. Creating Quality Metrics:")
    quality_metrics = QualityMetrics(
        response_time_ms=150.0,
        accuracy_score=0.92,
        protocol_compliance_score=0.98,
        message_clarity_score=0.88,
        error_rate=0.03,
        success_rate=0.95,
        timestamp=time.time()
    )
    
    print(f"   Response Time: {quality_metrics.response_time_ms}ms")
    print(f"   Accuracy Score: {quality_metrics.accuracy_score}")
    print(f"   Overall Quality Score: {quality_metrics.get_overall_quality_score():.3f}")
    
    # 3. Create Manual Review
    print("\n3. Creating Manual Agent Review:")
    manual_review = AgentReview(
        review_id=create_review_id(),
        reviewer_id="agent_alice",
        reviewed_agent_id="agent_bob",
        interaction_id="interaction_789",
        review_type=ReviewType.MANUAL,
        ratings={
            'communication_quality': 4.5,
            'response_time': 4.2,
            'accuracy': 4.8,
            'helpfulness': 4.3
        },
        text_feedback="Excellent response quality and very helpful information provided.",
        review_timestamp=time.time(),
        interaction_timestamp=time.time() - 300,
        review_signature="manual_signature_abc123",
        metadata={"interaction_type": "research_query", "domain": "cryptography"}
    )
    
    print(f"   Review ID: {manual_review.review_id}")
    print(f"   Reviewer: {manual_review.reviewer_id}")
    print(f"   Reviewed Agent: {manual_review.reviewed_agent_id}")
    print(f"   Review Type: {manual_review.review_type.value}")
    print(f"   Average Rating: {manual_review.get_average_rating():.2f}")
    print(f"   Feedback: {manual_review.text_feedback}")
    
    # 4. Create Automatic Review from Quality Metrics
    print("\n4. Creating Automatic Review from Quality Metrics:")
    auto_review = create_automatic_review(
        reviewer_id="system_assessor",
        reviewed_agent_id="agent_bob",
        interaction_id="interaction_456",
        quality_metrics=quality_metrics,
        interaction_timestamp=time.time() - 200,
        signature="auto_signature_xyz789"
    )
    
    print(f"   Review ID: {auto_review.review_id}")
    print(f"   Review Type: {auto_review.review_type.value}")
    print(f"   Auto-generated Ratings:")
    for criterion, rating in auto_review.ratings.items():
        print(f"     {criterion}: {rating:.2f}")
    print(f"   Average Rating: {auto_review.get_average_rating():.2f}")
    print(f"   Auto-generated: {auto_review.metadata.get('auto_generated', False)}")
    
    # 5. Create Review Statistics
    print("\n5. Creating Review Statistics:")
    stats = ReviewStatistics(
        agent_id="agent_bob",
        total_reviews=15,
        average_ratings={
            'communication_quality': 4.3,
            'response_time': 4.1,
            'accuracy': 4.6,
            'helpfulness': 4.2
        },
        review_trend=TrendDirection.IMPROVING,
        performance_percentile=82.5,
        last_updated=time.time(),
        review_distribution={
            '1-2': 1,
            '2-3': 2,
            '3-4': 5,
            '4-5': 7
        }
    )
    
    print(f"   Agent ID: {stats.agent_id}")
    print(f"   Total Reviews: {stats.total_reviews}")
    print(f"   Overall Average: {stats.get_overall_average_rating():.2f}")
    print(f"   Performance Trend: {stats.review_trend.value}")
    print(f"   Performance Percentile: {stats.performance_percentile}%")
    print(f"   Rating Distribution: {stats.review_distribution}")
    
    # 6. Demonstrate Serialization
    print("\n6. JSON Serialization Demo:")
    
    # Serialize manual review to JSON
    manual_json = manual_review.to_json()
    print(f"   Manual Review JSON length: {len(manual_json)} characters")
    
    # Deserialize and verify
    restored_review = AgentReview.from_json(manual_json)
    print(f"   Restored Review ID: {restored_review.review_id}")
    print(f"   Serialization successful: {restored_review.review_id == manual_review.review_id}")
    
    # 7. Create Empty Statistics for New Agent
    print("\n7. Empty Statistics for New Agent:")
    empty_stats = create_empty_review_statistics("agent_charlie")
    print(f"   New Agent ID: {empty_stats.agent_id}")
    print(f"   Total Reviews: {empty_stats.total_reviews}")
    print(f"   Default Ratings: {empty_stats.average_ratings}")
    print(f"   Trend: {empty_stats.review_trend.value}")
    
    # 8. Validation Demo
    print("\n8. Data Validation Demo:")
    try:
        # This should fail - invalid rating range
        invalid_review = AgentReview(
            review_id="test_review",
            reviewer_id="test_reviewer",
            reviewed_agent_id="test_agent",
            interaction_id="test_interaction",
            review_type=ReviewType.MANUAL,
            ratings={'communication_quality': 6.0},  # Invalid: > 5.0
            text_feedback=None,
            review_timestamp=time.time(),
            interaction_timestamp=time.time() - 100,
            review_signature="test_sig",
            metadata={}
        )
    except ValueError as e:
        print(f"   ✓ Validation caught invalid rating: {e}")
    
    try:
        # This should fail - negative response time
        invalid_metrics = QualityMetrics(
            response_time_ms=-100.0,  # Invalid: negative
            accuracy_score=0.9,
            protocol_compliance_score=0.95,
            message_clarity_score=0.85,
            error_rate=0.05,
            success_rate=0.92,
            timestamp=time.time()
        )
    except ValueError as e:
        print(f"   ✓ Validation caught negative response time: {e}")
    
    print("\n" + "=" * 60)
    print("Review System Demo Complete!")
    print("All data structures created and validated successfully.")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_review_system()