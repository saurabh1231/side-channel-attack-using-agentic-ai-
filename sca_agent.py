"""
Side-Channel Attack Detection Agent
Agentic AI implementation for autonomous security monitoring
"""

import asyncio
import json
import logging
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from enum import Enum
import time
import sqlite3
from pathlib import Path

from side_channel_cnn import SideChannelCNN, SCAConfig, SyntheticDataGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Security threat levels"""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityAlert:
    """Security alert data structure"""
    alert_id: str
    timestamp: float
    threat_level: ThreatLevel
    confidence: float
    predicted_key: int
    trace_id: str
    details: Dict


class SCADatabase:
    """Database for storing side-channel attack detection results"""
    
    def __init__(self, db_path: str = "sca_agent_memory.sqlite"):
        self.db_path = db_path
        self.conn = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database schema"""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS security_alerts (
                alert_id TEXT PRIMARY KEY,
                timestamp REAL,
                threat_level TEXT,
                confidence REAL,
                predicted_key INTEGER,
                trace_id TEXT,
                details TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_sessions (
                session_id TEXT PRIMARY KEY,
                start_time REAL,
                end_time REAL,
                traces_analyzed INTEGER,
                alerts_generated INTEGER,
                model_accuracy REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                accuracy REAL,
                key_rank REAL,
                inference_time REAL
            )
        """)
        
        self.conn.commit()
        logger.info(f"Database initialized at {self.db_path}")
    
    def store_alert(self, alert: SecurityAlert):
        """Store security alert"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO security_alerts VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.alert_id,
            alert.timestamp,
            alert.threat_level.value,
            alert.confidence,
            alert.predicted_key,
            alert.trace_id,
            json.dumps(alert.details)
        ))
        self.conn.commit()
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Retrieve recent security alerts"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM security_alerts 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
        
        alerts = []
        for row in cursor.fetchall():
            alerts.append({
                'alert_id': row[0],
                'timestamp': row[1],
                'threat_level': row[2],
                'confidence': row[3],
                'predicted_key': row[4],
                'trace_id': row[5],
                'details': json.loads(row[6])
            })
        return alerts
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


class SCAAgent:
    """Autonomous Side-Channel Attack Detection Agent"""
    
    def __init__(self, agent_id: str = "SCA-Agent-001"):
        self.agent_id = agent_id
        self.config = SCAConfig(
            trace_length=5000,
            num_classes=256,
            batch_size=32,
            epochs=30
        )
        self.model = SideChannelCNN(self.config)
        self.database = SCADatabase()
        self.is_trained = False
        self.monitoring_active = False
        
        logger.info(f"SCA Agent {agent_id} initialized")
    
    async def train_model(self, num_samples: int = 10000):
        """Train the CNN model asynchronously"""
        logger.info(f"Agent {self.agent_id}: Starting model training...")
        
        # Generate training data
        generator = SyntheticDataGenerator()
        traces, labels = generator.generate_aes_traces(
            num_traces=num_samples,
            trace_length=self.config.trace_length,
            snr=5.0
        )
        
        # Build model
        self.model.build_model()
        
        # Train in executor to avoid blocking
        loop = asyncio.get_event_loop()
        training_results = await loop.run_in_executor(
            None, 
            self.model.train, 
            traces, 
            labels
        )
        
        self.is_trained = True
        logger.info(f"Agent {self.agent_id}: Training complete - "
                   f"Accuracy: {training_results['final_accuracy']:.4f}")
        
        return training_results
    
    async def analyze_trace(self, trace: np.ndarray, trace_id: str) -> SecurityAlert:
        """Analyze a single power trace for side-channel leakage"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before analysis")
        
        start_time = time.time()
        
        # Predict key byte
        trace_reshaped = trace.reshape(1, -1)
        predictions = self.model.predict(trace_reshaped)[0]
        
        # Get top prediction
        predicted_key = int(np.argmax(predictions))
        confidence = float(predictions[predicted_key])
        
        inference_time = time.time() - start_time
        
        # Determine threat level based on confidence
        if confidence > 0.9:
            threat_level = ThreatLevel.CRITICAL
        elif confidence > 0.7:
            threat_level = ThreatLevel.HIGH
        elif confidence > 0.5:
            threat_level = ThreatLevel.MEDIUM
        elif confidence > 0.3:
            threat_level = ThreatLevel.LOW
        else:
            threat_level = ThreatLevel.SAFE
        
        # Create alert
        alert = SecurityAlert(
            alert_id=f"ALERT-{int(time.time() * 1000)}",
            timestamp=time.time(),
            threat_level=threat_level,
            confidence=confidence,
            predicted_key=predicted_key,
            trace_id=trace_id,
            details={
                'inference_time': inference_time,
                'top_5_predictions': [
                    {'key': int(k), 'prob': float(predictions[k])}
                    for k in np.argsort(predictions)[-5:][::-1]
                ]
            }
        )
        
        # Store alert if threat detected
        if threat_level != ThreatLevel.SAFE:
            self.database.store_alert(alert)
            logger.warning(f"Agent {self.agent_id}: {threat_level.value.upper()} threat detected - "
                          f"Key: {predicted_key}, Confidence: {confidence:.4f}")
        
        return alert
    
    async def monitor_stream(self, trace_generator, duration: int = 60):
        """Monitor continuous stream of power traces"""
        logger.info(f"Agent {self.agent_id}: Starting monitoring for {duration}s...")
        self.monitoring_active = True
        
        start_time = time.time()
        traces_analyzed = 0
        alerts_generated = 0
        
        try:
            while self.monitoring_active and (time.time() - start_time) < duration:
                # Get next trace
                trace = await trace_generator()
                trace_id = f"TRACE-{traces_analyzed}"
                
                # Analyze trace
                alert = await self.analyze_trace(trace, trace_id)
                traces_analyzed += 1
                
                if alert.threat_level != ThreatLevel.SAFE:
                    alerts_generated += 1
                
                # Small delay to simulate real-time monitoring
                await asyncio.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Agent {self.agent_id}: Monitoring error - {e}")
        
        finally:
            self.monitoring_active = False
            logger.info(f"Agent {self.agent_id}: Monitoring complete - "
                       f"Analyzed: {traces_analyzed}, Alerts: {alerts_generated}")
        
        return {
            'traces_analyzed': traces_analyzed,
            'alerts_generated': alerts_generated,
            'duration': time.time() - start_time
        }
    
    async def get_security_report(self) -> Dict:
        """Generate security analysis report"""
        recent_alerts = self.database.get_recent_alerts(limit=50)
        
        threat_distribution = {}
        for alert in recent_alerts:
            level = alert['threat_level']
            threat_distribution[level] = threat_distribution.get(level, 0) + 1
        
        report = {
            'agent_id': self.agent_id,
            'timestamp': time.time(),
            'total_alerts': len(recent_alerts),
            'threat_distribution': threat_distribution,
            'recent_alerts': recent_alerts[:10],
            'model_status': 'trained' if self.is_trained else 'untrained'
        }
        
        return report
    
    def shutdown(self):
        """Shutdown agent and cleanup resources"""
        self.monitoring_active = False
        self.database.close()
        logger.info(f"Agent {self.agent_id}: Shutdown complete")


async def demo_sca_agent():
    """Demonstration of SCA Agent capabilities"""
    
    # Initialize agent
    agent = SCAAgent(agent_id="SCA-Demo-Agent")
    
    # Train model
    logger.info("\n=== Phase 1: Model Training ===")
    await agent.train_model(num_samples=5000)
    
    # Create trace generator for monitoring
    async def trace_generator():
        """Generate synthetic traces for monitoring"""
        trace = np.random.normal(0, 1, agent.config.trace_length)
        # Add some leakage signal
        leakage_point = agent.config.trace_length // 2
        hw = np.random.randint(0, 9)
        trace[leakage_point-50:leakage_point+50] += hw * 2
        return trace
    
    # Monitor trace stream
    logger.info("\n=== Phase 2: Real-time Monitoring ===")
    monitoring_results = await agent.monitor_stream(trace_generator, duration=10)
    
    # Generate security report
    logger.info("\n=== Phase 3: Security Report ===")
    report = await agent.get_security_report()
    
    logger.info(f"\nMonitoring Results:")
    logger.info(f"  Traces Analyzed: {monitoring_results['traces_analyzed']}")
    logger.info(f"  Alerts Generated: {monitoring_results['alerts_generated']}")
    logger.info(f"  Duration: {monitoring_results['duration']:.2f}s")
    
    logger.info(f"\nSecurity Report:")
    logger.info(f"  Total Alerts: {report['total_alerts']}")
    logger.info(f"  Threat Distribution: {report['threat_distribution']}")
    
    # Save report
    with open('sca_security_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info("\nReport saved to 'sca_security_report.json'")
    
    # Shutdown
    agent.shutdown()


if __name__ == "__main__":
    asyncio.run(demo_sca_agent())
