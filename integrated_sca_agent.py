"""
Integrated Side-Channel Attack Detection Agent
Combines SCA detection with existing MCP agent framework
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Optional
import numpy as np

# Import existing agent framework
try:
    from base_agent import Agent, AgentCard, MCPMessageType
except ImportError:
    logging.warning("base_agent not available, using standalone mode")
    Agent = object
    AgentCard = dict
    MCPMessageType = None

# Import SCA components
from sca_agent import SCAAgent, ThreatLevel, SecurityAlert
from side_channel_cnn import SCAConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SCAServiceRequest:
    """Request for SCA analysis service"""
    request_id: str
    traces: list  # List of power traces
    requester_id: str
    priority: str = "normal"


@dataclass
class SCAServiceResponse:
    """Response from SCA analysis"""
    request_id: str
    alerts: list
    summary: Dict
    status: str


class IntegratedSCAAgent(Agent if Agent != object else object):
    """
    Integrated SCA Agent with MCP Protocol Support
    Provides side-channel attack detection as a service to other agents
    """
    
    def __init__(self, agent_id: str = "IntegratedSCA-001", 
                 mcp_server_url: Optional[str] = None):
        
        # Initialize base agent if available
        if Agent != object:
            super().__init__()
            self.agent_card = AgentCard(
                agent_id=agent_id,
                name="Side-Channel Attack Detection Agent",
                capabilities=[
                    "power_trace_analysis",
                    "key_recovery_attack",
                    "security_monitoring",
                    "threat_detection"
                ],
                description="Autonomous agent for detecting side-channel attacks using CNN"
            )
        
        self.agent_id = agent_id
        self.mcp_server_url = mcp_server_url
        
        # Initialize SCA agent
        self.sca_agent = SCAAgent(agent_id=f"SCA-{agent_id}")
        self.is_ready = False
        
        # Service statistics
        self.stats = {
            'requests_processed': 0,
            'traces_analyzed': 0,
            'threats_detected': 0,
            'uptime_start': None
        }
        
        logger.info(f"Integrated SCA Agent {agent_id} initialized")
    
    async def initialize(self):
        """Initialize and train the SCA model"""
        logger.info(f"Agent {self.agent_id}: Initializing SCA capabilities...")
        
        # Train model
        await self.sca_agent.train_model(num_samples=5000)
        
        self.is_ready = True
        self.stats['uptime_start'] = asyncio.get_event_loop().time()
        
        logger.info(f"Agent {self.agent_id}: Ready for service")
    
    async def handle_sca_request(self, request: SCAServiceRequest) -> SCAServiceResponse:
        """
        Handle SCA analysis request from another agent
        
        Args:
            request: SCA service request with traces to analyze
        
        Returns:
            SCA service response with alerts and summary
        """
        if not self.is_ready:
            return SCAServiceResponse(
                request_id=request.request_id,
                alerts=[],
                summary={'error': 'Agent not ready'},
                status='error'
            )
        
        logger.info(f"Processing request {request.request_id} from {request.requester_id}")
        
        alerts = []
        threat_counts = {level.value: 0 for level in ThreatLevel}
        
        # Analyze each trace
        for i, trace_data in enumerate(request.traces):
            trace = np.array(trace_data)
            trace_id = f"{request.request_id}-TRACE-{i}"
            
            # Analyze trace
            alert = await self.sca_agent.analyze_trace(trace, trace_id)
            
            # Convert alert to dict
            alert_dict = {
                'alert_id': alert.alert_id,
                'timestamp': alert.timestamp,
                'threat_level': alert.threat_level.value,
                'confidence': alert.confidence,
                'predicted_key': alert.predicted_key,
                'trace_id': alert.trace_id,
                'details': alert.details
            }
            alerts.append(alert_dict)
            
            # Update statistics
            threat_counts[alert.threat_level.value] += 1
            if alert.threat_level != ThreatLevel.SAFE:
                self.stats['threats_detected'] += 1
        
        # Update statistics
        self.stats['requests_processed'] += 1
        self.stats['traces_analyzed'] += len(request.traces)
        
        # Create summary
        summary = {
            'total_traces': len(request.traces),
            'threat_distribution': threat_counts,
            'high_risk_traces': threat_counts['high'] + threat_counts['critical'],
            'average_confidence': np.mean([a['confidence'] for a in alerts])
        }
        
        response = SCAServiceResponse(
            request_id=request.request_id,
            alerts=alerts,
            summary=summary,
            status='success'
        )
        
        logger.info(f"Request {request.request_id} completed - "
                   f"Threats: {summary['high_risk_traces']}/{summary['total_traces']}")
        
        return response
    
    async def handle_mcp_message(self, message: Dict) -> Dict:
        """
        Handle MCP protocol messages
        
        Args:
            message: MCP message from another agent
        
        Returns:
            Response message
        """
        msg_type = message.get('type')
        
        if msg_type == 'SCA_REQUEST':
            # Parse request
            request = SCAServiceRequest(
                request_id=message['request_id'],
                traces=message['traces'],
                requester_id=message['sender_id'],
                priority=message.get('priority', 'normal')
            )
            
            # Process request
            response = await self.handle_sca_request(request)
            
            # Return MCP response
            return {
                'type': 'SCA_RESPONSE',
                'request_id': response.request_id,
                'alerts': response.alerts,
                'summary': response.summary,
                'status': response.status,
                'sender_id': self.agent_id
            }
        
        elif msg_type == 'STATUS_REQUEST':
            return await self.get_status()
        
        elif msg_type == 'REPORT_REQUEST':
            return await self.get_security_report()
        
        else:
            return {
                'type': 'ERROR',
                'error': f'Unknown message type: {msg_type}',
                'sender_id': self.agent_id
            }
    
    async def get_status(self) -> Dict:
        """Get agent status"""
        uptime = None
        if self.stats['uptime_start']:
            uptime = asyncio.get_event_loop().time() - self.stats['uptime_start']
        
        return {
            'type': 'STATUS_RESPONSE',
            'agent_id': self.agent_id,
            'status': 'ready' if self.is_ready else 'initializing',
            'statistics': {
                **self.stats,
                'uptime_seconds': uptime
            },
            'capabilities': [
                'power_trace_analysis',
                'key_recovery_attack',
                'security_monitoring',
                'threat_detection'
            ]
        }
    
    async def get_security_report(self) -> Dict:
        """Get comprehensive security report"""
        report = await self.sca_agent.get_security_report()
        
        return {
            'type': 'REPORT_RESPONSE',
            'agent_id': self.agent_id,
            'report': report,
            'statistics': self.stats
        }
    
    async def continuous_monitoring(self, trace_source, duration: int = 300):
        """
        Continuous monitoring mode
        
        Args:
            trace_source: Async generator providing traces
            duration: Monitoring duration in seconds
        """
        logger.info(f"Agent {self.agent_id}: Starting continuous monitoring for {duration}s")
        
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < duration:
            try:
                # Get next trace
                trace = await trace_source()
                
                # Analyze
                trace_id = f"MONITOR-{int(asyncio.get_event_loop().time() * 1000)}"
                alert = await self.sca_agent.analyze_trace(trace, trace_id)
                
                # Log high-priority threats
                if alert.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                    logger.warning(f"HIGH PRIORITY ALERT: {alert.threat_level.value} - "
                                 f"Confidence: {alert.confidence:.4f}")
                
                self.stats['traces_analyzed'] += 1
                
                # Small delay
                await asyncio.sleep(0.1)
            
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(1)
        
        logger.info(f"Agent {self.agent_id}: Monitoring complete")
    
    def shutdown(self):
        """Shutdown agent"""
        logger.info(f"Agent {self.agent_id}: Shutting down...")
        self.sca_agent.shutdown()
        logger.info(f"Agent {self.agent_id}: Shutdown complete")


async def demo_integrated_agent():
    """Demonstration of integrated SCA agent"""
    
    logger.info("=== Integrated SCA Agent Demo ===\n")
    
    # Initialize agent
    agent = IntegratedSCAAgent(agent_id="IntegratedSCA-Demo")
    await agent.initialize()
    
    # Simulate request from another agent (e.g., Alice or Bob)
    logger.info("\n--- Simulating Service Request from Alice ---")
    
    # Generate sample traces
    sample_traces = [
        np.random.normal(0, 1, 5000).tolist()
        for _ in range(10)
    ]
    
    # Add leakage to some traces
    for i in [2, 5, 7]:
        trace = np.array(sample_traces[i])
        leakage_point = 2500
        trace[leakage_point-50:leakage_point+50] += 8  # Strong leakage
        sample_traces[i] = trace.tolist()
    
    # Create request
    request = SCAServiceRequest(
        request_id="REQ-ALICE-001",
        traces=sample_traces,
        requester_id="Alice-Research-Agent",
        priority="high"
    )
    
    # Process request
    response = await agent.handle_sca_request(request)
    
    # Display results
    logger.info(f"\nResponse Status: {response.status}")
    logger.info(f"Traces Analyzed: {response.summary['total_traces']}")
    logger.info(f"High-Risk Traces: {response.summary['high_risk_traces']}")
    logger.info(f"Threat Distribution: {response.summary['threat_distribution']}")
    
    # Get agent status
    logger.info("\n--- Agent Status ---")
    status = await agent.get_status()
    logger.info(f"Status: {status['status']}")
    logger.info(f"Requests Processed: {status['statistics']['requests_processed']}")
    logger.info(f"Traces Analyzed: {status['statistics']['traces_analyzed']}")
    logger.info(f"Threats Detected: {status['statistics']['threats_detected']}")
    
    # Save results
    with open('integrated_sca_demo_results.json', 'w') as f:
        json.dump({
            'response': {
                'request_id': response.request_id,
                'summary': response.summary,
                'alert_count': len(response.alerts)
            },
            'status': status
        }, f, indent=2)
    
    logger.info("\nResults saved to 'integrated_sca_demo_results.json'")
    
    # Shutdown
    agent.shutdown()


if __name__ == "__main__":
    asyncio.run(demo_integrated_agent())
