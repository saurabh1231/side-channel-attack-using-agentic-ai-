import asyncio
import logging
import time
from alice import Alice
from bob import Bob
from mcp_server import MCPServer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def run_mcp_server():
    """Run the MCP WebSocket server"""
    server = MCPServer()
    start_server = server.start_server()
    await start_server

async def run_agents():
    """Run Alice and Bob agents"""
    # Wait for server to start
    await asyncio.sleep(1)
    
    # Create agents
    alice = Alice()
    bob = Bob()
    
    # Display agent cards
    print("=== Agent Cards ===\n")
    alice.display_agent_card()
    bob.display_agent_card()
    
    # Connect to MCP server
    print("=== MCP Connection Phase ===")
    alice_connected = await alice.connect_to_mcp()
    bob_connected = await bob.connect_to_mcp()
    
    if not (alice_connected and bob_connected):
        logger.error("MCP connection failed. Terminating.")
        return
    
    # Send heartbeats
    await alice.send_heartbeat()
    await bob.send_heartbeat()
    
    # Wait for connections to stabilize
    await asyncio.sleep(2)
    
    print("\n=== Secure Communication via MCP WebSockets ===")
    
    # Alice sends network report
    metadata = {
        "latency": 45.2,
        "bandwidth": 100.5,
        "jitter": 2.1,
        "budget": 150.0
    }
    
    # Generate shared secret key for demo (in production, this would be exchanged securely)
    secret_key = Alice.create_shared_key_from_string('demo_secret_key_for_alice_and_bob')
    
    await alice.send_network_report(bob.agent_id, metadata, secret_key)
    
    # Wait for processing
    await asyncio.sleep(3)
    
    # Send another report with different values
    metadata2 = {
        "latency": 52.8,
        "bandwidth": 87.3,
        "jitter": 3.5,
        "budget": 200.0
    }
    
    logger.info("\n=== Sending Second Report ===")
    await alice.send_network_report(bob.agent_id, metadata2, secret_key)
    
    # Wait for processing
    await asyncio.sleep(3)
    
    # Cleanup
    print("\n=== Cleanup Phase ===")
    await alice.disconnect_from_mcp()
    await bob.disconnect_from_mcp()

async def main():
    """Main function to run MCP server and agents"""
    print("=== MCP WebSocket Communication System ===\n")
    
    # Start server and agents concurrently
    await asyncio.gather(
        run_mcp_server(),
        run_agents()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("System shutdown requested")
    except Exception as e:
        logger.error(f"System error: {e}")