import asyncio
from typing import Dict, Any
from loguru import logger
from ...utils.message_queue import AgentMessageQueue
from ...utils.monitoring import AgentMonitor
from ..data_collection_agent import DataCollectionAgent
from ...models.database import get_db
import time

class DataCollectionRunner:
    def __init__(self):
        self.agent = DataCollectionAgent()
        self.message_queue = AgentMessageQueue(
            agent_id=self.agent.agent_id,
            agent_type=self.agent.agent_type
        )
        self.monitor = AgentMonitor(self.agent.agent_id)
        self.running = False

    async def start(self):
        try:
            await self.message_queue.initialize()
            await self.monitor.connect()
            await self.message_queue.subscribe_to_input(self.process_message)
            self.running = True
            logger.info(f"Data Collection Agent {self.agent.agent_id} started")
            
            # Keep the agent running
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in Data Collection Agent: {e}")
            raise

    async def stop(self):
        self.running = False
        await self.message_queue.close()
        await self.monitor.close()
        logger.info(f"Data Collection Agent {self.agent.agent_id} stopped")

    async def process_message(self, message: Dict[str, Any]):
        start_time = time.time()
        success = False
        
        try:
            # Process the message using the agent
            response = await self.agent.process(message)
            
            # Publish the response
            await self.message_queue.publish_output(response)
            
            success = True
            logger.info(f"Successfully processed message: {message}")
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            # Handle error response
            error_response = await self.agent.handle_error(e)
            await self.message_queue.publish_output(error_response)
            
        finally:
            # Record metrics
            processing_time = time.time() - start_time
            await self.monitor.record_message_processed(processing_time, success)

async def main():
    runner = DataCollectionRunner()
    try:
        await runner.start()
    except KeyboardInterrupt:
        await runner.stop()

if __name__ == "__main__":
    asyncio.run(main())