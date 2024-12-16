import asyncio
from typing import Dict, Any, List
from loguru import logger
from ...utils.message_queue import AgentMessageQueue
from ...utils.monitoring import AgentMonitor
from ..expert_agents import RelationshipPsychologistAgent, BehavioralPsychologistAgent
import time

class ExpertAgentRunner:
    def __init__(self, agent_class, agent_id):
        self.agent = agent_class()
        self.agent_id = agent_id
        self.message_queue = AgentMessageQueue(
            agent_id=self.agent_id,
            agent_type=self.agent.agent_type
        )
        self.monitor = AgentMonitor(self.agent_id)
        self.running = False

    async def start(self):
        try:
            await self.message_queue.initialize()
            await self.monitor.connect()
            await self.message_queue.subscribe_to_input(self.process_message)
            self.running = True
            logger.info(f"Expert Agent {self.agent_id} started")
            
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in Expert Agent {self.agent_id}: {e}")
            raise

    async def stop(self):
        self.running = False
        await self.message_queue.close()
        await self.monitor.close()
        logger.info(f"Expert Agent {self.agent_id} stopped")

    async def process_message(self, message: Dict[str, Any]):
        start_time = time.time()
        success = False
        
        try:
            response = await self.agent.process(message)
            await self.message_queue.publish_output(response)
            success = True
            logger.info(f"Successfully processed message in {self.agent_id}: {message}")
            
        except Exception as e:
            logger.error(f"Error processing message in {self.agent_id}: {e}")
            error_response = await self.agent.handle_error(e)
            await self.message_queue.publish_output(error_response)
            
        finally:
            processing_time = time.time() - start_time
            await self.monitor.record_message_processed(processing_time, success)

class ExpertAgentsRunner:
    def __init__(self):
        self.runners: List[ExpertAgentRunner] = [
            ExpertAgentRunner(RelationshipPsychologistAgent, "relationship_psychologist"),
            ExpertAgentRunner(BehavioralPsychologistAgent, "behavioral_psychologist")
        ]

    async def start(self):
        try:
            # Start all runners
            await asyncio.gather(*[runner.start() for runner in self.runners])
            
        except Exception as e:
            logger.error(f"Error starting Expert Agents: {e}")
            # Stop any runners that may have started
            await self.stop()
            raise

    async def stop(self):
        # Stop all runners
        await asyncio.gather(*[runner.stop() for runner in self.runners])
        logger.info("All Expert Agents stopped")

async def main():
    runner = ExpertAgentsRunner()
    try:
        await runner.start()
    except KeyboardInterrupt:
        await runner.stop()

if __name__ == "__main__":
    asyncio.run(main())