import aio_pika
import json
from typing import Any, Dict, Callable, Awaitable
from loguru import logger
import os

class MessageQueue:
    def __init__(self):
        self.connection = None
        self.channel = None
        self.exchange = None
        self.queues = {}
        self.consumers = {}
        self.rabbitmq_url = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost/")

    async def connect(self):
        try:
            self.connection = await aio_pika.connect_robust(self.rabbitmq_url)
            self.channel = await self.connection.channel()
            self.exchange = await self.channel.declare_exchange(
                "relationship_assessment",
                aio_pika.ExchangeType.TOPIC
            )
            logger.info("Successfully connected to RabbitMQ")
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    async def close(self):
        try:
            if self.connection:
                await self.connection.close()
                logger.info("Closed RabbitMQ connection")
        except Exception as e:
            logger.error(f"Error closing RabbitMQ connection: {e}")

    async def declare_queue(self, queue_name: str, routing_key: str):
        try:
            queue = await self.channel.declare_queue(queue_name, durable=True)
            await queue.bind(self.exchange, routing_key)
            self.queues[queue_name] = queue
            logger.info(f"Declared queue: {queue_name} with routing key: {routing_key}")
            return queue
        except Exception as e:
            logger.error(f"Failed to declare queue {queue_name}: {e}")
            raise

    async def publish(self, routing_key: str, message: Dict[str, Any]):
        try:
            message_body = json.dumps(message).encode()
            await self.exchange.publish(
                aio_pika.Message(
                    body=message_body,
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                ),
                routing_key=routing_key
            )
            logger.debug(f"Published message to {routing_key}: {message}")
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            raise

    async def subscribe(
        self,
        queue_name: str,
        callback: Callable[[Dict[str, Any]], Awaitable[None]]
    ):
        try:
            queue = self.queues.get(queue_name)
            if not queue:
                raise ValueError(f"Queue {queue_name} not declared")

            async def process_message(message: aio_pika.IncomingMessage):
                async with message.process():
                    try:
                        body = json.loads(message.body.decode())
                        await callback(body)
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        # Depending on the error, you might want to:
                        # - Reject the message
                        # - Send to a dead letter queue
                        # - Retry processing
                        await message.reject(requeue=False)

            self.consumers[queue_name] = await queue.consume(process_message)
            logger.info(f"Subscribed to queue: {queue_name}")
        except Exception as e:
            logger.error(f"Failed to subscribe to queue {queue_name}: {e}")
            raise

    async def unsubscribe(self, queue_name: str):
        try:
            consumer = self.consumers.get(queue_name)
            if consumer:
                await consumer.cancel()
                del self.consumers[queue_name]
                logger.info(f"Unsubscribed from queue: {queue_name}")
        except Exception as e:
            logger.error(f"Failed to unsubscribe from queue {queue_name}: {e}")
            raise

class AgentMessageQueue:
    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.mq = MessageQueue()
        self.input_queue = f"{agent_type}.{agent_id}.input"
        self.output_queue = f"{agent_type}.{agent_id}.output"

    async def initialize(self):
        await self.mq.connect()
        await self.mq.declare_queue(self.input_queue, f"{self.agent_type}.{self.agent_id}.#")
        await self.mq.declare_queue(self.output_queue, f"output.{self.agent_type}.{self.agent_id}")

    async def subscribe_to_input(self, callback: Callable[[Dict[str, Any]], Awaitable[None]]):
        await self.mq.subscribe(self.input_queue, callback)

    async def publish_output(self, message: Dict[str, Any]):
        await self.mq.publish(f"output.{self.agent_type}.{self.agent_id}", message)

    async def close(self):
        await self.mq.close()