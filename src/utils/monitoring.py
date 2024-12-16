from typing import Dict, Any
from datetime import datetime, timedelta
import psutil
import asyncio
from loguru import logger
import aioredis
import os

class SystemMonitor:
    def __init__(self):
        self.metrics = {}
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost")
        self.redis = None

    async def connect(self):
        self.redis = await aioredis.from_url(self.redis_url)

    async def close(self):
        if self.redis:
            await self.redis.close()

    async def collect_system_metrics(self):
        try:
            metrics = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "timestamp": datetime.utcnow().isoformat()
            }
            await self.redis.hset("system_metrics", mapping=metrics)
            return metrics
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return None

    async def collect_agent_metrics(self, agent_id: str, metrics: Dict[str, Any]):
        try:
            metrics["timestamp"] = datetime.utcnow().isoformat()
            await self.redis.hset(f"agent_metrics:{agent_id}", mapping=metrics)
            return metrics
        except Exception as e:
            logger.error(f"Error collecting agent metrics: {e}")
            return None

    async def get_agent_metrics(self, agent_id: str):
        try:
            metrics = await self.redis.hgetall(f"agent_metrics:{agent_id}")
            return metrics
        except Exception as e:
            logger.error(f"Error getting agent metrics: {e}")
            return None

    async def collect_api_metrics(self, endpoint: str, response_time: float, status_code: int):
        try:
            metrics = {
                "endpoint": endpoint,
                "response_time": response_time,
                "status_code": status_code,
                "timestamp": datetime.utcnow().isoformat()
            }
            await self.redis.lpush("api_metrics", str(metrics))
            await self.redis.ltrim("api_metrics", 0, 999)  # Keep last 1000 requests
            return metrics
        except Exception as e:
            logger.error(f"Error collecting API metrics: {e}")
            return None

    async def get_api_metrics(self, time_range: timedelta = timedelta(minutes=5)):
        try:
            metrics = await self.redis.lrange("api_metrics", 0, -1)
            cutoff_time = datetime.utcnow() - time_range
            filtered_metrics = []
            for metric in metrics:
                metric_dict = eval(metric)
                metric_time = datetime.fromisoformat(metric_dict["timestamp"])
                if metric_time >= cutoff_time:
                    filtered_metrics.append(metric_dict)
            return filtered_metrics
        except Exception as e:
            logger.error(f"Error getting API metrics: {e}")
            return None

class AgentMonitor:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.start_time = datetime.utcnow()
        self.metrics = {
            "processed_messages": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "average_processing_time": 0.0,
            "total_processing_time": 0.0
        }
        self.system_monitor = SystemMonitor()

    async def connect(self):
        await self.system_monitor.connect()

    async def close(self):
        await self.system_monitor.close()

    async def record_message_processed(self, processing_time: float, success: bool = True):
        self.metrics["processed_messages"] += 1
        self.metrics["total_processing_time"] += processing_time
        self.metrics["average_processing_time"] = (
            self.metrics["total_processing_time"] / self.metrics["processed_messages"]
        )
        
        if success:
            self.metrics["successful_operations"] += 1
        else:
            self.metrics["failed_operations"] += 1

        await self.system_monitor.collect_agent_metrics(self.agent_id, self.metrics)

    async def get_metrics(self):
        uptime = datetime.utcnow() - self.start_time
        metrics = {
            **self.metrics,
            "uptime_seconds": uptime.total_seconds(),
            "success_rate": (
                self.metrics["successful_operations"] /
                max(self.metrics["processed_messages"], 1)
            ) * 100
        }
        return metrics

class APIMonitor:
    def __init__(self):
        self.system_monitor = SystemMonitor()

    async def connect(self):
        await self.system_monitor.connect()

    async def close(self):
        await self.system_monitor.close()

    async def record_request(self, endpoint: str, response_time: float, status_code: int):
        await self.system_monitor.collect_api_metrics(endpoint, response_time, status_code)

    async def get_metrics(self, time_range: timedelta = timedelta(minutes=5)):
        metrics = await self.system_monitor.get_api_metrics(time_range)
        if not metrics:
            return None

        total_requests = len(metrics)
        success_requests = len([m for m in metrics if 200 <= m["status_code"] < 300])
        avg_response_time = sum(m["response_time"] for m in metrics) / total_requests

        return {
            "total_requests": total_requests,
            "success_rate": (success_requests / total_requests) * 100,
            "average_response_time": avg_response_time,
            "time_range_minutes": time_range.total_seconds() / 60
        }