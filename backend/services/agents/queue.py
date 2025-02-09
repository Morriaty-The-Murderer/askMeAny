"""NL2SQL request queue system.

This module implements a priority-based request queue with timeout handling and
concurrent processing capabilities for the NL2SQL system.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RequestStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class Request:
    """Container for queue request data."""

    id: str
    text: str
    schema: Dict[str, Any]
    priority: int
    created_at: datetime
    timeout: int = 30
    status: RequestStatus = RequestStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None


class QueueError(Exception):
    """Base exception for queue-related errors."""
    pass


class QueueFullError(QueueError):
    """Raised when queue is at capacity."""
    pass


class RequestTimeoutError(QueueError):
    """Raised when request times out."""
    pass


class RequestQueue:
    """Priority-based request queue with timeout handling."""

    def __init__(
            self,
            max_size: int = 1000,
            workers: int = 5,
            default_timeout: int = 30
    ):
        """Initialize request queue.
        
        Args:
            max_size: Maximum queue size
            workers: Number of concurrent workers
            default_timeout: Default request timeout in seconds
        """
        self.max_size = max_size
        self.num_workers = workers
        self.default_timeout = default_timeout

        self.queue = asyncio.PriorityQueue(maxsize=max_size)
        self.requests: Dict[str, Request] = {}
        self.workers: List[asyncio.Task] = []
        self.running = False

        self.logger = logging.getLogger(__name__)
        self.stats = {
            "processed": 0,
            "failed": 0,
            "timeouts": 0
        }

    async def add_request(
            self,
            request_id: str,
            text: str,
            schema: Dict[str, Any],
            priority: int = 0,
            timeout: Optional[int] = None
    ) -> None:
        """Add request to queue.
        
        Args:
            request_id: Unique request identifier
            text: Input text to process
            schema: Database schema information
            priority: Request priority (lower = higher priority)
            timeout: Optional custom timeout in seconds
            
        Raises:
            QueueFullError: If queue is at capacity
        """
        if len(self.requests) >= self.max_size:
            raise QueueFullError("Queue is at maximum capacity")

        request = Request(
            id=request_id,
            text=text,
            schema=schema,
            priority=priority,
            created_at=datetime.now(),
            timeout=timeout or self.default_timeout
        )

        self.requests[request_id] = request
        await self.queue.put((priority, request))
        self.logger.debug(f"Added request {request_id} to queue")

    def get_request(self, request_id: str) -> Optional[Request]:
        """Get request by ID.
        
        Args:
            request_id: Request identifier
            
        Returns:
            Request object if found, None otherwise
        """
        return self.requests.get(request_id)

    async def process_request(self, request: Request, model: BaseModel) -> None:
        """Process single request with timeout handling.
        
        Args:
            request: Request to process
            model: Model instance for generation
        """
        try:
            request.status = RequestStatus.PROCESSING

            # Generate SQL query
            prompt = self._create_prompt(request.text, request.schema)

            if (datetime.now() - request.created_at).seconds > request.timeout:
                request.status = RequestStatus.TIMEOUT
                request.error = "Request timed out"
                self.stats["timeouts"] += 1
                return

            # Process using model
            request.result = await model.generate(prompt)
            request.status = RequestStatus.COMPLETED 
            self.stats["processed"] += 1

        except Exception as e:
            request.status = RequestStatus.FAILED
            request.error = str(e)
            self.stats["failed"] += 1
            self.logger.error(f"Error processing request {request.id}: {e}")

    async def worker(self, model: BaseModel) -> None:
        """Worker coroutine for processing queue items.
        
        Args:
            model: Model instance for generation
        """
        while self.running:
            try:
                priority, request = await self.queue.get()
                await self.process_request(request, model)
                self.queue.task_done()
            except Exception as e:
                self.logger.error(f"Worker error: {e}")

    def start(self) -> None:
        """Start queue processing workers."""
        self.running = True
        self.workers = [
            asyncio.create_task(self.worker())
            for _ in range(self.num_workers)
        ]
        self.logger.info(f"Started {self.num_workers} queue workers")

    async def stop(self) -> None:
        """Stop queue processing."""
        self.running = False
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.logger.info("Stopped queue processing")

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics.
        
        Returns:
            Dictionary of queue statistics
        """
        return {
            **self.stats,
            "queue_size": self.queue.qsize(),
            "num_requests": len(self.requests),
            "active_workers": len([w for w in self.workers if not w.done()]),
        }

    def cleanup_old_requests(self, max_age: int = 3600) -> None:
        """Clean up old completed requests.
        
        Args:
            max_age: Maximum request age in seconds
        """
        now = datetime.now()
        to_remove = []

        for request_id, request in self.requests.items():
            age = (now - request.created_at).seconds
            if age > max_age and request.status in (
                    RequestStatus.COMPLETED,
                    RequestStatus.FAILED,
                    RequestStatus.TIMEOUT
            ):
                to_remove.append(request_id)

        for request_id in to_remove:
            del self.requests[request_id]

        self.logger.info(f"Cleaned up {len(to_remove)} old requests")
