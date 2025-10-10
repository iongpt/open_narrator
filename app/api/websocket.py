"""Server-Sent Events (SSE) for real-time progress updates."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse

from app.database import get_db
from app.models import Job
from app.schemas import ProgressUpdate

router = APIRouter()


# Global event broadcaster
# In production, use Redis or message queue for multi-worker support
class ProgressBroadcaster:
    """Manages SSE connections and broadcasts progress updates."""

    def __init__(self) -> None:
        """Initialize the broadcaster with an empty client list."""
        self.clients: list[asyncio.Queue[dict[str, Any]]] = []

    def add_client(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """
        Add a new SSE client connection.

        Args:
            queue: Queue for sending events to this client
        """
        self.clients.append(queue)

    def remove_client(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """
        Remove a disconnected SSE client.

        Args:
            queue: Queue of the client to remove
        """
        if queue in self.clients:
            self.clients.remove(queue)

    async def broadcast(self, update: ProgressUpdate) -> None:
        """
        Broadcast a progress update to all connected clients.

        Args:
            update: Progress update to broadcast
        """
        # Convert to dict for JSON serialization
        update_dict = update.model_dump()

        # Send to all connected clients
        disconnected_clients = []

        for client_queue in self.clients:
            try:
                await client_queue.put(update_dict)
            except Exception:
                # Client disconnected, mark for removal
                disconnected_clients.append(client_queue)

        # Clean up disconnected clients
        for client in disconnected_clients:
            self.remove_client(client)


# Singleton broadcaster instance
broadcaster = ProgressBroadcaster()


async def event_generator(job_id: int | None, db: Session) -> AsyncIterator[dict[str, Any]]:
    """
    Generate SSE events for progress updates.

    Args:
        job_id: Optional job ID to filter updates (None for all jobs)
        db: Database session

    Yields:
        SSE events with progress updates
    """
    # Create queue for this client
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    broadcaster.add_client(queue)

    try:
        # Send initial state for the requested job(s)
        if job_id is not None:
            job = db.query(Job).filter(Job.id == job_id).first()
            if job:
                initial_update = ProgressUpdate(
                    job_id=job.id,
                    status=job.status,
                    progress=job.progress,
                    message=None,
                )
                yield {
                    "event": "progress",
                    "data": json.dumps(initial_update.model_dump()),
                }
        else:
            # Send state for all jobs
            jobs = db.query(Job).all()
            for job in jobs:
                initial_update = ProgressUpdate(
                    job_id=job.id,
                    status=job.status,
                    progress=job.progress,
                    message=None,
                )
                yield {
                    "event": "progress",
                    "data": json.dumps(initial_update.model_dump()),
                }

        # Stream updates from queue
        while True:
            update = await queue.get()

            # Filter by job_id if specified
            if job_id is not None and update.get("job_id") != job_id:
                continue

            yield {
                "event": "progress",
                "data": json.dumps(update),
            }

    except asyncio.CancelledError:
        # Client disconnected
        broadcaster.remove_client(queue)
        raise
    except Exception as e:
        # Log error and disconnect
        print(f"Error in SSE event generator: {e}")
        broadcaster.remove_client(queue)
        raise


@router.get("/progress")
async def progress_stream(
    job_id: int | None = None, db: Session = Depends(get_db)
) -> EventSourceResponse:
    """
    Server-Sent Events endpoint for real-time progress updates.

    Args:
        job_id: Optional job ID to subscribe to (None for all jobs)
        db: Database session

    Returns:
        SSE stream of progress updates

    Example usage:
        JavaScript:
        ```javascript
        const eventSource = new EventSource('/progress?job_id=123');
        eventSource.addEventListener('progress', (e) => {
            const data = JSON.parse(e.data);
            console.log(`Job ${data.job_id}: ${data.progress}%`);
        });
        ```
    """
    return EventSourceResponse(event_generator(job_id, db))


async def send_progress_update(update: ProgressUpdate) -> None:
    """
    Send a progress update to all connected SSE clients.

    This function should be called by the pipeline service when
    job status or progress changes.

    Args:
        update: Progress update to broadcast

    Example:
        ```python
        from app.api.websocket import send_progress_update
        from app.schemas import ProgressUpdate
        from app.models import JobStatus

        await send_progress_update(
            ProgressUpdate(
                job_id=123,
                status=JobStatus.TRANSCRIBING,
                progress=15.0,
                message="Transcribing audio..."
            )
        )
        ```
    """
    await broadcaster.broadcast(update)
