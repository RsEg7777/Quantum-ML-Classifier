from __future__ import annotations

import json
import time
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .runner import run_job
from .schemas import CreateJobInput

app = FastAPI(title="Quantum ML Worker", version="0.1.0")

# Worker metrics tracking
class WorkerMetrics:
    def __init__(self):
        self.startup_time = time.time()
        self.jobs_processed = 0
        self.jobs_failed = 0
        self.total_processing_time = 0.0

    def get_uptime_seconds(self) -> float:
        return time.time() - self.startup_time

    def record_success(self, duration: float):
        self.jobs_processed += 1
        self.total_processing_time += duration

    def record_failure(self):
        self.jobs_failed += 1

    def get_average_duration(self) -> float:
        if self.jobs_processed == 0:
            return 0.0
        return self.total_processing_time / self.jobs_processed


metrics = WorkerMetrics()

# WebSocket connection manager for broadcasting job updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                # Connection likely closed, mark for removal
                disconnected.add(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


@app.get("/health")
def health() -> dict:
    """Enhanced health check with worker metrics and diagnostics."""
    uptime = metrics.get_uptime_seconds()
    total_jobs = metrics.jobs_processed + metrics.jobs_failed
    success_rate = (
        (metrics.jobs_processed / total_jobs * 100)
        if total_jobs > 0
        else 0.0
    )

    return {
        "status": "healthy",
        "service": "quantum-ml-worker",
        "version": "0.1.0",
        "timestamp": time.time(),
        "metrics": {
            "uptime_seconds": round(uptime, 2),
            "jobs_processed": metrics.jobs_processed,
            "jobs_failed": metrics.jobs_failed,
            "total_jobs": total_jobs,
            "success_rate_percent": round(success_rate, 2),
            "average_duration_seconds": round(
                metrics.get_average_duration(), 4
            ),
        },
    }


@app.websocket("/ws/jobs")
async def websocket_jobs(websocket: WebSocket):
    """WebSocket endpoint for job status updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, receive ping/pong
            data = await websocket.receive_text()
            # Simple echo to keep connection active
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post("/jobs/run")
def run(payload: CreateJobInput):
    start_time = time.time()
    try:
        result = run_job(payload)
        duration = time.time() - start_time
        metrics.record_success(duration)

        # Keep /jobs/run synchronous and return typed result.
        # Frontend currently relies on API polling for status updates.
        return JSONResponse(
            content={
                "status": "completed",
                "result": result.model_dump(),
            }
        )
    except Exception:
        metrics.record_failure()
        raise
