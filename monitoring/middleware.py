import time
import psutil
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from api.schemas import MetricsResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("petr4-api")

_start_time = time.time()
_total_requests = 0
_response_times: list[float] = []


class MonitoringMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        global _total_requests

        t0 = time.time()
        response = await call_next(request)
        elapsed_ms = (time.time() - t0) * 1000

        _total_requests += 1
        _response_times.append(elapsed_ms)

        logger.info(
            f"{request.method} {request.url.path} "
            f"→ {response.status_code} | {elapsed_ms:.1f}ms"
        )
        return response


def get_metrics() -> MetricsResponse:
    process = psutil.Process()
    mem_mb = process.memory_info().rss / (1024 ** 2)
    cpu = psutil.cpu_percent(interval=0.1)
    avg_ms = sum(_response_times) / len(_response_times) if _response_times else 0.0

    return MetricsResponse(
        uptime_seconds=round(time.time() - _start_time, 1),
        total_requests=_total_requests,
        avg_response_time_ms=round(avg_ms, 2),
        cpu_percent=cpu,
        memory_mb=round(mem_mb, 2),
    )
