import asyncio
import time

class RateLimiter:
    """A simple async rate limiter to control requests per minute."""
    def __init__(self, max_rpm: int):
        if max_rpm <= 0:
            self.delay = 0
        else:
            self.delay = 60.0 / max_rpm
        self.lock = asyncio.Lock()
        self.last_request_time = 0

    async def __aenter__(self):
        if self.delay == 0:
            return
        
        async with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_request_time
            if elapsed < self.delay:
                await asyncio.sleep(self.delay - elapsed)
            self.last_request_time = time.time()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass