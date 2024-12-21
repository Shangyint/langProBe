from contextlib import contextmanager
from typing import Generator
import os
import subprocess
import threading
import time

from requests.exceptions import ConnectionError

from .app_world_client import AppWorldClient
from .appworld_setup import ensure_appworld_setup


class AppWorldServer:
    def __init__(self, port: int):
        self.port = port
        self.server = None
        self.lock = threading.Lock()
        self.base_url = f"http://localhost:{self.port}"
        self.request = AppWorldClient(self.base_url)

    @contextmanager
    def start_server(self):
        cmd, appworld_root = ensure_appworld_setup()

        # Start the appworld serve command in a separate process
        process = subprocess.Popen(
            f"{cmd} appworld serve environment --root {appworld_root} --port {self.port}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            yield
        finally:
            # Terminate the process when the context is exited
            process.terminate()
            process.wait()


class AppWorldServerManager:
    def __init__(self, num_servers: int):
        self.BASE_PORT = 8123
        self.servers = [AppWorldServer(self.BASE_PORT + i) for i in range(num_servers)]

    @contextmanager
    def acquire_server(
        self, experiment_name: str, task_id: str
    ) -> Generator[AppWorldServer, None, None]:
        acquired_idx = None
        while True:
            for i, server in enumerate(self.servers):
                if server.lock.acquire(blocking=False):
                    acquired_idx = i
                    break
            if acquired_idx is not None:
                break
            time.sleep(0.1)

        try:
            server = self.servers[acquired_idx]
            with server.start_server():
                s = time.time()
                connected = False
                while time.time() - s < 10:
                    try:
                        server.request.initialize(experiment_name=experiment_name, task_id=task_id)
                        connected = True
                        break
                    except ConnectionError:
                        time.sleep(0.1)
                if not connected:
                    raise ConnectionError("Failed to connect to the server")
                yield server
                server.request.close(task_id=task_id)
        finally:
            server.lock.release()


# This is a singleton
# TODO: Ensure that this is the same number of servers as the number of threads in Evaluate
appworld_manager = AppWorldServerManager(8)
