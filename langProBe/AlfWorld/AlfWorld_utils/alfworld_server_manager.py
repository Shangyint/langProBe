from contextlib import contextmanager
from requests import ConnectionError
from typing import Generator
import subprocess
import sys
import threading
import time
import traceback

from .alfworld_client import AlfWorldClient


class AlfWorldServer:
    def __init__(self, port: int):
        self.port = port
        self.server = None
        self.lock = threading.Lock()
        self.base_url = f"http://localhost:{self.port}"
        self.request = AlfWorldClient(self.base_url)
        self.process = None
        self.stdout = None
        self.stderr = None

    @contextmanager
    def start_server(self, game_filepath: str):
        # # Start the appworld serve command in a separate process
        self.process = subprocess.Popen(
            # [sys.executable, '-m', 'langProBe.AlfWorld.AlfWorld_utils.alfworld_server', game_filepath, str(self.port)],
            f"{sys.executable} -m langProBe.AlfWorld.AlfWorld_utils.alfworld_server {game_filepath} {self.port}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.stdout = self.process.stdout
        self.stderr = self.process.stderr

        try:
            yield
        finally:
            # Terminate the process when the context is exited
            self.process.terminate()
            self.process.wait()


class AlfWorldServerManager:
    def __init__(self, num_servers: int):
        self.BASE_PORT = 9123
        self.servers = [AlfWorldServer(self.BASE_PORT + i) for i in range(num_servers)]

    @contextmanager
    def acquire_server(
        self, game_filepath: str
    ) -> Generator[AlfWorldServer, None, None]:
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
            with server.start_server(game_filepath=game_filepath):
                s = time.time()
                connected = False
                while time.time() - s < 100:
                    try:
                        result = server.request.reset()
                        connected = True
                        break
                    except (ConnectionError, NewConnectionError) as ce:
                        time.sleep(1)
                    except Exception as e:
                        print(traceback.format_exc())
                        raise e
                if not connected:
                    raise ConnectionError("Failed to connect to the server")
                yield server, result[0], result[1]
        finally:
            server.lock.release()


# This is a singleton
# TODO: Ensure that this is the same number of servers as the number of threads in Evaluate
alfworld_manager = AlfWorldServerManager(8)
