"""Run TensorBoard next to NVC Studio and serve it under /tensorboard/.

TensorBoard listens on localhost only; Studio forwards requests to it, so the
dashboard works through the same address (and the same Colab tunnel) as the
Studio itself.
"""

import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

PATH_PREFIX = "/tensorboard"
HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-encoding",
    "content-length",
    "host",
}


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


class TensorBoardServer:
    def __init__(self, logdir="logs", python=None):
        self.logdir = Path(logdir)
        self.python = python or sys.executable
        self.process = None
        self.port = None
        self.error = ""
        self._lock = threading.Lock()
        self._client = httpx.Client(timeout=60.0)

    def running(self):
        return self.process is not None and self.process.poll() is None

    def status(self):
        return {
            "running": self.running(),
            "url": PATH_PREFIX + "/",
            "logdir": str(self.logdir),
            "error": self.error,
        }

    def start(self, timeout=60):
        with self._lock:
            if self.running() and _port_open(self.port):
                return self.status()
            self.stop_locked()
            self.logdir.mkdir(parents=True, exist_ok=True)
            self.port = _free_port()
            self.error = ""
            cmd = [
                self.python,
                "-m",
                "tensorboard.main",
                "--logdir",
                str(self.logdir),
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--path_prefix",
                PATH_PREFIX,
                "--reload_interval",
                "15",
            ]
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            deadline = time.time() + timeout
            while time.time() < deadline:
                if self.process.poll() is not None:
                    self.error = (self.process.stderr.read() or "").strip()[-2000:]
                    self.process = None
                    raise RuntimeError(self.error or "TensorBoard exited")
                if _port_open(self.port):
                    # Drain stderr so a chatty TensorBoard never blocks on a full pipe.
                    threading.Thread(
                        target=self._drain, args=(self.process,), daemon=True
                    ).start()
                    return self.status()
                time.sleep(0.3)
            self.stop_locked()
            raise TimeoutError("TensorBoard did not start in %s s" % timeout)

    @staticmethod
    def _drain(process):
        for _ in process.stderr:
            pass

    def stop_locked(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None

    def stop(self):
        with self._lock:
            self.stop_locked()
        return self.status()

    def forward(self, method, path, query, headers, body):
        if not self.running():
            self.start()
        url = "http://127.0.0.1:%s%s/%s" % (self.port, PATH_PREFIX, path)
        if query:
            url += "?" + query
        upstream = self._client.request(
            method,
            url,
            headers={k: v for k, v in headers.items() if k.lower() not in HOP_HEADERS},
            content=body,
        )
        response_headers = {
            k: v for k, v in upstream.headers.items() if k.lower() not in HOP_HEADERS
        }
        return upstream.status_code, response_headers, upstream.content
