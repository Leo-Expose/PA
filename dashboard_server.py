"""
Dashboard server for the PCAM Precision Agent benchmark.

Serves `pcam_dashboard.html` and exposes a small JSON API so the dashboard
can launch `run.py` for the three presets (quick / full / stress), tail
its stdout/stderr into the logs panel, and load the resulting `report.json`.

Stdlib only. Launch with:

    python dashboard_server.py
    # then open http://127.0.0.1:8765/

Optional flags:
    --port 8765
    --adapter adapters.archecho:Engine
    --host 127.0.0.1
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_HTML = os.path.join(ROOT, "pcam_dashboard.html")
REPORT_JSON = os.path.join(ROOT, "report.json")

# Preset → run.py CLI arguments. Quick mirrors `self_check.py --quick`.
PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "label": "Quick Test (G1)",
        "args": [
            "--seeds", "42", "101",
            "--noise-levels", "0.7", "0.8",
            "--n-per-level", "50",
            "--n-anisotropy", "5",
        ],
    },
    "full": {
        "label": "Full Evaluation (G3)",
        "args": [
            "--seeds", "42", "101", "202", "303", "404",
        ],
    },
    "stress": {
        "label": "Stress Test (G4)",
        "args": [
            "--seeds", "503", "1009", "9999",
        ],
    },
}


class RunState:
    """Thread-safe container for the current run + log buffer + last report."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.preset: str | None = None
        self.preset_label: str | None = None
        self.adapter: str = "adapters.archecho:Engine"
        self.started_at: float | None = None
        self.ended_at: float | None = None
        self.exit_code: int | None = None
        self.logs: list[dict[str, Any]] = []
        # Monotonic id so the client can fetch only new entries.
        self.next_log_id: int = 1
        # Start blank. Only populated once a run completes in this session.
        self.report: dict[str, Any] | None = None
        self.last_error: str | None = None

    # ------- logs -------

    def add_log(self, level: str, message: str) -> None:
        entry = {
            "id": self.next_log_id,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "level": level,
            "message": message,
        }
        self.next_log_id += 1
        self.logs.append(entry)
        # Keep the buffer bounded so a long stream never balloons memory.
        if len(self.logs) > 5000:
            self.logs = self.logs[-2500:]

    def clear_logs(self) -> None:
        self.logs = []
        self.next_log_id = 1

    # ------- run lifecycle -------

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def state_name(self) -> str:
        return "running" if self.is_running() else "idle"

    def elapsed_s(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.ended_at if (self.ended_at and not self.is_running()) else time.monotonic()
        return max(0.0, end - self.started_at)

    # ------- report -------

    def _read_report_safe(self) -> dict[str, Any] | None:
        try:
            with open(REPORT_JSON, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def reload_report(self) -> None:
        report = self._read_report_safe()
        if report is not None:
            self.report = report

    # ------- snapshot for the wire -------

    def snapshot(self, since_log_id: int = 0) -> dict[str, Any]:
        with self.lock:
            return {
                "state": self.state_name(),
                "preset": self.preset,
                "preset_label": self.preset_label,
                "adapter": self.adapter,
                "elapsed_s": round(self.elapsed_s(), 2),
                "exit_code": self.exit_code,
                "logs": [e for e in self.logs if e["id"] > since_log_id],
                "next_log_id": self.next_log_id,
                "report": self.report,
                "last_error": self.last_error,
            }


STATE = RunState()


# ---------- subprocess plumbing ----------

def _drain(stream, level: str) -> None:
    """Read a subprocess pipe line-by-line and append to the log buffer."""
    try:
        for raw in iter(stream.readline, ""):
            if not raw:
                break
            line = raw.rstrip("\n")
            if not line:
                continue
            with STATE.lock:
                STATE.add_log(level, line)
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _watch_completion(proc: subprocess.Popen, preset_label: str) -> None:
    """Wait for the subprocess to exit, then refresh the report."""
    code = proc.wait()
    with STATE.lock:
        STATE.ended_at = time.monotonic()
        STATE.exit_code = code
        if code == 0:
            STATE.add_log("success", f"{preset_label} finished cleanly.")
        elif code < 0:
            STATE.add_log("warning", f"{preset_label} stopped (signal {-code}).")
        else:
            STATE.add_log("error", f"{preset_label} exited with code {code}.")
    # Reload report.json regardless; partial runs may still be useful.
    STATE.reload_report()


def start_run(preset: str, adapter: str) -> tuple[bool, str]:
    if preset not in PRESETS:
        return False, f"unknown preset: {preset!r}"
    with STATE.lock:
        if STATE.is_running():
            return False, "a run is already in progress"
        spec = PRESETS[preset]
        cmd = [
            sys.executable, "-u", "run.py",
            "--adapter", adapter,
            "--out", "report.json",
            *spec["args"],
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            STATE.last_error = f"failed to launch run.py: {exc}"
            STATE.add_log("error", STATE.last_error)
            return False, STATE.last_error

        STATE.proc = proc
        STATE.preset = preset
        STATE.preset_label = spec["label"]
        STATE.adapter = adapter
        STATE.started_at = time.monotonic()
        STATE.ended_at = None
        STATE.exit_code = None
        STATE.last_error = None
        STATE.add_log("info", f"Starting {spec['label']} ({' '.join(cmd[1:])})")

    threading.Thread(target=_drain, args=(proc.stdout, "info"), daemon=True).start()
    threading.Thread(target=_drain, args=(proc.stderr, "info"), daemon=True).start()
    threading.Thread(target=_watch_completion, args=(proc, spec["label"]), daemon=True).start()
    return True, "started"


def stop_run() -> tuple[bool, str]:
    with STATE.lock:
        proc = STATE.proc
        if proc is None or proc.poll() is not None:
            return False, "no run in progress"
        STATE.add_log("warning", "Stop requested by user.")
    try:
        proc.terminate()
    except Exception as exc:
        return False, f"terminate failed: {exc}"
    # Give it 3s to exit gracefully, then kill.
    def _enforce():
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            with STATE.lock:
                STATE.add_log("warning", "Force-killing run.py.")
            try:
                proc.kill()
            except Exception:
                pass
    threading.Thread(target=_enforce, daemon=True).start()
    return True, "stopping"


# ---------- HTTP layer ----------

class Handler(BaseHTTPRequestHandler):
    server_version = "PCAMDashboard/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter access log
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    # ---- helpers ----

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: str, content_type: str) -> None:
        try:
            with open(path, "rb") as f:
                body = f.read()
        except FileNotFoundError:
            self.send_error(404, f"missing: {os.path.basename(path)}")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    # ---- routes ----

    def do_GET(self) -> None:  # noqa: N802 — http.server contract
        url = urlparse(self.path)
        path = url.path

        if path in ("/", "/index.html", "/dashboard"):
            self._send_file(DASHBOARD_HTML, "text/html; charset=utf-8")
            return

        if path == "/api/status":
            since = 0
            if url.query:
                from urllib.parse import parse_qs
                q = parse_qs(url.query)
                try:
                    since = int(q.get("since", ["0"])[0])
                except ValueError:
                    since = 0
            self._send_json(STATE.snapshot(since_log_id=since))
            return

        if path == "/api/report":
            STATE.reload_report()
            with STATE.lock:
                self._send_json(STATE.report or {})
            return

        if path == "/api/presets":
            self._send_json({k: {"label": v["label"]} for k, v in PRESETS.items()})
            return

        self.send_error(404, "not found")

    def do_POST(self) -> None:  # noqa: N802 — http.server contract
        url = urlparse(self.path)
        path = url.path
        body = self._read_json_body()

        if path == "/api/run":
            preset = (body.get("preset") or "").strip().lower()
            adapter = (body.get("adapter") or STATE.adapter).strip()
            ok, msg = start_run(preset, adapter)
            self._send_json({"ok": ok, "message": msg, **STATE.snapshot()})
            return

        if path == "/api/stop":
            ok, msg = stop_run()
            self._send_json({"ok": ok, "message": msg, **STATE.snapshot()})
            return

        if path == "/api/logs/clear":
            with STATE.lock:
                STATE.clear_logs()
            self._send_json({"ok": True})
            return

        self.send_error(404, "not found")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PCAM dashboard server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--adapter", default="adapters.archecho:Engine",
                    help="Default adapter spec, can be overridden per request.")
    args = ap.parse_args(argv)

    STATE.adapter = args.adapter

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"PCAM dashboard ready → {url}")
    print("Press Ctrl-C to stop.")

    def _shutdown(*_):
        print("\nShutting down…")
        try:
            stop_run()
        finally:
            threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
