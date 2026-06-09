"""gRPC server for the Gyroid Optimizer.

Usage:
    python server.py [--port 50051] [--host 0.0.0.0]

The server exposes gyroid_service.proto over an insecure channel.
Add TLS certificates to grpc.ssl_server_credentials() for production use.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import math
import os
import queue
import signal
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path
from typing import Iterator

import grpc
import yaml

import gyroid_service_pb2 as pb2
import gyroid_service_pb2_grpc as pb2_grpc

# ── Paths ─────────────────────────────────────────────────────────────────────

GRPC_DIR         = Path(__file__).resolve().parent
BASE_DIR         = GRPC_DIR.parent                          # 3Dheatsink_gyroid/
CONFIG_PATH      = BASE_DIR / "gyroid_case_config.yaml"
WRAPPER          = BASE_DIR / "gyroid_case_wrapper.py"
STL_SCRIPT       = BASE_DIR / "gyroid_to_stl.py"
QUAD_MESH_SCRIPT = BASE_DIR / "gyroid_to_quad_mesh_qf.py"
NURBS_SCRIPT     = BASE_DIR / "quad_to_nurbs.py"
APP_DIR          = BASE_DIR / "app"
HISTORY          = APP_DIR  / "gyroid_opt_history.txt"

# Named STL outputs produced by gyroid_to_stl.py (relative to BASE_DIR)
_STL_FILES = {
    "lattice": BASE_DIR / "gyroid_surface_lattice.stl",
    "encap":   BASE_DIR / "gyroid_surface_encap.stl",
    "surface": BASE_DIR / "gyroid_surface.stl",
}

# Default OBJ and STEP paths for the two gyroid sheets
_NURBS_OBJ = {
    "plus":  BASE_DIR / "gyroid_implicit_qf_plus.obj",
    "minus": BASE_DIR / "gyroid_implicit_qf_minus.obj",
}
_NURBS_STEP = {
    "plus":  BASE_DIR / "gyroid_implicit_qf_plus.step",
    "minus": BASE_DIR / "gyroid_implicit_qf_minus.step",
}

CHUNK_SIZE  = 64 * 1024  # 64 KB per FileChunk


# ── Subprocess / output manager ───────────────────────────────────────────────

class RunnerState:
    """Manages one gyroid_case_wrapper.py subprocess at a time.

    Multiple gRPC clients can subscribe to output simultaneously; each gets
    the full buffered history plus live lines from that point onward.
    """

    def __init__(self) -> None:
        self._proc:           subprocess.Popen | None = None
        self._proc_lock       = threading.Lock()
        self._return_code:    int | None = None

        # Protected by _sub_lock
        self._history:        list[dict] = []
        self._subscribers:    list[queue.Queue] = []
        self._reader_done:    bool = True   # True = no active reader thread
        self._sub_lock        = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        with self._proc_lock:
            return self._proc is not None and self._proc.poll() is None

    @property
    def pid(self) -> int | None:
        with self._proc_lock:
            return self._proc.pid if self._proc is not None else None

    @property
    def return_code(self) -> int | None:
        with self._proc_lock:
            if self._proc is not None:
                rc = self._proc.poll()
                if rc is not None:
                    self._return_code = rc
            return self._return_code

    def start(self, extra_args: list[str], script: Path = None,
              use_raw_cmd: bool = False) -> None:
        with self._proc_lock:
            if self._proc is not None and self._proc.poll() is None:
                raise RuntimeError("A run is already in progress")
            # Clear history for the new run
            with self._sub_lock:
                self._history.clear()
                self._reader_done = False

            if use_raw_cmd:
                # extra_args is already a complete command list
                cmd = extra_args
            else:
                target = script if script is not None else WRAPPER
                cmd = [sys.executable, str(target)] + extra_args
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self._return_code = None
            t = threading.Thread(target=self._reader, daemon=True)
            t.start()

    def stop(self) -> None:
        with self._proc_lock:
            if self._proc is None or self._proc.poll() is not None:
                return
            self._proc.send_signal(signal.SIGTERM)
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def subscribe(self) -> tuple[list[dict], queue.Queue | None]:
        """Return a history snapshot and (if running) a live-output queue.

        The caller must call unsubscribe(q) when done to avoid memory leaks.
        If the run is not active, the returned queue is None.
        """
        with self._sub_lock:
            snapshot = list(self._history)
            if self._reader_done:
                return snapshot, None
            q: queue.Queue = queue.Queue()
            self._subscribers.append(q)
            return snapshot, q

    def unsubscribe(self, q: queue.Queue | None) -> None:
        if q is None:
            return
        with self._sub_lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def output_stream(self) -> Iterator[dict]:
        """Yield history entries, then live entries until the run ends."""
        snapshot, q = self.subscribe()
        try:
            for entry in snapshot:
                yield entry
            if q is None:
                return
            while True:
                try:
                    entry = q.get(timeout=1.0)
                    if entry is None:   # sentinel: reader thread finished
                        break
                    yield entry
                except queue.Empty:
                    continue
        finally:
            self.unsubscribe(q)

    # ── Private ───────────────────────────────────────────────────────────────

    def _reader(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for raw_line in proc.stdout:
                entry = {
                    "line":         raw_line.rstrip("\n"),
                    "is_stderr":    False,
                    "timestamp_ms": int(time.time() * 1000),
                }
                with self._sub_lock:
                    self._history.append(entry)
                    for q in self._subscribers:
                        q.put(entry)
        finally:
            with self._sub_lock:
                self._reader_done = True
                for q in self._subscribers:
                    q.put(None)   # signal end of stream


_runner           = RunnerState()   # optimizer subprocess
_stl_runner       = RunnerState()   # gyroid_to_stl.py subprocess
_quad_mesh_runner = RunnerState()   # gyroid_to_quad_mesh_qf.py subprocess
_nurbs_runner     = RunnerState()   # quad_to_nurbs.py subprocess


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_history() -> tuple[list[str], list[pb2.HistoryRow]]:
    try:
        raw_lines = [l for l in HISTORY.read_text().splitlines() if l.strip()]
    except OSError:
        return [], []
    if not raw_lines:
        return [], []

    header = raw_lines[0].split()
    rows: list[pb2.HistoryRow] = []
    for line in raw_lines[1:]:
        parts = line.split()
        if len(parts) != len(header):
            continue
        values: dict[str, float] = {}
        strings: dict[str, str]  = {}
        for key, val in zip(header, parts):
            try:
                values[key] = float(val)
            except ValueError:
                strings[key] = val
        rows.append(pb2.HistoryRow(values=values, strings=strings))
    return header, rows


def _entry_to_proto(entry: dict) -> pb2.OutputLine:
    return pb2.OutputLine(
        line=entry["line"],
        is_stderr=entry["is_stderr"],
        timestamp_ms=entry["timestamp_ms"],
    )


def _stream_file(path: Path, filename: str) -> Iterator[pb2.FileChunk]:
    total = path.stat().st_size
    with open(path, "rb") as fh:
        sent = 0
        while True:
            chunk = fh.read(CHUNK_SIZE)
            if not chunk:
                break
            sent += len(chunk)
            yield pb2.FileChunk(
                data=chunk,
                filename=filename,
                total_size=total,
                is_last=(sent >= total),
            )


def _stream_tar(src: Path) -> Iterator[pb2.FileChunk]:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(str(src), arcname=src.name)
    data  = buf.getvalue()
    total = len(data)
    fname = src.name + ".tar.gz"
    for offset in range(0, total, CHUNK_SIZE):
        chunk   = data[offset : offset + CHUNK_SIZE]
        is_last = (offset + CHUNK_SIZE) >= total
        yield pb2.FileChunk(data=chunk, filename=fname, total_size=total, is_last=is_last)


# ── Servicer ──────────────────────────────────────────────────────────────────

class GyroidOptimizerServicer(pb2_grpc.GyroidOptimizerServicer):

    # ── Config ────────────────────────────────────────────────────────────────

    def GetConfig(self, request, context):
        try:
            return pb2.ConfigResponse(yaml_content=CONFIG_PATH.read_text(), success=True)
        except Exception as exc:
            return pb2.ConfigResponse(success=False, error=str(exc))

    def SetConfig(self, request, context):
        try:
            yaml.safe_load(request.yaml_content)   # validate
            CONFIG_PATH.write_text(request.yaml_content)
            return pb2.StatusResponse(success=True, message="Config written")
        except Exception as exc:
            return pb2.StatusResponse(success=False, message=str(exc))

    def PatchConfig(self, request, context):
        try:
            patch: dict = json.loads(request.json_patch)
            with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
                cfg: dict = yaml.safe_load(fh) or {}

            for dot_key, value in patch.items():
                keys = dot_key.split(".")
                node = cfg
                for k in keys[:-1]:
                    node = node.setdefault(k, {})
                node[keys[-1]] = value

            with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
                yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)

            return pb2.StatusResponse(success=True, message=f"Patched {len(patch)} key(s)")
        except Exception as exc:
            return pb2.StatusResponse(success=False, message=str(exc))

    # ── Run control ───────────────────────────────────────────────────────────

    def StartRun(self, request, context):
        try:
            _runner.start(list(request.extra_args))
            return pb2.StatusResponse(success=True, message=f"Started (pid={_runner.pid})")
        except Exception as exc:
            return pb2.StatusResponse(success=False, message=str(exc))

    def StopRun(self, request, context):
        try:
            _runner.stop()
            return pb2.StatusResponse(success=True, message="Stop signal sent")
        except Exception as exc:
            return pb2.StatusResponse(success=False, message=str(exc))

    def GetRunStatus(self, request, context):
        rc  = _runner.return_code
        pid = _runner.pid or 0
        if _runner.is_running:
            state = pb2.RunStatusResponse.RUNNING
            msg   = f"Running (pid={pid})"
        elif rc is None:
            state = pb2.RunStatusResponse.IDLE
            msg   = "Idle — no run has been started yet"
        elif rc == 0:
            state = pb2.RunStatusResponse.DONE
            msg   = "Completed successfully"
        else:
            state = pb2.RunStatusResponse.ERROR
            msg   = f"Exited with return code {rc}"
        return pb2.RunStatusResponse(state=state, pid=pid, message=msg, return_code=rc or 0)

    # ── Output streaming ──────────────────────────────────────────────────────

    def StreamOutput(self, request, context):
        for entry in _runner.output_stream():
            if not context.is_active():
                return
            yield _entry_to_proto(entry)

    # ── Results ───────────────────────────────────────────────────────────────

    def GetHistory(self, request, context):
        try:
            columns, rows = _parse_history()
            return pb2.HistoryResponse(success=True, columns=columns, rows=rows)
        except Exception as exc:
            return pb2.HistoryResponse(success=False, error=str(exc))

    def GetLatestMetrics(self, request, context):
        try:
            _, rows = _parse_history()
            if not rows:
                return pb2.LatestMetricsResponse(available=False)
            return pb2.LatestMetricsResponse(available=True, latest=rows[-1])
        except Exception as exc:
            return pb2.LatestMetricsResponse(available=False, error=str(exc))

    # ── File access ───────────────────────────────────────────────────────────

    def ListFiles(self, request, context):
        try:
            base = (APP_DIR / request.path) if request.path else APP_DIR
            base = base.resolve()
            if not str(base).startswith(str(APP_DIR)):
                context.abort(grpc.StatusCode.PERMISSION_DENIED, "Path escapes app dir")
                return
            paths = sorted(
                str(p.relative_to(APP_DIR))
                for p in base.rglob("*")
                if p.is_file()
            )
            return pb2.FileListResponse(paths=paths, success=True)
        except Exception as exc:
            return pb2.FileListResponse(success=False, error=str(exc))

    def DownloadFile(self, request, context):
        target = (APP_DIR / request.path).resolve() if request.path else APP_DIR.resolve()

        # Safety: prevent path traversal outside app dir
        if not str(target).startswith(str(APP_DIR.resolve())):
            context.abort(grpc.StatusCode.PERMISSION_DENIED, "Path escapes app dir")
            return

        if not target.exists():
            context.abort(grpc.StatusCode.NOT_FOUND, f"Not found: {request.path}")
            return

        try:
            if target.is_dir() or request.as_tar:
                yield from _stream_tar(target)
            else:
                yield from _stream_file(target, target.name)
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    # ── STL export ────────────────────────────────────────────────────────────

    def StartStlExport(self, request, context):
        try:
            _stl_runner.start(list(request.extra_args), script=STL_SCRIPT)
            return pb2.StatusResponse(success=True, message=f"STL export started (pid={_stl_runner.pid})")
        except Exception as exc:
            return pb2.StatusResponse(success=False, message=str(exc))

    def StopStlExport(self, request, context):
        try:
            _stl_runner.stop()
            return pb2.StatusResponse(success=True, message="STL export stop signal sent")
        except Exception as exc:
            return pb2.StatusResponse(success=False, message=str(exc))

    def GetStlStatus(self, request, context):
        rc  = _stl_runner.return_code
        pid = _stl_runner.pid or 0
        if _stl_runner.is_running:
            state = pb2.RunStatusResponse.RUNNING
            msg   = f"Running (pid={pid})"
        elif rc is None:
            state = pb2.RunStatusResponse.IDLE
            msg   = "Idle — no STL export has been started yet"
        elif rc == 0:
            state = pb2.RunStatusResponse.DONE
            msg   = "Completed successfully"
        else:
            state = pb2.RunStatusResponse.ERROR
            msg   = f"Exited with return code {rc}"
        return pb2.RunStatusResponse(state=state, pid=pid, message=msg, return_code=rc or 0)

    def StreamStlOutput(self, request, context):
        for entry in _stl_runner.output_stream():
            if not context.is_active():
                return
            yield _entry_to_proto(entry)

    def DownloadStl(self, request, context):
        which = (request.which or "lattice").lower().strip()

        if which == "all":
            # Tar all gyroid_surface*.stl files that exist next to the script
            stl_paths = sorted(BASE_DIR.glob("gyroid_surface*.stl"))
            if not stl_paths:
                context.abort(grpc.StatusCode.NOT_FOUND,
                              "No gyroid_surface*.stl files found in " + str(BASE_DIR))
                return
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                for p in stl_paths:
                    tar.add(str(p), arcname=p.name)
            data  = buf.getvalue()
            total = len(data)
            fname = "gyroid_surface_all.tar.gz"
            for offset in range(0, total, CHUNK_SIZE):
                chunk   = data[offset : offset + CHUNK_SIZE]
                is_last = (offset + CHUNK_SIZE) >= total
                yield pb2.FileChunk(data=chunk, filename=fname, total_size=total, is_last=is_last)
            return

        target = _STL_FILES.get(which)
        if target is None:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                          f"Unknown STL type {which!r}. Use: lattice, encap, surface, all")
            return
        if not target.exists():
            context.abort(grpc.StatusCode.NOT_FOUND,
                          f"{target.name} not found — run StartStlExport first")
            return
        try:
            yield from _stream_file(target, target.name)
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    # ── Gyroid → quad mesh ────────────────────────────────────────────────────

    def StartGyroidToQuadMesh(self, request, context):
        try:
            _quad_mesh_runner.start(list(request.extra_args), script=QUAD_MESH_SCRIPT)
            return pb2.StatusResponse(
                success=True,
                message=f"Gyroid-to-quad-mesh started (pid={_quad_mesh_runner.pid})",
            )
        except Exception as exc:
            return pb2.StatusResponse(success=False, message=str(exc))

    def StopGyroidToQuadMesh(self, request, context):
        try:
            _quad_mesh_runner.stop()
            return pb2.StatusResponse(success=True, message="Quad-mesh stop signal sent")
        except Exception as exc:
            return pb2.StatusResponse(success=False, message=str(exc))

    def GetGyroidToQuadMeshStatus(self, request, context):
        rc  = _quad_mesh_runner.return_code
        pid = _quad_mesh_runner.pid or 0
        if _quad_mesh_runner.is_running:
            state = pb2.RunStatusResponse.RUNNING
            msg   = f"Running (pid={pid})"
        elif rc is None:
            state = pb2.RunStatusResponse.IDLE
            msg   = "Idle — no quad-mesh run has been started yet"
        elif rc == 0:
            state = pb2.RunStatusResponse.DONE
            msg   = "Completed successfully"
        else:
            state = pb2.RunStatusResponse.ERROR
            msg   = f"Exited with return code {rc}"
        return pb2.RunStatusResponse(state=state, pid=pid, message=msg, return_code=rc or 0)

    def StreamGyroidToQuadMeshOutput(self, request, context):
        for entry in _quad_mesh_runner.output_stream():
            if not context.is_active():
                return
            yield _entry_to_proto(entry)

    # ── Quad mesh → NURBS STEP ────────────────────────────────────────────────

    def StartQuadToNurbs(self, request, context):
        which = (request.which or "plus").lower().strip()
        if which not in _NURBS_OBJ:
            return pb2.StatusResponse(
                success=False,
                message=f"Unknown sheet {which!r}. Use 'plus' or 'minus'.",
            )
        in_obj   = _NURBS_OBJ[which]
        out_step = _NURBS_STEP[which]
        cmd = [
            sys.executable, str(NURBS_SCRIPT),
            str(in_obj), str(out_step),
        ] + list(request.extra_args)
        try:
            _nurbs_runner.start(cmd, use_raw_cmd=True)
            return pb2.StatusResponse(
                success=True,
                message=f"QuadToNurbs started for '{which}' (pid={_nurbs_runner.pid})",
            )
        except Exception as exc:
            return pb2.StatusResponse(success=False, message=str(exc))

    def StopQuadToNurbs(self, request, context):
        try:
            _nurbs_runner.stop()
            return pb2.StatusResponse(success=True, message="QuadToNurbs stop signal sent")
        except Exception as exc:
            return pb2.StatusResponse(success=False, message=str(exc))

    def GetQuadToNurbsStatus(self, request, context):
        rc  = _nurbs_runner.return_code
        pid = _nurbs_runner.pid or 0
        if _nurbs_runner.is_running:
            state = pb2.RunStatusResponse.RUNNING
            msg   = f"Running (pid={pid})"
        elif rc is None:
            state = pb2.RunStatusResponse.IDLE
            msg   = "Idle — no QuadToNurbs run has been started yet"
        elif rc == 0:
            state = pb2.RunStatusResponse.DONE
            msg   = "Completed successfully"
        else:
            state = pb2.RunStatusResponse.ERROR
            msg   = f"Exited with return code {rc}"
        return pb2.RunStatusResponse(state=state, pid=pid, message=msg, return_code=rc or 0)

    def StreamQuadToNurbsOutput(self, request, context):
        for entry in _nurbs_runner.output_stream():
            if not context.is_active():
                return
            yield _entry_to_proto(entry)

    def DownloadNurbsFile(self, request, context):
        which  = (request.which  or "plus").lower().strip()
        fmt    = (request.format or "step").lower().strip()

        if which not in _NURBS_OBJ:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                          f"Unknown sheet {which!r}. Use 'plus' or 'minus'.")
            return
        if fmt == "obj":
            target = _NURBS_OBJ[which]
        elif fmt == "step":
            target = _NURBS_STEP[which]
        else:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                          f"Unknown format {fmt!r}. Use 'obj' or 'step'.")
            return

        if not target.exists():
            context.abort(grpc.StatusCode.NOT_FOUND,
                          f"{target.name} not found — run the relevant pipeline first")
            return
        try:
            yield from _stream_file(target, target.name)
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))


# ── Server bootstrap ──────────────────────────────────────────────────────────

def serve(host: str = "0.0.0.0", port: int = 50051) -> None:
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=16))
    pb2_grpc.add_GyroidOptimizerServicer_to_server(GyroidOptimizerServicer(), server)
    addr = f"{host}:{port}"
    server.add_insecure_port(addr)
    server.start()
    print(f"[gyroid-grpc] Listening on {addr}  (insecure)")
    print(f"[gyroid-grpc] Config : {CONFIG_PATH}")
    print(f"[gyroid-grpc] App dir: {APP_DIR}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n[gyroid-grpc] Shutting down …")
        server.stop(grace=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gyroid Optimizer gRPC server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=50051, help="Bind port (default: 50051)")
    args = parser.parse_args()
    serve(host=args.host, port=args.port)
