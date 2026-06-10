"""Example gRPC client for the Gyroid Optimizer server.

Run the server first:
    python grpc_server/server.py --port 50051 (this is done automatically if using run_container.ps1)

Then run this script:
    python client.py [--host localhost] [--port 50051] <command>

Commands
--------
  get-config                       Print current YAML config
  set-config <file.yaml>           Replace config from a local YAML file
  patch-config KEY=VALUE ...       Patch individual config keys using dot-notation.
                                   Values are parsed as JSON where possible
                                   (numbers, true/false, null), otherwise as strings.
                                   Examples:
                                     patch-config run.iters=30
                                     patch-config optimization.mode=heat
                                     patch-config optimization.no_overhang=false
                                     patch-config run.iters=10 optimization.mode=pressure
  start [--restart] [-- extra args …]  
                  Start a run (pass extra wrapper args after --). --restart makes it restartable from the latest checkpoint.
  stop                             Stop the running process
  status                           Show run state
  stream                           Tail the optimizer output (Ctrl-C to quit)
  history                          Dump the full optimization history
  latest                           Show the latest optimization metrics
  list-files [subpath]             List files in app/ (or a subdirectory)
  download <path> [out_dir]        Download a file from app/ to out_dir (default: .)
  download-app [out_dir]           Download the entire app/ folder as a .tar.gz
  stl-export  [--out] [--res] [--target-faces] [--fluid-stl] [-- extra args …]     
                                    Run gyroid_to_stl.py on the server (pass extra args after --)
                                    --out allows you to specify file name.
                                    --res sets the STL resolution (default: 0.4 mm)
                                    --target-faces sets the target number of faces (default: 5 million)
                                    --fluid-stl exports an additional STL of the fluid domain. This can be useful to analyse if the 
                                    fluid domain is continuous or blocked by a wall at some point.
  stl-stream                        Tail the STL export output (Ctrl-C to quit)
  stl-status                        Show STL export process state
  download-stl all|lattice|encap|surface [out_dir]
                                   Download generated STL file(s) from the server to out_dir (default: .)


  --- Gyroid → quad mesh ---
  quad-mesh [--out] [--angular] [--radius] [--distance] [--target-faces] [--no-boundary] [--crease] [--smooth] [--taubin-iter] [--weld-tol] [extra_args ...]       
                                    Run gyroid_to_quad_mesh_qf.py on the server
                                    --out allows you to specify file name (default: gyroid_implicit_qf.obj)
                                    --angular sets the CGAL angular criterion (default: 30°)
                                    --radius sets the CGAL radius criterion (default: 1.0 mm)
                                    --distance sets the CGAL distance criterion (default: 3.5 mm)
                                    --target-faces sets the target number of faces (default: 50000) for the triangular mesh, the quad mesh will have a different
                                    number of faces. This parameter (together with --radius) determines the resolution/size of the quad mesh.
                                    --no-boundary disables boundary surfaces
                                    --crease sets the crease angle (default: 25°)
                                    --smooth sets the smoothing iterations (default: 2)
                                    --taubin-iter sets the Taubin smoothing iterations (default: 10)
                                    --weld-tol sets the welding tolerance (default: 0.5 mm)
                                   Default output: gyroid_implicit_qf_plus/minus.obj
  quad-mesh-stop                   Stop the running quad-mesh process
  quad-mesh-status                 Show quad-mesh process state
  quad-mesh-stream                 Tail live quad-mesh output

  --- Quad mesh → NURBS STEP ---
  nurbs [plus|minus] [--subd] [--deg-min] [--deg-max] [--tol] [--sew-tol] [extra_args ...]  
                                        Run quad_to_nurbs.py for a gyroid sheet
                                       Default sheet: plus
                                       Default input : gyroid_implicit_qf_<sheet>.obj
                                       Default output: gyroid_implicit_qf_<sheet>.step
                                       --subd sets number of subdivisions for each quad (default: 1, i.e. no subdision)
                                        --deg-min sets the minimum degree of the NURBS surfaces (default: 3)
                                        --deg-max sets the maximum degree of the NURBS surfaces (default: 8)
                                        --tol sets the B-spline tolerance (default: 0.001 mm)
                                        --sew-tol sets the sewing tolerance for stitching NURBS surfaces together (default: 0.05 mm)

  nurbs-stop                       Stop the running quad-to-NURBS process
  nurbs-status                     Show quad-to-NURBS process state
  nurbs-stream                     Tail live quad-to-NURBS output
  download-nurbs [plus|minus] [obj|step] [out_dir]
                                   Download the OBJ (quad mesh) or STEP (NURBS) file
                                   Defaults: sheet=plus  format=step  out_dir=.

For the "extra args", go to the server source code in https://github.com/HuiLucas/MTO

"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import grpc

import gyroid_service_pb2 as pb2
import gyroid_service_pb2_grpc as pb2_grpc


def _channel(host: str, port: int) -> grpc.Channel:
    return grpc.insecure_channel(f"{host}:{port}")


def cmd_get_config(stub, _args):
    resp = stub.GetConfig(pb2.Empty())
    if resp.success:
        print(resp.yaml_content)
    else:
        print(f"ERROR: {resp.error}", file=sys.stderr)
        sys.exit(1)


def cmd_set_config(stub, args):
    path = Path(args.file)
    resp = stub.SetConfig(pb2.SetConfigRequest(yaml_content=path.read_text()))
    print(resp.message)
    if not resp.success:
        sys.exit(1)


def cmd_patch_config(stub, args):
    patch: dict = {}
    for assignment in args.assignments:
        if "=" not in assignment:
            print(f"ERROR: expected KEY=VALUE, got: {assignment!r}", file=sys.stderr)
            print("Example: patch-config run.iters=30 optimization.no_overhang=false",
                  file=sys.stderr)
            sys.exit(1)
        key, _, raw_val = assignment.partition("=")
        try:
            # Parses numbers, true/false, null correctly; falls back to string
            value = json.loads(raw_val)
        except json.JSONDecodeError:
            value = raw_val
        patch[key] = value
    resp = stub.PatchConfig(pb2.PatchConfigRequest(json_patch=json.dumps(patch)))
    print(resp.message)
    if not resp.success:
        sys.exit(1)


def cmd_start(stub, args):
    resp = stub.StartRun(pb2.StartRunRequest(extra_args=args.extra_args))
    print(resp.message)
    if not resp.success:
        sys.exit(1)


def cmd_stop(stub, _args):
    resp = stub.StopRun(pb2.Empty())
    print(resp.message)
    if not resp.success:
        sys.exit(1)


def cmd_status(stub, _args):
    resp = stub.GetRunStatus(pb2.Empty())
    state_name = pb2.RunStatusResponse.State.Name(resp.state)
    print(f"State      : {state_name}")
    print(f"PID        : {resp.pid or '—'}")
    print(f"Return code: {resp.return_code}")
    print(f"Message    : {resp.message}")
    print("To list commands use: python client.py help")



def cmd_stream(stub, _args):
    print("[streaming output — Ctrl-C to quit]\n")
    try:
        for line in stub.StreamOutput(pb2.Empty()):
            ts = time.strftime("%H:%M:%S", time.localtime(line.timestamp_ms / 1000))
            prefix = "ERR" if line.is_stderr else "OUT"
            print(f"[{ts}][{prefix}] {line.line}")
    except KeyboardInterrupt:
        print("\n[stream closed]")
    except grpc.RpcError as exc:
        print(f"gRPC error: {exc.details()}", file=sys.stderr)
        sys.exit(1)


def cmd_history(stub, _args):
    resp = stub.GetHistory(pb2.Empty())
    if not resp.success:
        print(f"ERROR: {resp.error}", file=sys.stderr)
        sys.exit(1)
    print("\t".join(resp.columns))
    for row in resp.rows:
        parts = []
        for col in resp.columns:
            if col in row.values:
                v = row.values[col]
                parts.append("nan" if math.isnan(v) else f"{v:g}")
            elif col in row.strings:
                parts.append(row.strings[col])
            else:
                parts.append("—")
        print("\t".join(parts))


def cmd_latest(stub, _args):
    resp = stub.GetLatestMetrics(pb2.Empty())
    if not resp.available:
        msg = resp.error or "No history yet"
        print(f"Not available: {msg}")
        return
    row = resp.latest
    all_keys = list(row.values.keys()) + list(row.strings.keys())
    for k in sorted(all_keys):
        if k in row.values:
            v = row.values[k]
            print(f"  {k:30s} {'nan' if math.isnan(v) else f'{v:g}'}")
        else:
            print(f"  {k:30s} {row.strings[k]}")


def cmd_list_files(stub, args):
    path = getattr(args, "subpath", "") or ""
    resp = stub.ListFiles(pb2.ListFilesRequest(path=path))
    if not resp.success:
        print(f"ERROR: {resp.error}", file=sys.stderr)
        sys.exit(1)
    for p in resp.paths:
        print(p)


def cmd_download(stub, args):
    rel_path = args.rel_path
    out_dir  = Path(args.out_dir) if hasattr(args, "out_dir") and args.out_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = stub.DownloadFile(pb2.DownloadRequest(path=rel_path, as_tar=False))
    _save_chunks(chunks, out_dir)


def cmd_download_app(stub, args):
    out_dir = Path(args.out_dir) if hasattr(args, "out_dir") and args.out_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = stub.DownloadFile(pb2.DownloadRequest(path="", as_tar=True))
    _save_chunks(chunks, out_dir)


def cmd_stl_export(stub, args):
    resp = stub.StartStlExport(pb2.StlExportRequest(extra_args=args.extra_args))
    print(resp.message)
    if not resp.success:
        sys.exit(1)


def cmd_stl_stop(stub, _args):
    resp = stub.StopStlExport(pb2.Empty())
    print(resp.message)
    if not resp.success:
        sys.exit(1)


def cmd_stl_status(stub, _args):
    resp = stub.GetStlStatus(pb2.Empty())
    state_name = pb2.RunStatusResponse.State.Name(resp.state)
    print(f"State      : {state_name}")
    print(f"PID        : {resp.pid or '—'}")
    print(f"Return code: {resp.return_code}")
    print(f"Message    : {resp.message}")


def cmd_stl_stream(stub, _args):
    print("[streaming STL export output — Ctrl-C to quit]\n")
    try:
        for line in stub.StreamStlOutput(pb2.Empty()):
            ts = time.strftime("%H:%M:%S", time.localtime(line.timestamp_ms / 1000))
            print(f"[{ts}] {line.line}")
    except KeyboardInterrupt:
        print("\n[stream closed]")
    except grpc.RpcError as exc:
        print(f"gRPC error: {exc.details()}", file=sys.stderr)
        sys.exit(1)


def cmd_download_stl(stub, args):
    which   = getattr(args, "which", "lattice") or "lattice"
    out_dir = Path(args.out_dir) if hasattr(args, "out_dir") and args.out_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = stub.DownloadStl(pb2.StlFileRequest(which=which))
    _save_chunks(chunks, out_dir)



def cmd_quad_mesh(stub, args):
    resp = stub.StartGyroidToQuadMesh(pb2.QuadMeshRequest(extra_args=args.extra_args))
    print(resp.message)
    if not resp.success:
        sys.exit(1)


def cmd_quad_mesh_stop(stub, _args):
    resp = stub.StopGyroidToQuadMesh(pb2.Empty())
    print(resp.message)
    if not resp.success:
        sys.exit(1)


def cmd_quad_mesh_status(stub, _args):
    resp = stub.GetGyroidToQuadMeshStatus(pb2.Empty())
    state_name = pb2.RunStatusResponse.State.Name(resp.state)
    print(f"State      : {state_name}")
    print(f"PID        : {resp.pid or '—'}")
    print(f"Return code: {resp.return_code}")
    print(f"Message    : {resp.message}")


def cmd_quad_mesh_stream(stub, _args):
    print("[streaming quad-mesh output — Ctrl-C to quit]\n")
    try:
        for line in stub.StreamGyroidToQuadMeshOutput(pb2.Empty()):
            ts = time.strftime("%H:%M:%S", time.localtime(line.timestamp_ms / 1000))
            print(f"[{ts}] {line.line}")
    except KeyboardInterrupt:
        print("\n[stream closed]")
    except grpc.RpcError as exc:
        print(f"gRPC error: {exc.details()}", file=sys.stderr)
        sys.exit(1)


def cmd_nurbs(stub, args):
    resp = stub.StartQuadToNurbs(pb2.QuadToNurbsRequest(
        which=args.which,
        extra_args=args.extra_args,
    ))
    print(resp.message)
    if not resp.success:
        sys.exit(1)


def cmd_nurbs_stop(stub, _args):
    resp = stub.StopQuadToNurbs(pb2.Empty())
    print(resp.message)
    if not resp.success:
        sys.exit(1)


def cmd_nurbs_status(stub, _args):
    resp = stub.GetQuadToNurbsStatus(pb2.Empty())
    state_name = pb2.RunStatusResponse.State.Name(resp.state)
    print(f"State      : {state_name}")
    print(f"PID        : {resp.pid or '—'}")
    print(f"Return code: {resp.return_code}")
    print(f"Message    : {resp.message}")


def cmd_nurbs_stream(stub, _args):
    print("[streaming quad-to-NURBS output — Ctrl-C to quit]\n")
    try:
        for line in stub.StreamQuadToNurbsOutput(pb2.Empty()):
            ts = time.strftime("%H:%M:%S", time.localtime(line.timestamp_ms / 1000))
            print(f"[{ts}] {line.line}")
    except KeyboardInterrupt:
        print("\n[stream closed]")
    except grpc.RpcError as exc:
        print(f"gRPC error: {exc.details()}", file=sys.stderr)
        sys.exit(1)


def cmd_download_nurbs(stub, args):
    out_dir = Path(args.out_dir) if hasattr(args, "out_dir") and args.out_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = stub.DownloadNurbsFile(pb2.NurbsFileRequest(
        which=args.which,
        format=args.format,
    ))
    _save_chunks(chunks, out_dir)


def _save_chunks(stream, out_dir: Path) -> None:
    out_path: Path | None = None
    fh = None
    received = 0
    try:
        for chunk in stream:
            if out_path is None:
                out_path = out_dir / chunk.filename
                fh = open(out_path, "wb")
                total = chunk.total_size
                print(f"Saving {chunk.filename} ({total:,} bytes) → {out_path}")
            fh.write(chunk.data)
            received += len(chunk.data)
            pct = 100 * received / max(total, 1)
            print(f"\r  {received:,} / {total:,} bytes  ({pct:.0f}%)", end="", flush=True)
    finally:
        if fh:
            fh.close()
    print(f"\nDone: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Gyroid Optimizer gRPC client")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("get-config")

    p = sub.add_parser("set-config")
    p.add_argument("file")

    p = sub.add_parser("patch-config",
                       help="Patch config keys with KEY=VALUE pairs (no shell quoting needed)")
    p.add_argument("assignments", nargs="+", metavar="KEY=VALUE",
                   help="Dot-notation key=value, e.g. run.iters=30  optimization.no_overhang=false")

    p = sub.add_parser("start")
    p.add_argument("extra_args", nargs="*")

    sub.add_parser("help",         help="Print full usage explanation and exit")
    sub.add_parser("stop")
    sub.add_parser("status")
    sub.add_parser("stream")
    sub.add_parser("history")
    sub.add_parser("latest")

    p = sub.add_parser("list-files")
    p.add_argument("subpath", nargs="?", default="")

    p = sub.add_parser("download")
    p.add_argument("rel_path")
    p.add_argument("out_dir", nargs="?", default=".")

    p = sub.add_parser("download-app")
    p.add_argument("out_dir", nargs="?", default=".")

    p = sub.add_parser("stl-export",
                       help="Run gyroid_to_stl.py on the server")
    p.add_argument("extra_args", nargs="*",
                   help="Extra flags forwarded to gyroid_to_stl.py, "
                        "e.g. --res 0.015 --mirror-y")

    sub.add_parser("stl-stop",   help="Stop the running STL export")
    sub.add_parser("stl-status", help="Show STL export process state")
    sub.add_parser("stl-stream", help="Tail live STL export output")

    p = sub.add_parser("download-stl",
                       help="Download a generated STL file from the server")
    p.add_argument("which", nargs="?", default="lattice",
                   choices=["lattice", "encap", "surface", "all"],
                   help="lattice (default), encap, surface, or all (→ .tar.gz)")
    p.add_argument("out_dir", nargs="?", default=".")

    
    p = sub.add_parser("quad-mesh",
                       help="Run gyroid_to_quad_mesh_qf.py on the server")
    p.add_argument("extra_args", nargs="*",
                   help="Extra flags forwarded to gyroid_to_quad_mesh_qf.py, "
                        "e.g. --target-faces 30000 --backend quadriflow")

    sub.add_parser("quad-mesh-stop",   help="Stop the running quad-mesh process")
    sub.add_parser("quad-mesh-status", help="Show quad-mesh process state")
    sub.add_parser("quad-mesh-stream", help="Tail live quad-mesh output")

    p = sub.add_parser("nurbs",
                       help="Run quad_to_nurbs.py for a gyroid sheet (plus or minus)")
    p.add_argument("which", nargs="?", default="plus",
                   choices=["plus", "minus"],
                   help="Gyroid sheet to convert (default: plus)")
    p.add_argument("extra_args", nargs="*",
                   help="Extra flags forwarded to quad_to_nurbs.py, "
                        "e.g. --subd 2 --tol 0.001")

    sub.add_parser("nurbs-stop",   help="Stop the running quad-to-NURBS process")
    sub.add_parser("nurbs-status", help="Show quad-to-NURBS process state")
    sub.add_parser("nurbs-stream", help="Tail live quad-to-NURBS output")

    p = sub.add_parser("download-nurbs",
                       help="Download OBJ (quad mesh) or STEP (NURBS) file from the server")
    p.add_argument("which", nargs="?", default="plus",
                   choices=["plus", "minus"],
                   help="Gyroid sheet (default: plus)")
    p.add_argument("format", nargs="?", default="step",
                   choices=["obj", "step"],
                   help="File format: obj (quad mesh) or step (NURBS, default)")
    p.add_argument("out_dir", nargs="?", default=".")

    args = parser.parse_args()

    if args.command == "help":
        print(__doc__)
        return

    with _channel(args.host, args.port) as channel:
        stub = pb2_grpc.GyroidOptimizerStub(channel)
        dispatch = {
           "get-config":         cmd_get_config,
            "set-config":         cmd_set_config,
            "patch-config":       cmd_patch_config,
            "start":              cmd_start,
            "stop":               cmd_stop,
            "status":             cmd_status,
            "stream":             cmd_stream,
            "history":            cmd_history,
            "latest":             cmd_latest,
            "list-files":         cmd_list_files,
            "download":           cmd_download,
            "download-app":       cmd_download_app,
            "stl-export":         cmd_stl_export,
            "stl-stop":           cmd_stl_stop,
            "stl-status":         cmd_stl_status,
            "stl-stream":         cmd_stl_stream,
            "download-stl":       cmd_download_stl,
            "quad-mesh":          cmd_quad_mesh,
            "quad-mesh-stop":     cmd_quad_mesh_stop,
            "quad-mesh-status":   cmd_quad_mesh_status,
            "quad-mesh-stream":   cmd_quad_mesh_stream,
            "nurbs":              cmd_nurbs,
            "nurbs-stop":         cmd_nurbs_stop,
            "nurbs-status":       cmd_nurbs_status,
            "nurbs-stream":       cmd_nurbs_stream,
            "download-nurbs":     cmd_download_nurbs,
        }
        dispatch[args.command](stub, args)


if __name__ == "__main__":
    main()
