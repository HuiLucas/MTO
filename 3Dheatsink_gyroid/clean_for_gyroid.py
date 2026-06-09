"""
clean_for_gyroid.py — Reset the 3D heat-sink case for a fresh Gyroid RBF run.

Removes:
  • All numeric time directories except 0 (covers both SIMP/MMA runs such as
    10, 20, … and gyroid-optimizer iterations such as 1, 2, …) from both the
    case root and every processor* subdirectory.
  • Monitoring files written by the solver: meanT.txt, Disspower.txt,
    Voluse.txt, Time.txt.
  • Optimizer state files: cell_centers_mm.npy, gyroid_ctrl_pts*.txt,
    gyroid_opt_history.txt.
  • The app/0/gamma file written by the Gyroid optimizer, so gamma reverts to
    the default uniform value (voluse) on the next run.
  • Resets controlDict to startTime=0 / endTime=400 / writeInterval=10.

Keeps:
  • app/0/  — all original initial-condition fields (p, U, T, Tb, …)
  • app/constant/ and app/system/
  • app/processor*/0/ and app/processor*/constant/
  • latest_fluid_state/ backup directory (if present)

Not removed (written by solver per-run, accumulate across runs):
  • massflow.txt, deltaP.txt, outletT.txt, alphaMax.txt

Usage:
    python clean_for_gyroid.py [--case app] [--dry-run]
"""

import argparse
import shutil
from pathlib import Path

KEEP_TIMES = {0}          # time directories to keep in the case root
REMOVE_OPTIMIZER_FILES = [
    'meanT.txt', 'Disspower.txt', 'Voluse.txt', 'Time.txt',
    'gyroid_opt_history.txt', 'cell_centers_mm.npy',
]


def numeric_time_dirs(directory: Path) -> list[Path]:
    """Return all subdirectories whose name is a non-negative integer."""
    result = []
    for d in directory.iterdir():
        if d.is_dir():
            try:
                t = int(d.name)
                if t >= 0:
                    result.append(d)
            except ValueError:
                pass
    return sorted(result, key=lambda p: int(p.name))


def clean_case(case_dir: Path, dry_run: bool) -> None:
    def remove(p: Path, reason: str) -> None:
        if dry_run:
            print(f"  [dry-run] would remove  {p}  ({reason})")
        else:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink(missing_ok=True)
            print(f"  removed  {p}  ({reason})")

    # ── 1. Case-root time directories ───────────────────────────────────────
    print("\n[1] Case-root time directories")
    for td in numeric_time_dirs(case_dir):
        t = int(td.name)
        if t not in KEEP_TIMES:
            remove(td, f"time {t}")

    # Also remove app/0/gamma written by the Gyroid optimizer so the solver
    # falls back to the uniform initial value (voluse = 0.2).
    gamma0 = case_dir / '0' / 'gamma'
    if gamma0.exists():
        remove(gamma0, "Gyroid-written gamma in time 0")

    # ── 2. Processor subdirectory time steps ────────────────────────────────
    print("\n[2] Processor time directories")
    proc_dirs = sorted(
        [d for d in case_dir.iterdir()
         if d.is_dir() and d.name.startswith('processor')],
        key=lambda p: int(p.name.replace('processor', ''))
    )
    for proc in proc_dirs:
        for td in numeric_time_dirs(proc):
            t = int(td.name)
            if t not in KEEP_TIMES:
                remove(td, f"{proc.name}/time {t}")

    # ── 3. Monitoring and optimizer files ────────────────────────────────────
    print("\n[3] Monitoring and optimizer files")
    for fname in REMOVE_OPTIMIZER_FILES:
        p = case_dir / fname
        if p.exists():
            remove(p, "monitoring/optimizer file")

    # Remove gyroid_ctrl_pts_*.txt files (wildcard)
    for p in case_dir.glob('gyroid_ctrl_pts*.txt'):
        remove(p, "ctrl pts file")

    # ── 4. Reset controlDict to original settings ────────────────────────────
    print("\n[4] Resetting controlDict")
    cd_path = case_dir / 'system' / 'controlDict'
    if cd_path.exists():
        import re
        text = cd_path.read_text()
        text = re.sub(r'(startTime\s+)\S+;',    r'\g<1>0;',   text)
        text = re.sub(r'(endTime\s+)\S+;',       r'\g<1>400;', text)
        text = re.sub(r'(writeInterval\s+)\S+;', r'\g<1>10;',  text)
        if not dry_run:
            cd_path.write_text(text)
            print(f"  controlDict reset: startTime=0, endTime=400, writeInterval=10")
        else:
            print(f"  [dry-run] would reset controlDict: startTime=0 endTime=400 writeInterval=10")

    print("\nDone." if not dry_run else "\nDry-run complete — nothing was deleted.")


def main() -> None:
    parser = argparse.ArgumentParser(description='Reset case for a fresh Gyroid RBF run.')
    parser.add_argument('--case',    default='app', help='Path to OpenFOAM case (default: app)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be deleted without doing it')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    case_dir   = (script_dir / args.case).resolve()

    if not case_dir.is_dir():
        raise SystemExit(f"ERROR: case directory not found: {case_dir}")

    mode = "DRY RUN" if args.dry_run else "LIVE"
    print(f"=== clean_for_gyroid [{mode}]  case: {case_dir} ===")
    clean_case(case_dir, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
