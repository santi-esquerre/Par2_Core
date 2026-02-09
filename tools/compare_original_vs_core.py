#!/usr/bin/env python3
"""
compare_original_vs_core.py - Compare outputs between original PAR2 and Par2_Core legacy runner.

Usage:
    python3 compare_original_vs_core.py \
        --yaml path/to/config.yaml \
        --par2_original path/to/par2_original \
        --par2_core path/to/par2core_legacy_runner \
        --workdir /tmp/par2_compare

Exit codes:
    0 = All comparisons passed
    1 = Comparison failed (mismatch detected)
    2 = Execution error (binary failed to run, missing files, etc.)
"""

import argparse
import copy
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import yaml
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare PAR2 original vs Par2_Core legacy runner outputs"
    )
    parser.add_argument("--yaml", required=True, help="Path to YAML configuration file")
    parser.add_argument(
        "--par2_original", required=True, help="Path to par2_original executable"
    )
    parser.add_argument(
        "--par2_core", required=True, help="Path to par2core_legacy_runner executable"
    )
    parser.add_argument(
        "--workdir", default=None, help="Working directory (temp if not specified)"
    )
    parser.add_argument(
        "--keep-workdir", action="store_true", help="Keep workdir after comparison"
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for float comparison",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for float comparison",
    )
    parser.add_argument(
        "--stat-rtol",
        type=float,
        default=0.20,
        help="Relative tolerance for statistical comparison (mean/std)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Override seed in YAML for determinism"
    )
    parser.add_argument(
        "--par2-core-args",
        default="",
        help="Extra arguments to pass to par2core_legacy_runner (e.g., '--legacy-strict')",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    return parser.parse_args()


def load_yaml_config(yaml_path: str) -> dict:
    """Load YAML configuration file."""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def save_yaml_config(config: dict, yaml_path: str):
    """Save YAML configuration file."""
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def get_input_files(yaml_path: str, config: dict) -> List[str]:
    """Get list of input files referenced by the YAML config."""
    yaml_dir = Path(yaml_path).parent
    files = [Path(yaml_path).name]  # The YAML itself

    # Velocity file
    if "physics" in config and "velocity" in config["physics"]:
        vel_file = config["physics"]["velocity"].get("file")
        if vel_file:
            files.append(vel_file)

    return files


def get_expected_outputs(config: dict) -> Tuple[Optional[str], List[Tuple[str, int]]]:
    """
    Parse expected outputs from config.

    Returns:
        (csv_file, [(snapshot_pattern, step), ...])
    """
    csv_file = None
    snapshots = []

    if "output" in config:
        out_cfg = config["output"]

        # CSV output
        if "csv" in out_cfg:
            csv_file = out_cfg["csv"].get("file")

        # Snapshot output
        if "snapshot" in out_cfg:
            snap_cfg = out_cfg["snapshot"]
            snap_file = snap_cfg.get("file")
            if snap_file:
                if "steps" in snap_cfg:
                    for step in snap_cfg["steps"]:
                        snapshots.append((snap_file.replace("*", str(step)), step))
                elif "skip" in snap_cfg:
                    skip = snap_cfg["skip"]
                    steps = config.get("simulation", {}).get("steps", 0)
                    for step in range(0, steps + 1, skip):
                        snapshots.append((snap_file.replace("*", str(step)), step))

    return csv_file, snapshots


def setup_case_dir(
    workdir: Path,
    case_name: str,
    yaml_path: str,
    config: dict,
    input_files: List[str],
    seed: Optional[int] = None,
) -> Path:
    """
    Set up a case directory with all necessary files.

    Returns path to the copied YAML file.
    """
    case_dir = workdir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    yaml_dir = Path(yaml_path).parent
    yaml_name = Path(yaml_path).name

    # Copy input files
    for f in input_files:
        src = yaml_dir / f
        dst = case_dir / f
        if src.exists():
            # Create parent directory if needed (for subpaths like tmp/model.ftl)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    # Create output directory if referenced
    if "output" in config:
        for key in ["csv", "snapshot"]:
            if key in config["output"]:
                out_file = config["output"][key].get("file", "")
                out_dir = Path(out_file).parent
                if out_dir and str(out_dir) != ".":
                    (case_dir / out_dir).mkdir(parents=True, exist_ok=True)

    # Optionally override seed
    if seed is not None:
        if "simulation" not in config:
            config["simulation"] = {}
        config["simulation"]["seed"] = seed

    # Save modified YAML
    case_yaml = case_dir / yaml_name
    save_yaml_config(config, str(case_yaml))

    return case_yaml


def run_executable(
    exe_path: str, yaml_path: str, extra_args: List[str] = None, verbose: bool = False
) -> Tuple[int, str, str]:
    """Run an executable and return (exit_code, stdout, stderr)."""
    try:
        cmd = [exe_path, yaml_path]
        if extra_args:
            cmd.extend(extra_args)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        if verbose:
            print(
                f"  stdout: {result.stdout[:500]}..."
                if len(result.stdout) > 500
                else f"  stdout: {result.stdout}"
            )
            if result.stderr:
                print(f"  stderr: {result.stderr}")
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Timeout expired"
    except Exception as e:
        return -1, "", str(e)


def load_csv(path: str) -> Tuple[List[str], np.ndarray]:
    """
    Load a CSV file and return (headers, data_array).

    Handles the "step, time, ..." format.
    """
    with open(path, "r") as f:
        header_line = f.readline().strip()
        headers = [h.strip() for h in header_line.split(",")]

        data = []
        for line in f:
            line = line.strip()
            if line:
                values = [float(v.strip()) for v in line.split(",")]
                data.append(values)

        return headers, np.array(data)


def load_snapshot(path: str) -> np.ndarray:
    """Load a snapshot file (CSV format: id,x coord,y coord,z coord)."""
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1, 2, 3))
    return data


def compare_csv(
    path1: str,
    path2: str,
    atol: float,
    rtol: float,
    stat_rtol: float = 0.20,
    verbose: bool = False,
) -> Tuple[bool, str]:
    """
    Compare two CSV files.

    Returns:
        (passed, message)
    """
    try:
        headers1, data1 = load_csv(path1)
        headers2, data2 = load_csv(path2)
    except Exception as e:
        return False, f"Failed to load CSV files: {e}"

    # Check headers
    if headers1 != headers2:
        return False, f"Header mismatch:\n  Original: {headers1}\n  Core: {headers2}"

    # Check shape
    if data1.shape != data2.shape:
        return False, f"Shape mismatch: {data1.shape} vs {data2.shape}"

    # Try exact comparison first
    if np.allclose(data1, data2, atol=atol, rtol=rtol):
        return True, f"CSV exact match: {data1.shape[0]} rows, {data1.shape[1]} columns"

    # Fall back to statistical comparison (for stochastic simulations)
    # Compare each column's statistics
    all_close = True
    details = []
    stats_details = []

    for col in range(data1.shape[1]):
        col_name = headers1[col] if col < len(headers1) else f"col{col}"
        mean1, mean2 = np.mean(data1[:, col]), np.mean(data2[:, col])
        std1, std2 = np.std(data1[:, col]), np.std(data2[:, col])

        # Skip "step" and "time" columns (should match exactly)
        if col_name.lower() in ["step", "time"]:
            if not np.allclose(data1[:, col], data2[:, col], atol=atol, rtol=rtol):
                return False, f"Column '{col_name}' should match exactly but differs"
            continue

        # Statistical comparison for metric columns
        mean_diff = abs(mean1 - mean2) / (abs(mean1) + 1e-10)
        std_diff = abs(std1 - std2) / (std1 + 1e-10)

        # Add detailed stats for this column
        stats_details.append(
            f"    {col_name}:\n"
            f"      Original: mean={mean1:.6f}, std={std1:.6f}\n"
            f"      Core:     mean={mean2:.6f}, std={std2:.6f}\n"
            f"      Rel err:  mean={mean_diff*100:.2f}%, std={std_diff*100:.2f}%"
        )

        if mean_diff > stat_rtol:
            all_close = False
            details.append(
                f"  {col_name}: mean {mean1:.4f} vs {mean2:.4f} (diff: {mean_diff:.2%})"
            )

        if std_diff > stat_rtol:
            all_close = False
            details.append(
                f"  {col_name}: std {std1:.4f} vs {std2:.4f} (diff: {std_diff:.2%})"
            )

    stats_msg = "\n".join(stats_details)

    if all_close:
        return (
            True,
            f"CSV statistical match: {data1.shape[0]} rows, {data1.shape[1]} columns\n{stats_msg}",
        )
    else:
        return (
            False,
            "CSV statistical mismatch:\n" + "\n".join(details) + f"\n{stats_msg}",
        )


def compare_snapshots_exact(
    path1: str, path2: str, atol: float, rtol: float, verbose: bool = False
) -> Tuple[bool, str]:
    """Compare two snapshot files particle-by-particle."""
    try:
        data1 = load_snapshot(path1)
        data2 = load_snapshot(path2)
    except Exception as e:
        return False, f"Failed to load snapshot files: {e}"

    if data1.shape != data2.shape:
        return False, f"Shape mismatch: {data1.shape} vs {data2.shape}"

    if np.allclose(data1, data2, atol=atol, rtol=rtol):
        return True, f"Snapshot exact match: {data1.shape[0]} particles"

    # Find mismatch details
    diff = np.abs(data1 - data2)
    max_diff = np.max(diff)
    max_idx = np.unravel_index(np.argmax(diff), diff.shape)

    return False, (
        f"Snapshot particle mismatch (max diff: {max_diff:.2e})\n"
        f"  Particle {max_idx[0]}, coord {max_idx[1]}: "
        f"{data1[max_idx]:.6f} vs {data2[max_idx]:.6f}"
    )


def compare_snapshots_statistical(
    path1: str, path2: str, moment_rtol: float = 0.05, verbose: bool = False
) -> Tuple[bool, str]:
    """
    Compare two snapshot files statistically (mean, std).

    Useful for stochastic simulations where exact match is not expected.
    """
    try:
        data1 = load_snapshot(path1)
        data2 = load_snapshot(path2)
    except Exception as e:
        return False, f"Failed to load snapshot files: {e}"

    if data1.shape[0] != data2.shape[0]:
        return False, f"Particle count mismatch: {data1.shape[0]} vs {data2.shape[0]}"

    # Compare moments
    mean1, mean2 = np.mean(data1, axis=0), np.mean(data2, axis=0)
    std1, std2 = np.std(data1, axis=0), np.std(data2, axis=0)

    axis_names = ["X", "Y", "Z"]

    # Build detailed stats message
    stats_lines = [
        f"    Particles: {data1.shape[0]}",
        f"    Original - Mean: [{mean1[0]:.4f}, {mean1[1]:.4f}, {mean1[2]:.4f}]",
        f"    Core     - Mean: [{mean2[0]:.4f}, {mean2[1]:.4f}, {mean2[2]:.4f}]",
        f"    Original - Std:  [{std1[0]:.4f}, {std1[1]:.4f}, {std1[2]:.4f}]",
        f"    Core     - Std:  [{std2[0]:.4f}, {std2[1]:.4f}, {std2[2]:.4f}]",
    ]

    # Compute relative errors
    mean_diff = np.abs(mean1 - mean2) / (np.abs(mean1) + 1e-10)
    std_diff = np.abs(std1 - std2) / (std1 + 1e-10)

    stats_lines.append(
        f"    Mean rel err: [{mean_diff[0]:.4f}, {mean_diff[1]:.4f}, {mean_diff[2]:.4f}] ({mean_diff[0]*100:.2f}%, {mean_diff[1]*100:.2f}%, {mean_diff[2]*100:.2f}%)"
    )
    stats_lines.append(
        f"    Std  rel err: [{std_diff[0]:.4f}, {std_diff[1]:.4f}, {std_diff[2]:.4f}] ({std_diff[0]*100:.2f}%, {std_diff[1]*100:.2f}%, {std_diff[2]*100:.2f}%)"
    )

    stats_msg = "\n".join(stats_lines)

    # Check means with relative tolerance
    if np.any(mean_diff > moment_rtol):
        worst_axis = np.argmax(mean_diff)
        return False, (
            f"Mean mismatch on axis {axis_names[worst_axis]}: "
            f"{mean1[worst_axis]:.4f} vs {mean2[worst_axis]:.4f} "
            f"(rel diff: {mean_diff[worst_axis]:.4f})\n{stats_msg}"
        )

    # Check std with relative tolerance
    if np.any(std_diff > moment_rtol):
        worst_axis = np.argmax(std_diff)
        return False, (
            f"Std mismatch on axis {axis_names[worst_axis]}: "
            f"{std1[worst_axis]:.4f} vs {std2[worst_axis]:.4f} "
            f"(rel diff: {std_diff[worst_axis]:.4f})\n{stats_msg}"
        )

    msg = f"Snapshot statistical match\n{stats_msg}"

    return True, msg


def main():
    args = parse_args()

    # Validate inputs
    if not os.path.exists(args.yaml):
        print(f"ERROR: YAML file not found: {args.yaml}")
        return 2

    if not os.path.exists(args.par2_original):
        print(f"ERROR: par2_original not found: {args.par2_original}")
        return 2

    if not os.path.exists(args.par2_core):
        print(f"ERROR: par2_core not found: {args.par2_core}")
        return 2

    # Load configuration
    config = load_yaml_config(args.yaml)
    input_files = get_input_files(args.yaml, config)
    csv_file, snapshots = get_expected_outputs(config)

    print(f"=== PAR2 Comparison: {Path(args.yaml).name} ===")
    print(f"  CSV output: {csv_file}")
    print(f"  Snapshots: {len(snapshots)} files")

    # Setup workdir
    if args.workdir:
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        workdir = Path(tempfile.mkdtemp(prefix="par2_compare_"))

    print(f"  Workdir: {workdir}")

    try:
        # Setup case directories
        print("\n--- Setting up cases ---")
        orig_yaml = setup_case_dir(
            workdir,
            "orig_case",
            args.yaml,
            copy.deepcopy(config),
            input_files,
            seed=args.seed,
        )
        core_yaml = setup_case_dir(
            workdir,
            "core_case",
            args.yaml,
            copy.deepcopy(config),
            input_files,
            seed=args.seed,
        )

        # Run original
        print("\n--- Running par2_original ---")
        ret1, out1, err1 = run_executable(
            args.par2_original, str(orig_yaml), verbose=args.verbose
        )
        if ret1 != 0:
            print(f"ERROR: par2_original failed with exit code {ret1}")
            print(f"  stderr: {err1}")
            return 2
        print("  Completed successfully")

        # Run core
        print("\n--- Running par2core_legacy_runner ---")
        core_extra_args = args.par2_core_args.split() if args.par2_core_args else None
        ret2, out2, err2 = run_executable(
            args.par2_core,
            str(core_yaml),
            extra_args=core_extra_args,
            verbose=args.verbose,
        )
        if ret2 != 0:
            print(f"ERROR: par2core_legacy_runner failed with exit code {ret2}")
            print(f"  stderr: {err2}")
            return 2
        print("  Completed successfully")

        # Compare outputs
        print("\n--- Comparing outputs ---")
        all_passed = True

        # Compare CSV
        if csv_file:
            orig_csv = workdir / "orig_case" / csv_file
            core_csv = workdir / "core_case" / csv_file

            if not orig_csv.exists():
                print(f"  CSV: SKIP (original not found: {orig_csv})")
            elif not core_csv.exists():
                print(f"  CSV: FAIL (core not found: {core_csv})")
                all_passed = False
            else:
                passed, msg = compare_csv(
                    str(orig_csv),
                    str(core_csv),
                    args.atol,
                    args.rtol,
                    args.stat_rtol,
                    args.verbose,
                )
                status = "PASS" if passed else "FAIL"
                print(f"  CSV: {status}")
                # Always show detailed stats
                for line in msg.split("\n"):
                    print(f"    {line}")
                if not passed:
                    all_passed = False

        # Compare snapshots
        for snap_file, step in snapshots:
            orig_snap = workdir / "orig_case" / snap_file
            core_snap = workdir / "core_case" / snap_file

            if not orig_snap.exists():
                print(f"  Snapshot step {step}: SKIP (original not found)")
                continue
            if not core_snap.exists():
                print(f"  Snapshot step {step}: FAIL (core not found)")
                all_passed = False
                continue

            # Try exact match first
            passed, msg = compare_snapshots_exact(
                str(orig_snap), str(core_snap), args.atol, args.rtol, args.verbose
            )
            if passed:
                print(f"  Snapshot step {step}: PASS (exact)")
            else:
                # Fall back to statistical comparison
                passed, msg = compare_snapshots_statistical(
                    str(orig_snap),
                    str(core_snap),
                    moment_rtol=args.stat_rtol,
                    verbose=args.verbose,
                )
                status = "PASS (statistical)" if passed else "FAIL"
                print(f"  Snapshot step {step}: {status}")
                # Always show detailed stats
                for line in msg.split("\n"):
                    print(f"    {line}")
                if not passed:
                    all_passed = False

        # Summary
        print("\n=== Summary ===")
        if all_passed:
            print("All comparisons PASSED")
            return 0
        else:
            print("Some comparisons FAILED")
            return 1

    finally:
        # Cleanup
        if not args.keep_workdir and args.workdir is None:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
