# SPDX-License-Identifier: MIT AND CC-BY-4.0
# The solver-derived loop retains CC-BY-4.0; see THIRD_PARTY_NOTICES.md.

"""CUDA companion to the unchanged CPU Mittag-Leffler comparison.

The scalar test, reference time convention and gradient mode are retained.
Coelho's solve loop is copied below with only its initial buffer placed on the
input device. This is a device port, not a parallel rewrite of that algorithm.
GPU timings exclude input/output transfers and use explicit synchronization.
CPU result files are never written.
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from Constants import EVALUATION_SOURCE, NEW_RESULTS_PATH, OUTPUT_PATH
from src.models.FDEint.FixedStepFDEint_with_forcing import FDEint as forced_fde_solver
from src.models.FDEint import Original_FDE_Solver as coelho_reference
from src.models.FDEint.mittag_leffler import ml as mittag_leffler

DEFAULT_RESULTS_DIR = NEW_RESULTS_PATH / "solver/zero_forcing"
ORIGINAL_SOLVER_NAME = "Original Solver"
FDEINT_SOLVER_NAME = "FDEint"
LEGACY_FORCED_SOLVER_NAME = "Forced Solver ($u=0$)"
LEGACY_THIS_WORK_SOLVER_NAME = "This Work"
FDEINT_DISPLAY_NAME = "\\texttt{FDEint}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fractional_rhs_original(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -y


class ZeroForcedRelaxation(nn.Module):
    """D_C^alpha y(t) = -y(t) + u(t), evaluated with u(t)=0."""

    def forward(self, t: torch.Tensor, y: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        if u_t.dim() == 1:
            u_t = u_t.unsqueeze(-1)
        return -y + u_t


def reference_solution(t: torch.Tensor, alpha: float) -> np.ndarray:
    t_np = t.squeeze().detach().cpu().numpy()
    return np.asarray([mittag_leffler(-(float(ti) ** alpha), alpha) for ti in t_np], dtype=np.float64)


def compute_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = reference - prediction
    abs_error = np.abs(error)
    rmse = float(np.sqrt(np.mean(error**2)))
    return {
        "mae": float(np.mean(abs_error)),
        "mse": float(np.mean(error**2)),
        "rmse": rmse,
        "max_abs_error": float(np.max(abs_error)),
        "nrmse_range": float(rmse / (np.max(reference) - np.min(reference))),
    }



def original_fde_solver_device_compatible(a, f, y0, tspan):
    """Coelho's solve loop with just the initial y buffer allocated on y0.device.

    Original predictor/right helpers, scalar conversions, float32 temporary,
    autograd behavior and output indexing are intentionally unchanged.
    """
    t0 = tspan[0]
    h = tspan[1] - t0
    N = len(tspan)
    y = torch.zeros((1, 1), device=y0.device)  # The only device-port change.
    device = y0.device

    for n in range(0, N):
        temp = torch.zeros(1, dtype=torch.float, requires_grad=True).to(device)
        for k in range(0, int(torch.ceil(a))):
            temp.data += y0 * (tspan[n])**k / math.factorial(k)

        y_p = coelho_reference.predictor(f, y, a, n, h, y0, t0, tspan)
        y_new = temp + h**a / math.gamma(a+2) * (
            f(tspan[n], y_p) + coelho_reference.right(f, y, a, n, h)
        ).unsqueeze(1)
        if n == 0:
            y = y_new
        else:
            y = torch.cat([y, y_new], dim=0)
    return y


def run_original_solver(t: torch.Tensor, y0: torch.Tensor, alpha: float) -> torch.Tensor:
    alpha_tensor = torch.tensor([alpha], dtype=torch.float64, device=t.device)
    y = original_fde_solver_device_compatible(alpha_tensor, fractional_rhs_original, y0, t)
    return y.squeeze()


def run_forced_solver_with_zero_input(t: torch.Tensor, y0: torch.Tensor, alpha: float) -> torch.Tensor:
    alpha_tensor = torch.tensor([alpha], dtype=torch.float64, device=t.device)
    u_zero = torch.zeros(t.squeeze().shape, dtype=torch.float64, device=t.device)
    solver = ZeroForcedRelaxation()
    y = forced_fde_solver(
        f=solver,
        t=t.squeeze(),
        y0=y0,
        alpha=alpha_tensor,
        h=None,
        dtype=torch.float64,
        DEBUG=False,
        return_internals=False,
        u=u_zero,
        u_fn=None,
    )
    return y.squeeze()


def as_numpy(y: torch.Tensor) -> np.ndarray:
    return y.detach().cpu().numpy().astype(np.float64)


def time_solver(fn, t, y0, alpha) -> tuple[np.ndarray, float]:
    gc.collect()
    torch.cuda.synchronize(t.device)
    start = time.perf_counter()
    prediction = fn(t, y0, alpha)
    torch.cuda.synchronize(t.device)
    runtime_s = time.perf_counter() - start
    # Transfer and metrics deliberately do not form part of the solver time.
    return as_numpy(prediction), runtime_s


def check_device_port(device: torch.device, alpha: float) -> dict:
    """Small compatibility check only; this does not validate the numerical method."""
    t_cpu = torch.linspace(0.0, 1.0, 11, dtype=torch.float64).unsqueeze(1)
    y0_cpu = torch.tensor([1.0], dtype=torch.float64)
    a_cpu = torch.tensor([alpha], dtype=torch.float64)
    baseline = coelho_reference.solve(a_cpu, fractional_rhs_original, y0_cpu, t_cpu)
    port_cpu = run_original_solver(t_cpu, y0_cpu, alpha)
    np.testing.assert_array_equal(as_numpy(baseline).squeeze(), as_numpy(port_cpu))
    report = {"original_loop_cpu_bitwise_equal": True, "per_solver": {}}

    t_gpu, y0_gpu = t_cpu.to(device), y0_cpu.to(device)
    try:
        coelho_reference.solve(a_cpu.to(device), fractional_rhs_original, y0_gpu, t_gpu)
        torch.cuda.synchronize(device)
        report["unmodified_original_cuda"] = "succeeded"
    except RuntimeError as exc:
        if "device" not in str(exc).lower():
            raise
        report["unmodified_original_cuda"] = str(exc)

    for name, fn, atol, rtol in [
        (ORIGINAL_SOLVER_NAME, run_original_solver, 1e-7, 1e-6),
        (FDEINT_SOLVER_NAME, run_forced_solver_with_zero_input, 1e-10, 1e-9),
    ]:
        cpu_y = fn(t_cpu, y0_cpu, alpha)
        gpu_y = fn(t_gpu, y0_gpu, alpha)
        torch.cuda.synchronize(device)
        if gpu_y.device != device or gpu_y.shape != cpu_y.shape:
            raise RuntimeError(f"{name}: unexpected device or shape")
        cpu_arr, gpu_arr = as_numpy(cpu_y), as_numpy(gpu_y)
        if not np.isfinite(gpu_arr).all():
            raise RuntimeError(f"{name}: non-finite CUDA result")
        np.testing.assert_allclose(gpu_arr, cpu_arr, atol=atol, rtol=rtol)
        report["per_solver"][name] = {
            "device": str(gpu_y.device), "dtype": str(gpu_y.dtype),
            "max_cpu_gpu_difference": float(np.max(np.abs(cpu_arr - gpu_arr))),
            "atol": atol, "rtol": rtol,
        }
    return report


def format_mean_std(mean: float, std: float, digits: int = 3) -> str:
    if math.isnan(mean):
        return "--"
    if math.isnan(std) or abs(std) < 1e-15:
        return f"{mean:.{digits}g}"
    return f"{mean:.{digits}g} ({std:.2g})"


def format_sci_mean_std(mean: float, std: float) -> str:
    if math.isnan(mean):
        return "--"
    mean_text = latex_sci_number(mean, digits=3)
    if math.isnan(std) or abs(std) < 1e-15:
        return mean_text
    return rf"{mean_text}\,({latex_sci_number(std, digits=1)})"


def latex_sci_number(value: float, digits: int = 3) -> str:
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"{float(mantissa):.{digits}f}\times 10^{{{int(exponent)}}}"


def summarize_results(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.groupby(["solver", "dt"], as_index=False)
        .agg(
            runtime_s_mean=("runtime_s", "mean"),
            runtime_s_std=("runtime_s", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            max_abs_error_mean=("max_abs_error", "mean"),
            max_abs_error_std=("max_abs_error", "std"),
            nrmse_range_mean=("nrmse_range", "mean"),
            nrmse_range_std=("nrmse_range", "std"),
        )
        .sort_values(["dt", "solver"])
    )


def make_latex_table(summary: pd.DataFrame, gpu_name: str, num_runs: int) -> str:
    summary = summary.copy()
    summary["solver"] = summary["solver"].replace(
        {
            LEGACY_FORCED_SOLVER_NAME: FDEINT_SOLVER_NAME,
            LEGACY_THIS_WORK_SOLVER_NAME: FDEINT_SOLVER_NAME,
        }
    )
    pivot = summary.pivot(index="dt", columns="solver")
    rows = []
    for dt in sorted(summary["dt"].unique(), reverse=True):
        original = pivot.xs(dt)
        row = {
            "dt": dt,
            "time_original": format_mean_std(
                original.loc["runtime_s_mean", ORIGINAL_SOLVER_NAME],
                original.loc["runtime_s_std", ORIGINAL_SOLVER_NAME],
            ),
            "time_forced": format_mean_std(
                original.loc["runtime_s_mean", FDEINT_SOLVER_NAME],
                original.loc["runtime_s_std", FDEINT_SOLVER_NAME],
            ),
            "rmse_original": format_sci_mean_std(
                original.loc["rmse_mean", ORIGINAL_SOLVER_NAME],
                original.loc["rmse_std", ORIGINAL_SOLVER_NAME],
            ),
            "rmse_forced": format_sci_mean_std(
                original.loc["rmse_mean", FDEINT_SOLVER_NAME],
                original.loc["rmse_std", FDEINT_SOLVER_NAME],
            ),
            "max_original": format_sci_mean_std(
                original.loc["max_abs_error_mean", ORIGINAL_SOLVER_NAME],
                original.loc["max_abs_error_std", ORIGINAL_SOLVER_NAME],
            ),
            "max_forced": format_sci_mean_std(
                original.loc["max_abs_error_mean", FDEINT_SOLVER_NAME],
                original.loc["max_abs_error_std", FDEINT_SOLVER_NAME],
            ),
        }
        rows.append(row)

    lines = [
        "\\begin{table}[H]",
        "    \\centering",
        f"    \\caption[Unforced FDE solver verification]{{Unforced solver verification against the Mittag-Leffler reference on an {gpu_name}. The \\protect\\texttt{{FDEint}} implementation extended in this work is evaluated through the forced solver interface with $u(t)=0$. The reference implementation is adapted only for GPU device allocation. Runtimes are reported as mean (standard deviation) over {num_runs} repetitions.}}",
        "    \\label{tab:FDE_unforced_zero_forcing_validation}",
        "    \\resizebox{\\columnwidth}{!}{%",
        "    \\begin{tabular}{lcc cc cc}",
        "        \\toprule",
        "        & \\multicolumn{2}{c}{Runtime (s)} & \\multicolumn{2}{c}{RMSE} & \\multicolumn{2}{c}{Maximum absolute error} \\\\",
        "        \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        f"        $\\Delta t$ & Original & {FDEINT_DISPLAY_NAME} & Original & {FDEINT_DISPLAY_NAME} & Original & {FDEINT_DISPLAY_NAME} \\\\",
        "        \\midrule",
    ]
    for row in rows:
        lines.append(
            f"        ${row['dt']:.3g}$ & "
            f"${row['time_original']}$ & "
            f"${row['time_forced']}$ & "
            f"${row['rmse_original']}$ & "
            f"${row['rmse_forced']}$ & "
            f"${row['max_original']}$ & "
            f"${row['max_forced']}$ \\\\"
        )
    lines.extend(
        [
            "        \\bottomrule",
            "    \\end{tabular}%",
            "    }",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Separate CUDA comparison of the same unforced Mittag-Leffler benchmark."
    )
    parser.add_argument("--dt-values", nargs="+", type=float, default=[1.0, 0.1, 0.04, 0.02, 0.01, 0.005])
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--length", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--raw-csv",
        type=Path,
        default=None,
        help="Regenerate the LaTeX table from per-run results without rerunning the solvers.",
    )
    source.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Regenerate the LaTeX table from an existing summary CSV without rerunning the solvers.",
    )
    args = parser.parse_args()

    replay_csv = args.raw_csv if args.raw_csv is not None else args.summary_csv
    if args.output_dir is None:
        args.output_dir = (OUTPUT_PATH / "pdf/Evaluation" / EVALUATION_SOURCE / "Supplementary/FDE_Solver_ZeroForcing"
                           if replay_csv is not None else DEFAULT_RESULTS_DIR)

    if replay_csv is not None:
        results = pd.read_csv(replay_csv)
        summary = summarize_results(results) if args.raw_csv is not None else results
        saved_metadata = json.loads((replay_csv.parent / "run_metadata.json").read_text())
        latex_table = make_latex_table(summary, saved_metadata["gpu_name"],
                                      saved_metadata["arguments"]["num_runs"])
        args.output_dir.mkdir(parents=True, exist_ok=True)
        table_path = args.output_dir / "solver_unforced_zero_forcing_gpu_table.tex"
        table_path.write_text(latex_table, encoding="utf-8")
        print(f"Wrote {table_path}")
        return

    if args.num_runs < 1 or args.warmup_runs < 0:
        parser.error("num-runs must be positive and warmup-runs nonnegative")
    if args.length <= 0 or not 0 < args.alpha <= 1:
        parser.error("length must be positive and alpha must lie in (0, 1]")
    if any(dt <= 0 or dt > args.length for dt in args.dt_values):
        parser.error("Each step size must be positive and no larger than length")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this script never falls back to CPU.")
    device = torch.device(f"cuda:{args.gpu_index}")
    torch.cuda.set_device(device)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "platform": platform.platform(),
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "arguments": {k: v for k, v in vars(args).items()
                      if k not in {"output_dir", "raw_csv", "summary_csv"}},
        "input_dtype": "torch.float64",
        "reference_internal_dtype_note": "Original float32 temp buffer is retained.",
        "grad_enabled": torch.is_grad_enabled(),
        "timing": "Synchronized wall clock including Python/solver setup, excluding input and output transfers, reference evaluation, metrics, gc and warmups.",
        "reference_port": "Initial y buffer on y0.device; original helpers and algorithm unchanged.",
        "reference_alignment_note": "CPU benchmark output indexing retained; the open reference time-alignment issue is not fixed by this GPU port.",
    }
    metadata_path = args.output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Device: {device} ({metadata['gpu_name']}), inputs=float64, grad_enabled={torch.is_grad_enabled()}", flush=True)
    print(f"Runs per solver/step: {args.num_runs}; untimed warmups: {args.warmup_runs}", flush=True)
    device_checks = check_device_port(device, args.alpha)
    print(json.dumps(device_checks, indent=2), flush=True)
    print("Device checks passed: original CPU loop unchanged; both CUDA results agree with their CPU results.", flush=True)
    if args.check_only:
        print(f"Wrote {metadata_path}")
        return

    solvers = [
        (ORIGINAL_SOLVER_NAME, run_original_solver),
        (FDEINT_SOLVER_NAME, run_forced_solver_with_zero_input),
    ]
    records = []

    for dt in args.dt_values:
        steps = int(round(args.length / dt)) + 1
        t_cpu = torch.linspace(0.0, args.length, steps, dtype=torch.float64).unsqueeze(1)
        reference = reference_solution(t_cpu, args.alpha)
        t = t_cpu.to(device)
        y0 = torch.tensor([1.0], dtype=torch.float64, device=device)

        print(f"Testing dt={dt:g} with {steps} samples on {device}", flush=True)
        for solver_name, solver_fn in solvers:
            for warmup_idx in range(args.warmup_runs):
                print(f"  warmup {warmup_idx + 1}/{args.warmup_runs}, {solver_name}", flush=True)
                warmup = solver_fn(t, y0, args.alpha)
                torch.cuda.synchronize(device)
                del warmup
        for run_idx in range(args.num_runs):
            randomized_solvers = solvers.copy()
            random.shuffle(randomized_solvers)

            for solver_name, solver_fn in randomized_solvers:
                prediction, runtime_s = time_solver(solver_fn, t, y0, args.alpha)

                if prediction.shape != reference.shape:
                    raise RuntimeError(
                        f"{solver_name} returned shape {prediction.shape}, expected {reference.shape}."
                    )

                metrics = compute_metrics(reference[1:], prediction[1:])
                records.append(
                    {
                        "solver": solver_name,
                        "dt": dt,
                        "run": run_idx + 1,
                        "n_samples": steps,
                        "length": args.length,
                        "alpha": args.alpha,
                        "runtime_s": runtime_s,
                        "device": str(device),
                        "gpu_name": metadata["gpu_name"],
                        **metrics,
                    }
                )
                pd.DataFrame(records).to_csv(
                    args.output_dir / "solver_unforced_zero_forcing_gpu_raw.csv", index=False
                )
                print(
                    f"  run {run_idx + 1}/{args.num_runs}, {solver_name}: "
                    f"runtime={runtime_s:.3f}s, RMSE={metrics['rmse']:.3e}, "
                    f"max_abs={metrics['max_abs_error']:.3e}", flush=True
                )

        pd.DataFrame(records).to_csv(args.output_dir / "solver_unforced_zero_forcing_gpu_raw.csv", index=False)

    raw = pd.DataFrame(records)
    summary = summarize_results(raw)

    summary_path = args.output_dir / "solver_unforced_zero_forcing_gpu_summary.csv"
    table_path = args.output_dir / "solver_unforced_zero_forcing_gpu_table.tex"
    latex_table = make_latex_table(summary, metadata["gpu_name"], args.num_runs)
    summary.to_csv(summary_path, index=False)
    table_path.write_text(latex_table, encoding="utf-8")

    print("\nSummary:")
    print(summary)
    print(f"\nWrote {summary_path}")
    print(f"Wrote {table_path}")
    metadata["measured_runs"] = len(records)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
