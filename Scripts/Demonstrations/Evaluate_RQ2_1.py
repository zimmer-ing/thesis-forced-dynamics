"""Reconstruct the four Laplace-transform examples used in the dissertation."""

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Constants import OUTPUT_PATH

output_dir = OUTPUT_PATH / "pdf/Demonstrations/RQ2_1"
step_terms = [17, 129, 513]
step_time = 3.5
step_interval = (0.02, 8.0, 2400)
step_display_radius = 0.6
noise_terms = [33, 257]
noise_interval = (4.0, 8.0, 1400)
noise_seed = 17
noise_components = 25
noise_frequency_band = (14.0, 34.0)  # Angular frequency, rad/s.
noise_rms = 0.25
slow_frequency = 1.5
ilt_settings = dict(alpha=2e-3, tol=2e-2, scale=2.6, shift=0.0)
include_forward_transforms = True
forward_settings = dict(alpha=1e-3, tol=1e-3, scale=2.0, shift=0.5)
forward_terms = 129
sine_sample_counts = [17, 129]
sine_frequency = 3.0
sine_target_points = 401
ramp_sample_count = 129
ramp_target_points = 2001


def inverse_transform(terms, settings):
    import torch
    from src.models.LaplaceNet_utils.InverseLaplaceTransform import ILTFourier

    return ILTFourier(ilt_reconstruction_terms=terms, **settings,
                      torch_float_dtype=torch.float64, torch_complex_dtype=torch.complex128,
                      device=torch.device("cpu"))


def compute_step():
    import numpy as np
    import torch

    time = torch.linspace(*step_interval, dtype=torch.float64)
    reference = (time.numpy() >= step_time).astype(float)
    curves, metrics = {}, []
    for terms in step_terms:
        inverse = inverse_transform(terms, ilt_settings)
        queries = inverse.get_query_points(time)
        values = torch.exp(-step_time * queries) / queries
        curve = inverse.reconstruct(values[None, :, :, None], time)[0, :, 0].numpy()
        queries_at_step = inverse.get_query_points(torch.tensor([step_time], dtype=torch.float64))[0]
        sigma, omega = float(queries_at_step[0].real), float(queries_at_step[-1].imag)
        error = curve - reference
        curves[terms] = curve
        metrics.append(dict(ilt_terms=terms, sigma_at_step=sigma, omega_max_at_step=omega,
                            frequency_max_hz_at_step=omega / (2 * np.pi),
                            rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(np.abs(error))),
                            max_abs_error=float(np.max(np.abs(error)))))
    return dict(time=time.numpy(), reference=reference, curves=curves), metrics


def compute_noisy_sine():
    import numpy as np
    import torch

    time = torch.linspace(*noise_interval, dtype=torch.float64)
    clean = np.sin(slow_frequency * time.numpy())
    rng = np.random.default_rng(noise_seed)
    frequencies = np.sort(rng.uniform(*noise_frequency_band, noise_components))
    coefficients = rng.normal(size=noise_components)
    disturbance = np.sum(coefficients[:, None] * np.sin(frequencies[:, None] * time.numpy()[None, :]), axis=0)
    coefficients *= noise_rms / np.sqrt(np.mean(disturbance**2))
    disturbance = np.sum(coefficients[:, None] * np.sin(frequencies[:, None] * time.numpy()[None, :]), axis=0)
    noisy = clean + disturbance
    omega = torch.as_tensor(frequencies, dtype=torch.float64).view(1, 1, -1)
    amplitude = torch.as_tensor(coefficients, dtype=torch.float64).view(1, 1, -1)
    curves, metrics = {}, []
    for terms in noise_terms:
        inverse = inverse_transform(terms, ilt_settings)
        queries = inverse.get_query_points(time)
        values = (slow_frequency / (queries**2 + slow_frequency**2)
                  + torch.sum(amplitude * omega / (queries.unsqueeze(-1)**2 + omega**2), dim=-1))
        curve = inverse.reconstruct(values[None, :, :, None], time)[0, :, 0].numpy()
        recovered_rms = float(np.sqrt(np.mean((curve - clean)**2)))
        curves[terms] = curve
        metrics.append(dict(ilt_terms=terms,
                            omega_max_at_time_start=(terms - 1) * np.pi / (ilt_settings["scale"] * noise_interval[0]),
                            omega_max_at_time_end=(terms - 1) * np.pi / (ilt_settings["scale"] * noise_interval[1]),
                            rmse_to_clean=recovered_rms,
                            rmse_to_noisy=float(np.sqrt(np.mean((curve - noisy)**2))),
                            recovered_disturbance_rms=recovered_rms,
                            recovered_disturbance_fraction=recovered_rms / noise_rms))
    return dict(time=time.numpy(), clean=clean, noisy=noisy, curves=curves), metrics


def compute_forward_transforms(case):
    import numpy as np
    import torch
    from src.models.LaplaceNet_utils.LaplaceTransform import discrete_laplace_transform, laplace_from_fft

    if case == "sine":
        counts = sine_sample_counts
        time = torch.linspace(max(0.08, 2 / min(counts)), 0.98, sine_target_points, dtype=torch.float64)
        reference = torch.sin(2 * torch.pi * sine_frequency * time)
    elif case == "ramp":
        counts = [ramp_sample_count]
        time = torch.linspace(1 / ramp_sample_count, 0.99, ramp_target_points, dtype=torch.float64)
        reference = time
    else:
        raise ValueError(f"Unknown example: {case}")
    inverse = inverse_transform(forward_terms, forward_settings)
    queries = inverse.get_query_points(time)
    curves, metrics = {}, []
    for count in counts:
        sample_time = torch.arange(1, count + 1, dtype=torch.float64) * (1 / count)
        if case == "ramp":
            sample_time = torch.arange(1, count + 1, dtype=torch.float64) / count
        values = torch.sin(2 * torch.pi * sine_frequency * sample_time) if case == "sine" else sample_time
        x = values.view(1, -1, 1)
        t = (sample_time + forward_settings["shift"]).view(1, -1, 1)
        s = queries.view(1, queries.shape[0], 1, queries.shape[1])
        curves[count] = dict(input_time=sample_time.numpy(), input_values=values.numpy())
        for name, transform in (("DLT", discrete_laplace_transform), ("FFLT", laplace_from_fft)):
            with torch.no_grad():
                transformed = transform(x, s, t, mode="chunk", chunk_size=1024)
                curve = inverse.reconstruct(transformed, time)[0, :, 0].numpy()
            if not np.isfinite(curve).all():
                raise FloatingPointError(f"Non-finite {case}/{count}/{name} reconstruction")
            curves[count][name] = curve
            masks = {"full": np.ones(len(time), dtype=bool)}
            if case == "ramp":
                masks.update(window_start=time.numpy() <= 0.12, window_end=time.numpy() >= 0.88)
            for region, mask in masks.items():
                error = curve[mask] - reference.numpy()[mask]
                metrics.append(dict(case=case, sampling_hz=count, ilt_terms=forward_terms, method=name,
                                    region=region, rmse=float(np.sqrt(np.mean(error**2))),
                                    max_abs_error=float(np.max(np.abs(error)))))
    return dict(time=time.numpy(), reference=reference.numpy(), curves=curves), metrics


def plot_step(data, metrics):
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(7.15, 4.25))
    grid = fig.add_gridspec(2, 3, height_ratios=[0.72, 1.0], hspace=0.72, wspace=0.13)
    upper = fig.add_subplot(grid[0, :])
    axes = [fig.add_subplot(grid[1, index]) for index in range(3)]
    sigma = metrics[0]["sigma_at_step"]
    limits = [row["frequency_max_hz_at_step"] for row in metrics]
    frequency = np.linspace(0, 1.1 * max(limits), 2400)
    upper.plot(frequency, 1 / np.sqrt(sigma**2 + (2 * np.pi * frequency)**2), color="#303030", linewidth=1.45)
    for index, (terms, limit, axis) in enumerate(zip(step_terms, limits, axes)):
        upper.axvline(limit, color="#0072B2", linewidth=1.25, linestyle="--")
        upper.scatter([limit], [1 / np.sqrt(sigma**2 + (2 * np.pi * limit)**2)], color="#0072B2", s=15, zorder=4)
        last = index == len(step_terms) - 1
        upper.annotate(rf"$N_{{\mathrm{{ILT}}}}={terms}$", xy=(limit, 0.92), xycoords=("data", "axes fraction"),
                       xytext=(-4 if last else 4, 0), textcoords="offset points", ha="right" if last else "left", va="top")
        axis.plot(data["time"], data["reference"], color="#202020", linewidth=1.55, linestyle="--", label="Unit-step reference")
        axis.plot(data["time"], data["curves"][terms], color="#0072B2", linewidth=1.35, label="FS-ILT reconstruction")
        axis.set(title=rf"$N_{{\mathrm{{ILT}}}}={terms}$", xlabel=r"Time $t$ (s)",
                 xlim=(step_time - step_display_radius, step_time + step_display_radius), ylim=(-0.18, 1.22))
        axis.grid(True, color="#e6e6e6", linewidth=0.5)
    upper.set(yscale="log", xlim=(0, frequency[-1]), ylabel="Magnitude (log scale)",
              xlabel=r"Frequency $f=\omega/(2\pi)$ (Hz)",
              title=rf"Unit-step transform and FS-ILT frequency limits at $t_{{\mathrm{{s}}}}={step_time:g}$ s")
    upper.grid(True, which="both", color="#e4e4e4", linewidth=0.5)
    for axis in axes[1:]:
        axis.tick_params(labelleft=False)
    axes[0].set_ylabel(r"Signal $y(t)$")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.080, right=0.992, top=0.945, bottom=0.155)
    return fig


def plot_noisy_sine(data):
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(7.15, 4.55))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.12], hspace=0.42, wspace=0.20)
    upper = fig.add_subplot(grid[0, :])
    axes = [fig.add_subplot(grid[1, index]) for index in range(2)]
    upper.plot(data["time"], data["noisy"], color="#8c8c8c", linewidth=0.75, label="Disturbed signal")
    upper.plot(data["time"], data["clean"], color="#111111", linewidth=1.7, label="Main signal")
    upper.set(title="Low-frequency oscillation with high-frequency disturbance", ylabel=r"Signal $y(t)$")
    upper.legend(loc="upper right", frameon=False, ncol=2)
    for axis, terms in zip(axes, noise_terms):
        axis.plot(data["time"], data["noisy"], color="#b5b5b5", linewidth=0.6, alpha=0.75, label="Disturbed signal")
        axis.plot(data["time"], data["clean"], color="#111111", linewidth=1.35, linestyle="--", label="Main signal")
        axis.plot(data["time"], data["curves"][terms], color="#0072B2", linewidth=1.35, label="FS-ILT reconstruction")
        axis.set(title=rf"$N_{{\mathrm{{ILT}}}}={terms}$", ylabel=r"Signal $y(t)$", xlabel=r"Time $t$")
    padding = 0.12 * np.ptp(data["noisy"])
    for axis in [upper, *axes]:
        axis.set(xlim=noise_interval[:2], ylim=(min(data["noisy"]) - padding, max(data["noisy"]) + padding))
        axis.grid(True, color="#e6e6e6", linewidth=0.55)
    fig.legend(*axes[-1].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=3, frameon=False)
    fig.subplots_adjust(left=0.080, right=0.992, top=0.935, bottom=0.135)
    return fig


def plot_forward_transforms(case, data):
    import matplotlib.pyplot as plt

    if case == "sine":
        fig, axes = plt.subplots(1, 2, figsize=(15 / 2.54, 2.8), sharey=True)
        counts = sine_sample_counts
    else:
        fig = plt.figure(figsize=(15 / 2.54, 3.70))
        grid = fig.add_gridspec(2, 2, height_ratios=[0.88, 1], hspace=0.70, wspace=0.40)
        axes = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
        counts = [ramp_sample_count] * 3
    for axis, count in zip(axes, counts):
        curves = data["curves"][count]
        for name, values, color, width, style, order in (("DLT + FS-ILT", curves["DLT"], "#D55E00", 1.35, "-", 2),
                                                       ("FFLT + FS-ILT", curves["FFLT"], "#0072B2", 1.3, "-", 3),
                                                       ("Reference", data["reference"], "#202020", 1.25, "--", 4)):
            axis.plot(data["time"], values, color=color, linewidth=width, linestyle=style, zorder=order, label=name)
        axis.set_xlabel(r"Time $t$ (s)")
        axis.grid(True, color="#e6e6e6", linewidth=0.5)
    if case == "sine":
        for axis, count in zip(axes, counts):
            axis.set(title=rf"$f_{{\mathrm{{s}}}}={count}\,\mathrm{{Hz}}$", xlim=(data["time"][0], data["time"][-1]),
                     ylim=(-5.25, 5.25), yticks=[-4, -2, 0, 2, 4], xticks=[0.2, 0.4, 0.6, 0.8])
        samples = data["curves"][counts[0]]
        visible = (samples["input_time"] >= data["time"][0]) & (samples["input_time"] <= data["time"][-1])
        axes[0].plot(samples["input_time"][visible], samples["input_values"][visible], linestyle="none", marker="o",
                     markersize=4.2, markerfacecolor="white", markeredgecolor="#202020", label="Input samples", clip_on=False)
        fig.suptitle(rf"$u(t)=\sin(2\pi f_0 t),\quad f_0={sine_frequency:g}\,\mathrm{{Hz}},\quad T=1\,\mathrm{{s}},"
                     rf"\quad N_{{\mathrm{{ILT}}}}={forward_terms}$", fontsize=10, y=0.975)
        fig.subplots_adjust(left=0.12, right=0.985, top=0.80, bottom=0.36, wspace=0.15)
    else:
        title = rf"$u(t)=t/T,\quad T=1\,\mathrm{{s}},\quad f_{{\mathrm{{s}}}}={ramp_sample_count}\,\mathrm{{Hz}},\quad N_{{\mathrm{{ILT}}}}={forward_terms}$"
        for axis, xlim, ylim, title in zip(axes, [(0, 1), (data["time"][0], 0.12), (0.88, 0.99)],
                                         [(-0.12, 1.13), (-0.10, 0.14), (0.84, 1.10)], [title, "Window start (zoom)", "Window end (zoom)"]):
            axis.set(xlim=xlim, ylim=ylim, title=title)
            axis.set_ylabel(r"Signal $u(t)$")
        axes[0].set(xticks=[0, 0.2, 0.4, 0.6, 0.8, 1], yticks=[0, 0.5, 1])
        axes[1].set(xticks=[0.02, 0.06, 0.10], yticks=[-0.10, 0, 0.10])
        axes[2].set(xticks=[0.88, 0.92, 0.96], yticks=[0.85, 0.95, 1.05])
        for left, right in ((1 / ramp_sample_count, 0.12), (0.88, 0.99)):
            axes[0].axvspan(left, right, color="#777777", alpha=0.08, zorder=0)
        fig.subplots_adjust(left=0.15, right=0.965, top=0.90, bottom=0.22)
    axes[0].set_ylabel(r"Signal $u(t)$")
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(name) for name in ("Reference", "DLT + FS-ILT", "FFLT + FS-ILT")]
    if case == "sine":
        order.append(labels.index("Input samples"))
    fig.legend([handles[index] for index in order], [labels[index] for index in order], loc="lower center",
               bbox_to_anchor=(0.5, 0.015 if case == "sine" else 0.02), ncol=2 if case == "sine" else 3, frameon=False)
    return fig


def main(dry_run=False):
    print(f"ILT step {step_terms}; noisy sine {noise_terms}; DLT/FFLT examples: {include_forward_transforms}")
    print(f"Output: {output_dir}")
    if dry_run:
        return 0
    import matplotlib.pyplot as plt
    import torch
    from Scripts.utils.evaluation import configure_plot, save_figure

    torch.set_num_threads(2)
    configure_plot()
    output_dir.mkdir(parents=True, exist_ok=True)
    step, step_metrics = compute_step()
    noisy, noise_metrics = compute_noisy_sine()
    results = [("rq2_1_ilt_step_reconstruction", plot_step(step, step_metrics), step_metrics, True),
               ("rq2_1_ilt_noisy_sine", plot_noisy_sine(noisy), noise_metrics, True)]
    if include_forward_transforms:
        with plt.rc_context({"font.family": "serif", "font.serif": ["cmr10"], "axes.formatter.use_mathtext": True,
                             "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                             "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10}):
            for case in ("sine", "ramp"):
                data, metrics = compute_forward_transforms(case)
                results.append((f"rq2_1_dlt_fflt_{case}", plot_forward_transforms(case, data), metrics, False))
    for stem, figure, metrics, tight in results:
        save_figure(figure, output_dir / stem, tight=tight)
        plt.close(figure)
        with (output_dir / f"{stem}_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show the editable examples and output directory without computing or writing.")
    raise SystemExit(main(dry_run=parser.parse_args().dry_run))
