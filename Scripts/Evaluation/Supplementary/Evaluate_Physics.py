"""Evaluate whether the RLC NO-Prior gain comes from physics alone or from the NN residual.

This diagnostic compares a trained NO-Prior model against its isolated components:

1. Physics only with the initial grey-box parameters from the config.
2. Physics only after copying the physics parameters learned inside NO-Prior.
3. The neural residual learned inside NO-Prior, integrated without the physics term.
4. The full NO-Prior model, i.e. physics plus neural residual.

Removing a component at inference is not equivalent to retraining without it.

The default channel groups are tailored to the fourth-order RLC dataset:
symbolic channels V_out/I_in and internal channels V_int/I_int.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Constants import NEW_RESULTS_PATH
from src.datasets import DATASET_REGISTRY
from src.models import MODEL_REGISTRY
from Scripts.utils.naming import model_directory, model_key
from src.models.PhysicsOnlyNODE import PhysicsOnlyAdaptiveNODE


RESULTS_ROOT = NEW_RESULTS_PATH / "full_data_hpo"
DEFAULT_RUN_DIR = model_directory(RESULTS_ROOT, "RLCDataset", "NO-Prior") / "Seed_42"
DEFAULT_OUTPUT_DIR = NEW_RESULTS_PATH / "physics_components/RLCDataset/NO-Prior/Seed_42"

VARIANT_LABELS = {
    "physics_only_initial_greybox_params": "Physical component, initial parameters",
    "physics_only_with_learned_physics_params": "Physical component, trained parameters",
    "nn_only_with_learned_weights": "Neural component, trained within NO-Prior",
    "full_physics_plus_nn": "NO-Prior, complete model",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--physics-prior-level", default="Greybox")
    parser.add_argument("--symbolic-channels", nargs="+", default=["V_out", "I_in"])
    parser.add_argument("--internal-channels", nargs="+", default=["V_int", "I_int"])
    parser.add_argument(
        "--skip-scaling-ablation",
        action="store_true",
        help="Do not additionally evaluate NO-Prior with the NODE scaling factor forced to one.",
    )
    return parser.parse_args()


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_config(run_dir: Path, device: str) -> dict:
    config = read_yaml(run_dir / "model_config.yaml")
    config["chosen_device"] = device
    config.setdefault("experiment", {})["wanted_device"] = device
    return config


def response_channels_from_dataset(dataset_cls) -> list[str]:
    channels = dataset_cls.get_channels()
    if isinstance(channels, dict):
        return list(channels["response"])
    return list(channels[2])


def build_loader(config: dict, split: str, batch_size: int):
    dataset_name = config["dataset"]["name"]
    dataset_cls = DATASET_REGISTRY[dataset_name]
    dataset = dataset_cls(config["dataset"], data_split=split, device=config["chosen_device"])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset_cls.collate_fn,
    )
    return dataset_cls, dataset, loader


def load_prior_model(config: dict, run_dir: Path):
    model_name = config["model"]["name"]
    model = MODEL_REGISTRY[model_name](config).to(config["chosen_device"])
    state = torch.load(
        run_dir / "model_checkpoints" / "best_model.pth",
        map_location=config["chosen_device"],
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    return model, state


def build_physics_only_model(config: dict, physics_prior_level: str, train_physics_params: bool = False):
    physics_config = json.loads(json.dumps(config))
    physics_config["model"]["physics_prior_level"] = physics_prior_level
    physics_config["model"]["train_physics_params"] = train_physics_params
    model = PhysicsOnlyAdaptiveNODE(physics_config).to(physics_config["chosen_device"])
    model.eval()
    return model


def build_neural_only_model(prior_model):
    """Integrate the trained residual alone, preserving its scaling and solver."""
    model = copy.deepcopy(prior_model)
    model.ODE = model.ODE_bb
    model.eval()
    return model


def copy_prior_physics_parameters(prior_state: dict[str, torch.Tensor], physics_model) -> int:
    current_state = physics_model.state_dict()
    replacement = dict(current_state)
    transferred = 0
    for key, value in current_state.items():
        suffix = key.removeprefix("ODE.physics_ODE.")
        prior_key = "ODE_Phy.physics_ODE." + suffix
        if prior_key in prior_state and prior_state[prior_key].shape == value.shape:
            replacement[key] = prior_state[prior_key]
            transferred += 1
    physics_model.load_state_dict(replacement, strict=False)
    return transferred


def flatten_prediction_output(result: dict) -> tuple[torch.Tensor, torch.Tensor]:
    pred = result["predictions"].detach().cpu()
    target = result["response_forecast"].detach().cpu()
    if pred.ndim == 3:
        pred = pred.reshape(-1, pred.shape[-1])
    if target.ndim == 3:
        target = target.reshape(-1, target.shape[-1])
    return pred, target


def predict_flat(model, loader) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        result = model.predict(loader)
    return flatten_prediction_output(result)


def channel_indices(response_channels: list[str], selected_channels: Iterable[str]) -> list[int]:
    indices = []
    for channel in selected_channels:
        if channel not in response_channels:
            raise ValueError(
                f"Channel '{channel}' not found in response channels {response_channels}."
            )
        indices.append(response_channels.index(channel))
    return indices


def rmse_and_mse(pred: torch.Tensor, target: torch.Tensor, indices: list[int]) -> dict:
    error = pred[:, indices] - target[:, indices]
    mse = torch.mean(error**2).item()
    return {
        "rmse": math.sqrt(mse),
        "mse": mse,
        "n_values": int(error.numel()),
    }


def compute_group_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    response_channels: list[str],
    symbolic_channels: list[str],
    internal_channels: list[str],
) -> dict:
    groups = {
        "symbolic_channels": symbolic_channels,
        "internal_channels": internal_channels,
        "all_channels": response_channels,
    }
    metrics = {}
    for group_name, channels in groups.items():
        indices = channel_indices(response_channels, channels)
        values = rmse_and_mse(pred, target, indices)
        values["channels"] = channels
        metrics[group_name] = values
    per_channel = {}
    for channel in response_channels:
        idx = channel_indices(response_channels, [channel])
        per_channel[channel] = rmse_and_mse(pred, target, idx)
    metrics["per_channel"] = per_channel
    return metrics


def scaling_audit(config: dict, prior_state: dict[str, torch.Tensor], model) -> dict:
    fresh_model = MODEL_REGISTRY[config["model"]["name"]](config).to(config["chosen_device"])
    expected = [key for key in fresh_model.state_dict() if "scaling_factor" in key]
    checkpoint = [key for key in prior_state if "scaling_factor" in key]
    before = {
        key: fresh_model.state_dict()[key].detach().cpu().tolist()
        for key in expected
    }
    fresh_model.load_state_dict(prior_state)
    after = {
        key: fresh_model.state_dict()[key].detach().cpu().tolist()
        for key in expected
    }
    return {
        "pretraining_scaling": config["model"].get("pretraining_scaling"),
        "pretraining_epochs": config["model"].get("pretraining_epochs"),
        "expected_scaling_keys": expected,
        "checkpoint_scaling_keys": checkpoint,
        "fresh_scaling_before_load": before,
        "scaling_after_load": after,
        "strict_load_ok": True,
        "ode_bb_is_wrapper_blackbox": getattr(model, "ODE_bb", None) is getattr(model.ODE, "blackbox_ode", None),
    }


def write_csv(path: Path, payload: dict) -> None:
    rows = []
    for variant, metrics in payload["metrics"].items():
        for group_name, values in metrics["groups"].items():
            rows.append(
                {
                    "variant": variant,
                    "group": group_name,
                    "channels": ",".join(values["channels"]),
                    "rmse": values["rmse"],
                    "mse": values["mse"],
                    "n_values": values["n_values"],
                }
            )
        for channel, values in metrics["per_channel"].items():
            rows.append(
                {
                    "variant": variant,
                    "group": f"channel:{channel}",
                    "channels": channel,
                    "rmse": values["rmse"],
                    "mse": values["mse"],
                    "n_values": values["n_values"],
                }
            )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def latex_channel_label(channel: str) -> str:
    if "_" not in channel:
        return latex_escape(channel.lower())
    base, subscript = channel.split("_", 1)
    return rf"{latex_escape(base.lower())}_{{\mathrm{{{latex_escape(subscript)}}}}}"


def latex_channel_list(channels: Iterable[str]) -> str:
    return ", ".join(latex_channel_label(channel) for channel in channels)


def format_metric(value: float, best_value: float | None = None) -> str:
    formatted = f"{value:.4g}"
    if best_value is not None and math.isclose(value, best_value, rel_tol=1e-12, abs_tol=1e-12):
        return rf"\textbf{{{formatted}}}"
    return formatted


def write_latex_table(path: Path, payload: dict) -> None:
    variants = list(payload["metrics"].keys())
    all_values = {
        variant: payload["metrics"][variant]["groups"]["all_channels"]["rmse"]
        for variant in variants
    }
    physics_values = {
        variant: payload["metrics"][variant]["groups"]["symbolic_channels"]["rmse"]
        for variant in variants
    }
    best_all = min(all_values.values())
    best_physics = min(physics_values.values())
    physics_channels = latex_channel_list(payload["symbolic_channels"])
    response_channels = latex_channel_list(payload["response_channels"])

    lines = [
        r"\begin{table}[!htbp]",
        r"    \centering",
        r"    \begingroup",
        r"    \scriptsize",
        r"    \setlength{\tabcolsep}{3pt}",
        r"    \renewcommand{\arraystretch}{1.08}",
        (
            r"    \caption[Physical-component ablation on the RLC dataset]{Ablation of the physical component on the RLC dataset (E2a). The test-"
            rf"\(\mathrm{{RMSE}}\) is reported for all observed channels \(\mathbf{{y}}\) and for the channels described by the physical component, \(y_{{\mathrm{{phys}}}}=\{{{physics_channels}\}}\).}}"

        ),
        r"    \label{tab:rlc_prior_physics_only_ablation}",
        r"    \begin{adjustbox}{max width=\columnwidth}",
        r"    \begin{tabular}{lcc}",
        r"        \toprule",
        r"        Variant & \(\mathrm{RMSE}(y)\) & \(\mathrm{RMSE}(y_{\mathrm{phys}})\) \\",
        r"        \midrule",
    ]
    for variant in variants:
        label = VARIANT_LABELS.get(variant, variant)
        lines.append(
            "        "
            + latex_escape(label)
            + " & "
            + format_metric(all_values[variant], best_all)
            + " & "
            + format_metric(physics_values[variant], best_physics)
            + r" \\"
        )
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"    \end{adjustbox}",
            (
                r"    \vspace{0.25em}\par\footnotesize "
                r"All channels: \("
                + response_channels
                + r"\). Channels described by the physical component: \("
                + physics_channels
                + r"\)"
                + r"."
            ),
            r"    \endgroup",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown(path: Path, payload: dict) -> None:
    symbolic = payload["metrics"]["full_physics_plus_nn"]["groups"]["symbolic_channels"]["rmse"]
    physics = payload["metrics"]["physics_only_with_learned_physics_params"]["groups"]["symbolic_channels"]["rmse"]
    ratio = physics / symbolic if symbolic > 0 else float("inf")
    lines = [
        "# RLC physics-only ablation",
        "",
        f"Run: `{payload['run_dir']}`",
        f"Split: `{payload['split']}`",
        f"Forecast points: `{payload['forecast_points_total']}`",
        "",
        "| Variant | Symbolic channels RMSE | Internal channels RMSE | All channels RMSE |",
        "|---|---:|---:|---:|",
    ]
    for variant, metrics in payload["metrics"].items():
        groups = metrics["groups"]
        lines.append(
            "| "
            + variant
            + f" | {groups['symbolic_channels']['rmse']:.10g}"
            + f" | {groups['internal_channels']['rmse']:.10g}"
            + f" | {groups['all_channels']['rmse']:.10g} |"
        )
    lines.extend(
        [
            "",
            "Kurzinterpretation:",
            (
                "- Auf den symbolisch modellierten Kanaelen ist Physics-only mit den "
                f"NO-Prior-Physikparametern etwa {ratio:.1f}x schlechter als das volle NO-Prior."
            ),
            "- Der NN-Residualteil ist damit fuer diesen Check nicht ueberfluessig.",
            "- NN-only nutzt die im gemeinsamen Modell gelernten Gewichte ohne erneutes Training. Es ist kein separat trainiertes Modell ohne Vorwissen.",
            "- Fuer die Dissertation eignet sich das eher als kleine Ablation/Robustheitsnotiz, nicht als Hauptargument.",
        ]
    )
    if "scaling_ablation" in payload:
        lines.extend(
            [
                "",
                "Scaling check:",
                f"- Geladener NO-Prior-Skalierungsfaktor: `{payload['scaling_ablation']['loaded_scaling_factor']}`",
                (
                    "- NO-Prior mit korrekt geladenem Scaling: "
                    f"`{payload['scaling_ablation']['prior_correct_all_rmse']:.10g}` all-channel RMSE."
                ),
                (
                    "- NO-Prior mit Scaling kuenstlich auf 1 gesetzt: "
                    f"`{payload['scaling_ablation']['prior_forced_unit_all_rmse']:.10g}` all-channel RMSE."
                ),
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.run_dir, args.device)
    dataset_cls, dataset, loader = build_loader(config, args.split, args.batch_size)
    response_channels = response_channels_from_dataset(dataset_cls)

    prior_model, prior_state = load_prior_model(config, args.run_dir)
    physics_initial = build_physics_only_model(config, args.physics_prior_level)
    physics_prior_params = build_physics_only_model(config, args.physics_prior_level)
    transferred = copy_prior_physics_parameters(prior_state, physics_prior_params)
    neural_only = build_neural_only_model(prior_model)

    variants = {
        "physics_only_initial_greybox_params": physics_initial,
        "physics_only_with_learned_physics_params": physics_prior_params,
        "nn_only_with_learned_weights": neural_only,
        "full_physics_plus_nn": prior_model,
    }

    payload = {
        "run_dir": str(args.run_dir),
        "output_dir": str(args.output_dir),
        "split": args.split,
        "dataset": config["dataset"]["name"],
        "model": model_key(config["model"]["name"]),
        "history_length": config["dataset"].get("history_length"),
        "forecast_length": config["dataset"].get("forecast_length"),
        "test_or_split_windows": len(dataset),
        "response_channels": response_channels,
        "symbolic_channels": args.symbolic_channels,
        "internal_channels": args.internal_channels,
        "transferred_physics_state_tensors": transferred,
        "neural_only_ablation": {
            "retrained": False,
            "removed_term": "physics RHS",
            "retained": "trained neural residual, scaling, initial state, inputs and solver settings",
        },
        "scaling_audit": scaling_audit(config, prior_state, prior_model),
        "metrics": {},
    }

    for variant_name, model in variants.items():
        print(f"Evaluating {variant_name} ({len(dataset)} {args.split} windows)", flush=True)
        pred, target = predict_flat(model, loader)
        payload["forecast_points_total"] = int(pred.shape[0])
        payload["metrics"][variant_name] = {
            "groups": compute_group_metrics(
                pred,
                target,
                response_channels,
                args.symbolic_channels,
                args.internal_channels,
            ),
        }
        payload["metrics"][variant_name]["per_channel"] = payload["metrics"][variant_name]["groups"].pop(
            "per_channel"
        )
        print(
            f"  all-channel RMSE: {payload['metrics'][variant_name]['groups']['all_channels']['rmse']:.10g}",
            flush=True,
        )

    if not args.skip_scaling_ablation:
        prior_forced = MODEL_REGISTRY[config["model"]["name"]](config).to(args.device)
        prior_forced.load_state_dict(prior_state)
        prior_forced.ODE_bb.set_scaling_factor(torch.ones_like(prior_forced.ODE_bb.scaling_factor))
        pred_forced, target_forced = predict_flat(prior_forced, loader)
        forced_all = rmse_and_mse(pred_forced, target_forced, list(range(len(response_channels))))
        correct_all = payload["metrics"]["full_physics_plus_nn"]["groups"]["all_channels"]
        payload["scaling_ablation"] = {
            "loaded_scaling_factor": prior_model.ODE_bb.scaling_factor.detach().cpu().tolist(),
            "prior_correct_all_rmse": correct_all["rmse"],
            "prior_forced_unit_all_rmse": forced_all["rmse"],
            "prior_forced_unit_all_mse": forced_all["mse"],
        }

    json_path = args.output_dir / "rlc_no_prior_physics_only_channel_ablation.json"
    csv_path = args.output_dir / "rlc_no_prior_physics_only_channel_ablation.csv"
    md_path = args.output_dir / "rlc_no_prior_physics_only_channel_ablation.md"
    tex_path = args.output_dir / "rlc_no_prior_physics_only_channel_ablation_table.tex"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(csv_path, payload)
    write_markdown(md_path, payload)
    write_latex_table(tex_path, payload)

    symbolic = payload["metrics"]["full_physics_plus_nn"]["groups"]["symbolic_channels"]["rmse"]
    physics = payload["metrics"]["physics_only_with_learned_physics_params"]["groups"]["symbolic_channels"][
        "rmse"
    ]
    return {
        "json": json_path,
        "csv": csv_path,
        "markdown": md_path,
        "latex_table": tex_path,
        "physics_only_symbolic_rmse": physics,
        "prior_full_symbolic_rmse": symbolic,
    }


def main() -> None:
    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("src").setLevel(logging.ERROR)

    paths = run_evaluation(parse_args())
    print(f"Wrote {paths['json']}")
    print(f"Wrote {paths['csv']}")
    print(f"Wrote {paths['markdown']}")
    print(f"Wrote {paths['latex_table']}")
    print(f"symbolic RMSE, physics-only with NO-Prior params: {paths['physics_only_symbolic_rmse']:.10g}")
    print(f"symbolic RMSE, full NO-Prior: {paths['prior_full_symbolic_rmse']:.10g}")


if __name__ == "__main__":
    main()
