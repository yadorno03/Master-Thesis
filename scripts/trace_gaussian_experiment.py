"""Trace the Gaussian notebook's six stages using its existing implementations.

Run from the repository root:
    python3.12 scripts/trace_gaussian_experiment.py --mode smoke
    python3.12 scripts/trace_gaussian_experiment.py --mode standard

See synthetic_dfm_conditional_benchmark.ipynb for formulas and function contracts.
Smoke keeps 100 assets and eight factors; it checks execution, not convergence.
"""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kstest
import torch

from guided_evaluation import conditional_diagnostics
from reference import FrozenDFMReference
from sampling import generate_samples
from synthetic_benchmark.configuration import (
    Config, ConditionalAssessmentConfig, write_json,
)
from synthetic_benchmark.unconditional import (
    initialize_unconditional_run, prepare_unconditional_case, fit_and_save_dfm,
    evaluate_unconditional_samples, save_unconditional_case,
    save_unconditional_summary,
)
from synthetic_benchmark.conditional import (
    initialize_h, sample_h_training_data, fit_and_evaluate_h, save_h_training,
    load_trained_h, sample_h_guided, filter_conditional_samples,
    assess_conditional_samples, save_conditional_samples,
)
from synthetic_benchmark.reporting import (
    plot_h_validation, conditional_sample_summary, extend_conditional_summary,
    plot_conditional_comparison, plot_conditional_ecdf,
    unconditional_recovery_report,
)


def conditional_truth_errors(samples, population, threshold):
    """Assess each saved draw against exact conditional moments and portfolio CDF."""
    if len(samples) < 2:
        return {"n": len(samples), "status": "unavailable: fewer than two rows"}
    mean, covariance = population.moments(c=threshold)
    # The denominator uses unconditional RMS scale, as in the notebook.
    scale = np.sqrt(np.trace(population.moments()[1]))
    empirical_covariance = np.cov(samples, rowvar=False)
    portfolio = samples @ population.w
    return {
        "n": len(samples),
        "status": "available",
        "event_rate": float(np.mean(portfolio <= threshold)),
        "mean_error": float(np.linalg.norm(samples.mean(0) - mean) / scale),
        "covariance_error": float(
            np.linalg.norm(empirical_covariance - covariance) / np.linalg.norm(covariance)
        ),
        "diversity_trace_ratio": float(np.trace(empirical_covariance) / np.trace(covariance)),
        "portfolio_population_ks": float(kstest(
            portfolio, lambda values: population.cdf(values, c=threshold),
        ).statistic),
    }


def run(mode, output=None):
    started = time.perf_counter()
    # Unlike the notebook's 12-asset smoke preset, preserve the 100-asset market.
    config = replace(Config.preset(mode), d=100, dfm_initialization='pca',
                     dfm_time_sampling='log_balanced', lr=.001,
                     dfm_loading_lr_scale=.1, dfm_loading_warmup_epochs=100)
    small = mode == "smoke"
    h_epochs = 10 if small else 200
    counts = {"train": 2048 if small else 10000,
              "validation": 1024 if small else 2000,
              "test": 1024 if small else 2000}
    h_seeds = {"train": 32001, "validation": 32002, "test": 32003, "shuffle": 32004}
    assessment_settings = ConditionalAssessmentConfig(
        rejection_n=128 if small else 1000,
        rejection_batch_size=256,
        rejection_max_proposals=10000 if small else 50000,
        rejection_reference_max_proposals=10000 if small else 50000,
        rejection_seed=42003, rejection_reference_seed=42004,
        truth_reference_seed=42005, projection_seed=42006, matching_seed=42007,
        projections=64,
    )
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = ROOT / "artifacts" / "synthetic_unconditional_benchmark" / f"trace_{mode}_{stamp}"
    output = Path(output).resolve()
    initialize_unconditional_run(output, config)
    print(f"Artifacts: {output}", flush=True)

    # 1. Population: R ~ N(Bm, B S B^T + diag(D)); no latent truth enters fitting.
    print("\n1. Synthetic population", flush=True)
    seed, case, tail = config.seeds[0], "gaussian", 0.05
    folder, streams, population = prepare_unconditional_case(output, config, case, seed)
    train = population.sample(config.n_train, streams.np("train"))
    validation = population.sample(config.n_val, streams.np("validation"))
    truth = population.sample(config.n_eval, streams.np("truth"))
    truth2 = population.sample(config.n_eval, streams.np("truth2"))
    threshold = population.threshold(tail)
    np.testing.assert_allclose(population.B.T @ population.B, np.eye(config.k), atol=1e-12)
    np.testing.assert_allclose(population.cdf(threshold), tail, atol=1e-12)
    print(f"train={train.shape}, validation={validation.shape}; 5% threshold={threshold:.6f}")

    # 2. Learn the OU score with DSM, select by validation, then sample reverse OU.
    print("\n2. Unconditional DFM", flush=True)
    model, history, selected, fit_seconds = fit_and_save_dfm(
        train, validation, config, streams, population, case, folder,
    )
    reference = FrozenDFMReference(model)
    generation_started = time.perf_counter()
    generated = generate_samples(
        reference, config.n_eval, num_steps=config.steps,
        generator=streams.torch("generated", config.device),
    ).cpu().numpy()
    result, baseline = evaluate_unconditional_samples(
        generated, truth, truth2, population, reference, config, case, seed,
        fit_seconds, time.perf_counter() - generation_started,
    )
    save_unconditional_case(folder, case, generated, truth, truth2, population, result, baseline)
    metrics, _ = save_unconditional_summary([result, baseline], output, started)
    report = unconditional_recovery_report(metrics.iloc[0], metrics.iloc[1], config)
    (output / "conclusions.md").write_text(report)
    ax = pd.DataFrame(history).plot(x="epoch", y=["train", "validation"])
    ax.set_ylabel("Denoising score-matching loss")
    ax.figure.savefig(folder / "dfm_loss.png", bbox_inches="tight")
    plt.close(ax.figure)
    print(f"Selected DFM epoch: {selected}; generated={generated.shape}")

    # 3. Y_t = G_t X_t = (Z_t,U_t), with U_tau = w^T X_tau, tau=T-t0.
    print("\n3. Conditioning coordinates", flush=True)
    event, h, _ = initialize_h(reference, population, tail, (64, 64), 2027)
    parameter = next(reference.parameters())
    probe = torch.as_tensor(train[:8], dtype=parameter.dtype, device=parameter.device)
    torch.testing.assert_close(h.ell(h.horizon), h.w)
    terminal_u = h.features(h.horizon, probe)[:, -1]
    torch.testing.assert_close(terminal_u, probe @ h.w)
    frozen_before = {name: value.detach().clone() for name, value in reference.state_dict().items()}
    print(f"G(0)={tuple(h.G(0.).shape)}; H input={config.k + 2}; terminal portfolio identity checked")

    # 4. Unguided reference trajectories supply terminal indicators at ALL earlier times.
    print("\n4. Learning H", flush=True)
    h_started = time.perf_counter()
    data = sample_h_training_data(
        h, counts.items(), steps=config.steps, seeds=h_seeds,
        path_batch_size=256, device=config.device,
    )
    print(f"features={tuple(data['train'].features.shape)}, labels={tuple(data['train'].labels.shape)}")
    selection, h_history, training_probability, test_metrics = fit_and_evaluate_h(
        h, reference, data, frozen_before, epochs=h_epochs, batch_size=64,
        lr=1e-3, seeds=h_seeds, device=config.device,
    )
    curve, _, h_output = save_h_training(
        output, h, population, data, selection, h_history, training_probability, test_metrics,
        case=case, seed=seed, tail=tail, n_train=counts["train"],
        n_validation=counts["validation"], n_test=counts["test"], epochs=h_epochs,
        lr=1e-3, batch_size=64, path_batch_size=256, steps=config.steps,
        stream_seeds=h_seeds, initialization_seed=2027,
        reference_checkpoint=folder / "dfm.pt", started=h_started,
    )
    plot_h_validation(curve, selection["best_epoch"], h_output)
    print(f"Selected H epoch: {selection['best_epoch']}; held-out losses: {test_metrics}")
    del data

    # 5. Accept N(0,I) starts with probability H(0,G_0 x); add G_t^T grad_y log H.
    print("\n5. Conditional generation", flush=True)
    h, metadata, guidance, population = load_trained_h(h_output, config.device)
    t = h.horizon / 2
    x = probe.detach().clone().requires_grad_(True)
    direct_gradient = torch.autograd.grad(h.log_prob(t, h.features(t, x)).sum(), x)[0]
    lifted_gradient = guidance(x.detach(), t)
    torch.testing.assert_close(lifted_gradient, direct_gradient)
    diagnostics = {}
    guided = sample_h_guided(
        h, guidance, n_samples=config.n_eval, batch_size=256, steps=config.steps,
        seed=42001, diagnostics=diagnostics,
    )
    conditional_truth = population.sample(config.n_eval, np.random.default_rng(42002), c=event.C)
    guided_p, truth_p, mask, filtered, filtered_p = filter_conditional_samples(
        guided, conditional_truth, population, event.C, config.n_eval, config.d,
    )

    # 6. Compare raw/filtered guidance with population truth AND the numerical DFM law.
    print("\n6. Conditional distribution assessment", flush=True)
    assessment = assess_conditional_samples(
        h.reference, population, event.C, guided, conditional_truth, filtered,
        steps=config.steps, settings=assessment_settings,
        existing_seeds=(42001, 42002, 2027, *h_seeds.values()),
    )
    summary = conditional_sample_summary(guided_p, filtered_p, truth_p, event.C, config.n_eval)
    summary = extend_conditional_summary(
        summary, assessment["samples"], population.w, event.C, assessment["rejection_diagnostics"],
    )
    sample_output = save_conditional_samples(
        h_output, metadata, population, event.C, guided, conditional_truth, filtered, mask,
        summary, n_samples=config.n_eval, steps=config.steps, batch_size=256,
        guided_seed=42001, truth_seed=42002, assessment=assessment,
        guided_sampling_diagnostics=diagnostics,
    )
    exact_metrics = {}
    for name, samples in assessment["samples"].items():
        rejection_status = assessment["rejection_diagnostics"].get(name, {}).get("status", "")
        exact_metrics[name] = (
            {"n": len(samples), "status": f"unavailable: {rejection_status}"}
            if rejection_status.startswith("nonfinite")
            else conditional_truth_errors(samples, population, event.C)
        )
    write_json(sample_output / "exact_conditional_metrics.json", exact_metrics)
    rejected = assessment["samples"]["rejection"]
    rejection_status = assessment["rejection_diagnostics"]["rejection"]["status"]
    if len(rejected) >= 2 and rejection_status in ("complete", "partial"):
        write_json(sample_output / "raw_guided_diagnostics.json", conditional_diagnostics(
            guided, rejected, population.w, event.C, seed=42006, projections=64,
        ))
    plot_conditional_comparison(guided, conditional_truth, filtered, population, event.C, case, sample_output)
    plot_conditional_ecdf(
        assessment["samples"], population, event.C, case,
        assessment["comparison_metrics"], sample_output, assessment["rejection_diagnostics"],
    )
    plt.close("all")
    write_json(output / "trace_checks.json", {
        "terminal_portfolio_max_abs_error": float((terminal_u - probe @ h.w).abs().max()),
        "guidance_chain_rule_max_abs_error": float((lifted_gradient - direct_gradient).abs().max()),
        "reference_unchanged_during_h_training": True,
        "dfm_selected_epoch": selected,
        "h_selected_epoch": selection["best_epoch"],
        "h_train_features_shape": [counts["train"], config.steps, config.k + 1],
        "sample_output": str(sample_output),
        "total_seconds": time.perf_counter() - started,
        "interpretation": "Smoke verifies the workflow; assess distribution errors before claiming recovery.",
    })
    print(summary.to_string(index=False))
    print(assessment["comparison_metrics"].to_string(index=False))
    print("Exact conditional moment/CDF errors:\n", pd.DataFrame(exact_metrics).T.to_string())
    print(f"\nComplete: {output}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "standard"), default="smoke")
    parser.add_argument("--output", type=Path, help="New artifact directory; existing directories are rejected")
    arguments = parser.parse_args()
    run(arguments.mode, arguments.output)
