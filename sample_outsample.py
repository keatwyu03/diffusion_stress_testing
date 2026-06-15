"""
Out-of-sample analysis script with configurable sampling parameters
"""
import argparse
import torch
import os
from dataclasses import asdict

from config import get_default_config
from data import DataProcessor
from models import DiffusionModel, HFunctionTrainer, ConditionalGenerator
from utils import PortfolioAnalyzer, set_seed


def main(args):
    """Main execution function for out-of-sample analysis"""
    # Get configuration
    config = get_default_config()

    # Override sampling parameters from command line
    if args.num_steps is not None:
        config.conditional.num_steps = args.num_steps
    if args.stoch is not None:
        config.conditional.stoch = args.stoch
    if args.eta is not None:
        config.conditional.eta = args.eta
    if args.batch_size is not None:
        config.conditional.batch_size = args.batch_size

    # Set use_q_model based on flag
    config.conditional.use_q_model = args.use_q_model

    # Handle wandb
    if args.no_wandb:
        config.wandb.enabled = False

    # Initialize wandb if enabled
    use_wandb = False
    if config.wandb.enabled:
        import wandb

        # Create run name
        q_suffix = "with_q" if args.use_q_model else "no_q"
        run_name = f"outsample_{q_suffix}_steps{config.conditional.num_steps}_stoch{config.conditional.stoch}"
        if args.run_suffix:
            run_name += f"_{args.run_suffix}"

        wandb_config = {
            "seed": config.seed,
            "analysis_type": "out-of-sample",
            "use_q_model": args.use_q_model,
            "num_steps": config.conditional.num_steps,
            "stoch": config.conditional.stoch,
            "eta": config.conditional.eta,
            "batch_size": config.conditional.batch_size,
        }

        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=run_name,
            tags=["out-of-sample", q_suffix] + (config.wandb.tags or []),
            config=wandb_config,
        )
        use_wandb = True
        print(f"Wandb initialized: {wandb.run.url}")

    # Set seed
    set_seed(config.seed)

    print("=" * 60)
    print("OUT-OF-SAMPLE ANALYSIS")
    print(f"use_q_model: {config.conditional.use_q_model}")
    print(f"num_steps: {config.conditional.num_steps}")
    print(f"stoch: {config.conditional.stoch}")
    print(f"eta: {config.conditional.eta}")
    print(f"batch_size: {config.conditional.batch_size}")
    print("=" * 60)

    # ==================== Data Processing ====================
    print("\n[1/5] Loading data...")

    data_processor = DataProcessor(
        csv_path=config.data.csv_path,
        tickers=config.data.tickers,
        weekday_col=config.data.weekday_col,
        seq_len=config.data.seq_len,
        test_days=config.data.test_days,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        train_end_date=config.data.train_end_date,
        winsorize_lower=config.data.winsorize_lower,
        winsorize_upper=config.data.winsorize_upper,
    )
    data_processor.process_all()

    print(f"Event threshold: {config.hfunction.event_threshold:.4f} std ({config.hfunction.event_type})")

    # ==================== Load Models ====================
    print("\n[2/5] Loading models from checkpoints...")

    # Load diffusion model
    diffusion_model = DiffusionModel(
        in_channels=config.diffusion.in_channels,
        out_channels=config.diffusion.out_channels,
        sample_size=config.diffusion.sample_size,
        layers_per_block=config.diffusion.layers_per_block,
        block_out_channels=config.diffusion.block_out_channels,
        b_min=config.diffusion.b_min,
        b_max=config.diffusion.b_max,
        device=config.diffusion.device,
        arch=config.diffusion.arch,
        embed_dim=config.diffusion.embed_dim,
        n_heads=config.diffusion.n_heads,
        n_layers=config.diffusion.n_layers,
        cond_dim=config.diffusion.cond_dim,
    )
    diffusion_model.load("ckpt_new/diffusion_model.pt")
    print("Loaded diffusion model")

    # Load H-function
    h_trainer = HFunctionTrainer(
        asset_dim=config.hfunction.asset_dim,
        time_steps=config.hfunction.time_steps,
        embed_dim=config.hfunction.embed_dim,
        event_asset_idx=config.hfunction.event_asset_idx,
        event_window=config.hfunction.event_window,
        event_threshold=config.hfunction.event_threshold,
        device=config.hfunction.device,
        event_type=config.hfunction.event_type,
        constraint_mode=config.hfunction.constraint_mode,
        reward_sharpness=config.hfunction.reward_sharpness,
        arch=config.hfunction.arch,
        n_heads=config.hfunction.n_heads,
        n_layers=config.hfunction.n_layers,
        cond_dim=config.hfunction.cond_dim,
    )
    h_trainer.load("ckpt_new/hfunction.pt")
    print("Loaded H-function")

    # ==================== Extract Test Set Events ====================
    print("\n[3/5] Extracting test set events...")

    X_test = data_processor.X_test
    asset_sums_test = X_test.sum(dim=2)

    last_window_test = X_test[:, -config.hfunction.event_window:, config.hfunction.event_asset_idx]
    if config.hfunction.event_type == "sum":
        metric_test = last_window_test.sum(dim=1)
        mask_test = metric_test <= config.hfunction.event_threshold
    elif config.hfunction.event_type == "change":
        metric_test = (last_window_test[:, -1] - last_window_test[:, 0]).abs()
        mask_test = metric_test >= config.hfunction.event_threshold
    elif config.hfunction.event_type == "absval":
        metric_test = last_window_test[:, -1].abs()
        mask_test = metric_test >= config.hfunction.event_threshold

    event_asset_sums_test = asset_sums_test[mask_test]
    N_event_test = event_asset_sums_test.shape[0]

    print(f"Test set size: {X_test.shape[0]}")
    print(f"Number of events in test set: {N_event_test}")

    # ==================== Conditional Generation ====================
    print("\n[4/5] Generating conditional samples...")

    cond_generator = ConditionalGenerator(
        score_model=diffusion_model.model,
        h_model=h_trainer.model,
        diffusion_coeff_fn=diffusion_model.diffusion_coeff_fn,
        drift_coeff_fn=diffusion_model.drift_coeff_fn,
        make_vp_std_grid_fn=DiffusionModel.make_vp_std_grid,
        b_min=config.diffusion.b_min,
        b_max=config.diffusion.b_max,
        device=config.conditional.device,
    )

    # Load Q-model if needed
    if config.conditional.use_q_model:
        q_model_path = "ckpt_new/q_model.pt"
        if os.path.exists(q_model_path):
            cond_generator.load_q_model(
                q_model_path,
                embed_dim=config.conditional.q_embed_dim,
                n_heads=config.conditional.q_n_heads,
                n_layers=config.conditional.q_n_layers,
                cond_dim=config.conditional.q_cond_dim,
            )
            print("Loaded Q-model")
        else:
            print(f"Warning: Q-model not found at {q_model_path}")
            print("Please train Q-model first or run without --use-q-model flag")
            return

    # Generate conditional samples for TEST set events
    print(f"Generating {N_event_test} conditional samples for out-of-sample events...")
    generated_samples_test = cond_generator.generate(
        num_samples=N_event_test,
        batch_size=config.conditional.batch_size,
        num_steps=config.conditional.num_steps,
        stoch=config.conditional.stoch,
        eta=config.conditional.eta,
        use_q_model=config.conditional.use_q_model,
    )

    # ==================== Portfolio Analysis ====================
    print("\n[5/5] Analyzing portfolios...")

    portfolio_analyzer = PortfolioAnalyzer(
        data_processor=data_processor,
        window_for_cov=config.portfolio.window_for_cov,
        last_days_sum=config.portfolio.last_days_sum,
        config=config,
    )

    # Analyze generated samples
    print("Analyzing generated samples (out-of-sample)...")
    gen_mv_test, gen_rp_test, gen_avg_test = portfolio_analyzer.analyze_samples(generated_samples_test)

    # Analyze baseline: real test set or pretrain-filtered samples
    if args.compare_pretrain:
        n_oversample = N_event_test * args.pretrain_oversample
        print(f"Generating {n_oversample} pretrain samples for baseline filtering...")
        pretrain_all = diffusion_model.sample(
            batch_size=n_oversample,
            num_steps=config.conditional.num_steps,
            stoch=config.conditional.stoch,
            eps=config.diffusion.eps,
        )
        pt_last = pretrain_all[:, config.hfunction.event_asset_idx, -config.hfunction.event_window:]
        if config.hfunction.event_type == "sum":
            pt_mask = pt_last.sum(dim=1) <= config.hfunction.event_threshold
        elif config.hfunction.event_type == "change":
            pt_mask = (pt_last[:, -1] - pt_last[:, 0]).abs() >= config.hfunction.event_threshold
        elif config.hfunction.event_type == "absval":
            pt_mask = pt_last[:, -1].abs() >= config.hfunction.event_threshold
        pretrain_events = pretrain_all[pt_mask]
        print(f"Pretrain events: {pretrain_events.shape[0]} / {n_oversample}")
        real_mv_test, real_rp_test, real_avg_test = portfolio_analyzer.analyze_samples(pretrain_events)
        baseline_label = "PRETRAIN (filtered)"
        baseline_tag   = "pretrain_filtered"
    else:
        print("Analyzing test set...")
        real_mv_test, real_rp_test, real_avg_test = portfolio_analyzer.analyze_test_set(X_test, mask_test, start_weekdays=data_processor.start_weekdays_test)
        baseline_label = "REAL DATA"
        baseline_tag   = "real_data"

    # Print statistics
    print("\n" + "=" * 60)
    print("OUT-OF-SAMPLE PORTFOLIO COMPARISON STATISTICS")
    print("=" * 60)
    portfolio_analyzer.summarize_statistics("GENERATED (out-of-sample)", gen_mv_test, gen_rp_test, gen_avg_test)
    portfolio_analyzer.summarize_statistics(baseline_label, real_mv_test, real_rp_test, real_avg_test)

    # Plot comparison
    results_dir = args.results_dir or "results"
    os.makedirs(results_dir, exist_ok=True)
    q_suffix = "with_q" if args.use_q_model else "no_q"
    plot_filename = f"portfolio_outsample_{q_suffix}_steps{config.conditional.num_steps}_stoch{config.conditional.stoch}"
    if args.run_suffix:
        plot_filename += f"_{args.run_suffix}"
    plot_filename += ".png"

    plot_path_test = os.path.join(results_dir, plot_filename)
    portfolio_analyzer.plot_comparison(
        gen_mv_test, gen_rp_test, gen_avg_test,
        real_mv_test, real_rp_test, real_avg_test,
        save_path=plot_path_test,
        real_label=baseline_label,
    )

    # Append stats row to master out-of-sample CSV
    stats_df = portfolio_analyzer.build_stats_df(
        gen_mv_test, gen_rp_test, gen_avg_test,
        real_mv_test, real_rp_test, real_avg_test,
    )
    stats_df.insert(0, "baseline",    baseline_tag)
    stats_df.insert(0, "use_q_model", args.use_q_model)
    stats_df.insert(0, "eta",         config.conditional.eta)
    stats_df.insert(0, "stoch",       config.conditional.stoch)
    csv_path = os.path.join(results_dir, "outsample_stats.csv")
    write_header = not os.path.exists(csv_path)
    stats_df.to_csv(csv_path, mode="a", header=write_header, index=False, float_format="%.6f")
    print(f"Stats appended to {csv_path}")

    # Log results to wandb
    if use_wandb:
        import wandb

        wandb.log({
            "outsample/gen_mv_mean": sum(gen_mv_test) / len(gen_mv_test),
            "outsample/gen_mv_std": torch.tensor(gen_mv_test).std().item(),
            "outsample/gen_rp_mean": sum(gen_rp_test) / len(gen_rp_test),
            "outsample/gen_rp_std": torch.tensor(gen_rp_test).std().item(),
            "outsample/gen_avg_mean": sum(gen_avg_test) / len(gen_avg_test),
            "outsample/gen_avg_std": torch.tensor(gen_avg_test).std().item(),
            "outsample/real_mv_mean": sum(real_mv_test) / len(real_mv_test),
            "outsample/real_mv_std": torch.tensor(real_mv_test).std().item(),
            "outsample/real_rp_mean": sum(real_rp_test) / len(real_rp_test),
            "outsample/real_rp_std": torch.tensor(real_rp_test).std().item(),
            "outsample/real_avg_mean": sum(real_avg_test) / len(real_avg_test),
            "outsample/real_avg_std": torch.tensor(real_avg_test).std().item(),
            "outsample/n_events": N_event_test,
        })

        wandb.log({"outsample/portfolio_comparison": wandb.Image(plot_path_test)})
        wandb.finish()

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print(f"Results saved to: {plot_path_test}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Out-of-sample analysis with configurable sampling parameters"
    )

    # Sampling parameters
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Number of sampling steps (default: use config value)",
    )
    parser.add_argument(
        "--stoch",
        type=float,
        default=None,
        help="Stochasticity parameter (default: use config value)",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=None,
        help="Eta parameter for conditional guidance (default: use config value)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for generation (default: use config value)",
    )

    # Q-model flag
    parser.add_argument(
        "--use-q-model",
        action="store_true",
        help="Use Q-model for faster sampling",
    )

    # Wandb flag
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )

    # Results directory
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory to save result plots (default: results/)",
    )

    # Run suffix for organizing multiple runs
    parser.add_argument(
        "--run-suffix",
        type=str,
        default="",
        help="Suffix to add to run name and output files",
    )

    # Pretrain comparison flag
    parser.add_argument(
        "--compare-pretrain",
        action="store_true",
        help="Compare against pretrain-filtered samples instead of real data",
    )
    parser.add_argument(
        "--pretrain-oversample",
        type=int,
        default=10,
        help="Oversample factor for pretrain filtering (default: 10x event count)",
    )

    args = parser.parse_args()
    main(args)
