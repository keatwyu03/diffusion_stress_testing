"""
In-sample analysis script with configurable sampling parameters
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
    """Main execution function for in-sample analysis"""
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
        run_name = f"insample_{q_suffix}_steps{config.conditional.num_steps}_stoch{config.conditional.stoch}"
        if args.run_suffix:
            run_name += f"_{args.run_suffix}"

        wandb_config = {
            "seed": config.seed,
            "analysis_type": "in-sample",
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
            tags=["in-sample", q_suffix] + (config.wandb.tags or []),
            config=wandb_config,
        )
        use_wandb = True
        print(f"Wandb initialized: {wandb.run.url}")

    # Set seed
    set_seed(config.seed)

    print("=" * 60)
    print("IN-SAMPLE ANALYSIS")
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

    # ==================== Extract Train Set Events ====================
    print("\n[3/5] Extracting train set events...")

    X_train = data_processor.X_train
    asset_sums_train = X_train.sum(dim=2)

    # Event mask must come from the real macro series (via get_z_windows),
    # not from X, which is stock-returns-only and has no macro channel.
    Z_start_train, Z_end_train, valid_idx_train = data_processor.get_z_windows()
    if config.hfunction.event_type == "change":
        metric_train = (Z_end_train - Z_start_train).abs()
        event_valid_train = metric_train >= config.hfunction.event_threshold
    elif config.hfunction.event_type == "absval":
        metric_train = Z_end_train.abs()
        event_valid_train = metric_train >= config.hfunction.event_threshold
    else:
        raise NotImplementedError(
            f"event_type={config.hfunction.event_type!r} not supported by the "
            "macro-based mask; only 'change' and 'absval' are implemented."
        )
    mask_train = torch.zeros(X_train.shape[0], dtype=torch.bool)
    mask_train[valid_idx_train] = event_valid_train

    event_asset_sums_train = asset_sums_train[mask_train]
    N_event_train = event_asset_sums_train.shape[0]

    print(f"Train set size: {X_train.shape[0]}")
    print(f"Number of events in train set: {N_event_train}")

    # ==================== Pretrain Baseline (if needed) ====================
    # Generate BEFORE conditional sampling so N_pretrain drives conditional count.
    # Cache is keyed by stoch so ODE and SDE baselines are stored separately.
    if args.compare_pretrain:
        cache_path = args.pretrain_cache
        if cache_path:
            base, ext = os.path.splitext(cache_path)
            cache_path = f"{base}_stoch{config.conditional.stoch}{ext}"
        if cache_path and os.path.exists(cache_path):
            pretrain_events = torch.load(cache_path)
            print(f"Loaded pretrain cache: {pretrain_events.shape} from {cache_path}")
        else:
            n_oversample = N_event_train * args.pretrain_oversample
            print(f"Generating {n_oversample} pretrain samples (stoch={config.conditional.stoch}) for baseline filtering...")
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
            if cache_path:
                torch.save(pretrain_events, cache_path)
                print(f"Saved pretrain cache to {cache_path}")
        N_cond = pretrain_events.shape[0]
        print(f"Pretrain events: {N_cond} → will generate {N_cond} conditional samples")
    else:
        N_cond = N_event_train

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

    # Generate conditional samples — count matches pretrain or real train events
    print(f"Generating {N_cond} conditional samples for in-sample events...")
    generated_samples_train = cond_generator.generate(
        num_samples=N_cond,
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
    print("Analyzing generated samples (in-sample)...")
    gen_mv_train, gen_rp_train, gen_avg_train = portfolio_analyzer.analyze_samples(generated_samples_train)

    # Analyze baseline: pretrain_events already generated above, or real train set
    if args.compare_pretrain:
        real_mv_train, real_rp_train, real_avg_train = portfolio_analyzer.analyze_samples(pretrain_events)
        baseline_label = "PRETRAIN (filtered)"
        baseline_tag   = "pretrain_filtered"
    else:
        print("Analyzing train set...")
        real_mv_train, real_rp_train, real_avg_train = portfolio_analyzer.analyze_test_set(X_train, mask_train, start_weekdays=data_processor.start_weekdays_train)
        baseline_label = "REAL DATA"
        baseline_tag   = "real_data"

    # Print statistics
    print("\n" + "=" * 60)
    print("IN-SAMPLE PORTFOLIO COMPARISON STATISTICS")
    print("=" * 60)
    portfolio_analyzer.summarize_statistics("GENERATED (in-sample)", gen_mv_train, gen_rp_train, gen_avg_train)
    portfolio_analyzer.summarize_statistics(baseline_label, real_mv_train, real_rp_train, real_avg_train)

    # Plot comparison
    results_dir = args.results_dir or "results"
    os.makedirs(results_dir, exist_ok=True)
    q_suffix = "with_q" if args.use_q_model else "no_q"
    plot_filename = f"portfolio_insample_{q_suffix}_steps{config.conditional.num_steps}_stoch{config.conditional.stoch}"
    if args.run_suffix:
        plot_filename += f"_{args.run_suffix}"
    plot_filename += ".png"

    plot_path_train = os.path.join(results_dir, plot_filename)
    portfolio_analyzer.plot_comparison(
        gen_mv_train, gen_rp_train, gen_avg_train,
        real_mv_train, real_rp_train, real_avg_train,
        save_path=plot_path_train,
        real_label=baseline_label,
    )

    # Append stats row to master in-sample CSV
    stats_df = portfolio_analyzer.build_stats_df(
        gen_mv_train, gen_rp_train, gen_avg_train,
        real_mv_train, real_rp_train, real_avg_train,
    )
    stats_df.insert(0, "baseline",    baseline_tag)
    stats_df.insert(0, "use_q_model", args.use_q_model)
    stats_df.insert(0, "eta",         config.conditional.eta)
    stats_df.insert(0, "stoch",       config.conditional.stoch)
    csv_path = os.path.join(results_dir, "insample_stats.csv")
    write_header = not os.path.exists(csv_path)
    stats_df.to_csv(csv_path, mode="a", header=write_header, index=False, float_format="%.6f")
    print(f"Stats appended to {csv_path}")

    # Log results to wandb
    if use_wandb:
        import wandb

        wandb.log({
            "insample/gen_mv_mean": sum(gen_mv_train) / len(gen_mv_train),
            "insample/gen_mv_std": torch.tensor(gen_mv_train).std().item(),
            "insample/gen_rp_mean": sum(gen_rp_train) / len(gen_rp_train),
            "insample/gen_rp_std": torch.tensor(gen_rp_train).std().item(),
            "insample/gen_avg_mean": sum(gen_avg_train) / len(gen_avg_train),
            "insample/gen_avg_std": torch.tensor(gen_avg_train).std().item(),
            "insample/real_mv_mean": sum(real_mv_train) / len(real_mv_train),
            "insample/real_mv_std": torch.tensor(real_mv_train).std().item(),
            "insample/real_rp_mean": sum(real_rp_train) / len(real_rp_train),
            "insample/real_rp_std": torch.tensor(real_rp_train).std().item(),
            "insample/real_avg_mean": sum(real_avg_train) / len(real_avg_train),
            "insample/real_avg_std": torch.tensor(real_avg_train).std().item(),
            "insample/n_events": N_event_train,
        })

        wandb.log({"insample/portfolio_comparison": wandb.Image(plot_path_train)})
        wandb.finish()

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print(f"Results saved to: {plot_path_train}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="In-sample analysis with configurable sampling parameters"
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
    parser.add_argument(
        "--pretrain-cache",
        type=str,
        default=None,
        help="Path to cache pretrain events tensor (.pt); loaded if exists, saved on first run",
    )

    args = parser.parse_args()
    main(args)
