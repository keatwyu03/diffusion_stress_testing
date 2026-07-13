"""
Main script for Conditional Diffusion Generation in Financial Time Series
"""
import argparse
import torch
import os
from dataclasses import asdict

from config import get_default_config
from data import DataProcessor
from models import DiffusionModel, HFunctionTrainer, HFunctionDirectTrainer, ConditionalGenerator, EllTrainer, HFunctionTwoStepTrainer
from utils import PortfolioAnalyzer, set_seed



def init_wandb(config):
    """Initialize Weights & Biases logging"""
    if not config.wandb.enabled:
        return False

    import wandb

    # Flatten config for wandb
    wandb_config = {
        "seed": config.seed,
        **{f"data/{k}": v for k, v in asdict(config.data).items()},
        **{f"diffusion/{k}": v for k, v in asdict(config.diffusion).items()},
        **{f"hfunction/{k}": v for k, v in asdict(config.hfunction).items()},
        **{f"conditional/{k}": v for k, v in asdict(config.conditional).items()},
        **{f"portfolio/{k}": v for k, v in asdict(config.portfolio).items()},
    }

    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.wandb.run_name,
        tags=config.wandb.tags,
        notes=config.wandb.notes,
        config=wandb_config,
    )

    return True


def main(args):
    """Main execution function"""
    # Get configuration
    config = get_default_config()

    # Handle wandb disable flag
    if args.no_wandb:
        config.wandb.enabled = False

    # Initialize wandb
    use_wandb = init_wandb(config)
    if use_wandb:
        import wandb
        print(f"Wandb initialized: {wandb.run.url}")

    # Set seed
    set_seed(config.seed)

    print(f"[DEBUG] PyTorch version: {torch.__version__}", flush=True)
    print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[DEBUG] CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

    # ==================== Data Processing ====================
    print("\n" + "=" * 60)
    print("STEP 1: Data Processing")
    print("=" * 60)

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

    # event_threshold is specified as "top X% of |Z_end - Z_start|" (e.g. 0.10 = top
    # 10%), computed from train data only, then converted to the equivalent raw
    # numeric cutoff. Everything downstream keeps comparing >= this raw value, unchanged.
    event_top_fraction = config.hfunction.event_threshold
    config.hfunction.event_threshold = data_processor.get_event_threshold_from_percentile(event_top_fraction)
    print(f"Event threshold: top {event_top_fraction:.1%} -> {config.hfunction.event_threshold:.4f} std "
          f"({config.hfunction.event_type})")

    # Derive asset count from tickers so model dims always match data
    n_assets = len(config.data.tickers) - 1
    config.diffusion.in_channels  = n_assets
    config.diffusion.out_channels = n_assets
    config.hfunction.asset_dim    = n_assets

    # ==================== Diffusion Model Training ====================
    if not args.skip_diffusion_training:
        print("\n" + "=" * 60)
        print("STEP 2: Diffusion Model Training")
        print("=" * 60)

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

        # Get training data
        train_data = data_processor.get_diffusion_data()

        # Train
        diffusion_model.train(
            train_data=train_data,
            batch_size=config.diffusion.batch_size,
            n_epochs=config.diffusion.n_epochs,
            learning_rate=config.diffusion.learning_rate,
            scheduler_patience=config.diffusion.scheduler_patience,
            scheduler_factor=config.diffusion.scheduler_factor,
            use_wandb=use_wandb,
        )

        # Save model
        os.makedirs("ckpt_new", exist_ok=True)
        diffusion_model.save("ckpt_new/diffusion_model.pt")
    else:
        print("\nSkipping diffusion training, loading from checkpoint...")
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

    # ==================== H-Function Training ====================
    # if not args.skip_hfunction_training:
    #     print("\n" + "=" * 60)
    #     print("STEP 3: H-Function Training")
    #     print("=" * 60)

    #     h_trainer = HFunctionTrainer(
    #         asset_dim=config.hfunction.asset_dim,
    #         time_steps=config.hfunction.time_steps,
    #         embed_dim=config.hfunction.embed_dim,
    #         event_asset_idx=config.hfunction.event_asset_idx,
    #         event_window=config.hfunction.event_window,
    #         event_threshold=config.hfunction.event_threshold,
    #         device=config.hfunction.device,
    #         event_type=config.hfunction.event_type,
    #         constraint_mode=config.hfunction.constraint_mode,
    #         reward_sharpness=config.hfunction.reward_sharpness,
    #         arch=config.hfunction.arch,
    #         n_heads=config.hfunction.n_heads,
    #         n_layers=config.hfunction.n_layers,
    #         cond_dim=config.hfunction.cond_dim,
    #     )

    #     # Generate training paths
    #     print("Generating training paths...")
    #     t_grid, y_grid, Y_T = diffusion_model.sample(
    #         batch_size=config.hfunction.train_batch_size,
    #         num_steps=config.diffusion.num_steps,
    #         stoch=config.hfunction.train_stoch,
    #         return_path=True,
    #     )

    #     # Train
    #     h_trainer.train(
    #         t_grid=t_grid,
    #         y_grid=y_grid,
    #         Y_T=Y_T,
    #         n_epochs=config.hfunction.n_epochs,
    #         batch_size=config.hfunction.h_mini_batch_size,
    #         learning_rate=config.hfunction.learning_rate,
    #         weight_decay=config.hfunction.weight_decay,
    #         scheduler_patience=config.hfunction.scheduler_patience,
    #         scheduler_factor=config.hfunction.scheduler_factor,
    #         use_wandb=use_wandb,
    #     )

    #     # Save model
    #     h_trainer.save("ckpt_new/hfunction.pt")
    # else:
    #     print("\nSkipping H-function training, loading from checkpoint...")
    #     h_trainer = HFunctionTrainer(
    #         asset_dim=config.hfunction.asset_dim,
    #         time_steps=config.hfunction.time_steps,
    #         embed_dim=config.hfunction.embed_dim,
    #         event_asset_idx=config.hfunction.event_asset_idx,
    #         event_window=config.hfunction.event_window,
    #         event_threshold=config.hfunction.event_threshold,
    #         device=config.hfunction.device,
    #         event_type=config.hfunction.event_type,
    #         constraint_mode=config.hfunction.constraint_mode,
    #         reward_sharpness=config.hfunction.reward_sharpness,
    #         arch=config.hfunction.arch,
    #         n_heads=config.hfunction.n_heads,
    #         n_layers=config.hfunction.n_layers,
    #         cond_dim=config.hfunction.cond_dim,
    #     )
    #     h_trainer.load("ckpt_new/hfunction.pt")

    if not args.skip_hfunction_training:
        print("\n" + "=" * 60)
        print("STEP 3: H-Function Training (Direct BCE)")
        print("=" * 60)

        X_train_direct = data_processor.get_diffusion_data()
        Z_start, Z_end, valid_idx = data_processor.get_z_windows_train_aligned()
        X_train_direct = X_train_direct[valid_idx]

        print(f"Using HFunctionTraining with {config.hfunction.one_two_step} steps")
        if config.hfunction.one_two_step == "one":
            h_trainer = HFunctionDirectTrainer(
                cfg=config.hfunction,
                b_min=config.diffusion.b_min,
                b_max=config.diffusion.b_max,
            )

            h_trainer.train(
                X_train=X_train_direct,
                Z_start=Z_start,
                Z_end=Z_end,
                use_wandb=use_wandb,
            )
            
        else:
            ell_trainer = EllTrainer(cfg=config.hfunction)
            ell_trainer.train(X_train=X_train_direct, 
                              Z_start=Z_start, 
                              Z_end=Z_end, 
                              use_wandb=use_wandb)
            
            ell_trainer.save("ckpt_new/ell_function.pt")
            h_trainer = HFunctionTwoStepTrainer(cfg=config.hfunction, 
                                                diffusion_model=diffusion_model, 
                                                ell_model=ell_trainer.model)
            h_trainer.train(use_wandb=use_wandb)

        h_trainer.save("ckpt_new/hfunction.pt")
    else:
        print("\nSkipping H-function training, loading from checkpoint...")
        if config.hfunction.one_two_step == "one":
            h_trainer = HFunctionDirectTrainer(cfg=config.hfunction, b_min=config.diffusion.b_min, b_max=config.diffusion.b_max)
            h_trainer.load("ckpt_new/hfunction.pt")
        else:
            ell_trainer = EllTrainer(cfg=config.hfunction)
            ell_trainer.load("ckpt_new/ell_function.pt")
            h_trainer = HFunctionTwoStepTrainer(cfg=config.hfunction, diffusion_model=diffusion_model, ell_model=ell_trainer.model)
            h_trainer.load("ckpt_new/hfunction.pt")
        

    # ==================== Extract Events ====================
    print("\n" + "=" * 60)
    print("STEP 4: Extract Events (Train & Test)")
    print("=" * 60)

    # Extract events from TRAIN set — event mask must come from the real macro
    # series (via get_z_windows), not from X, which is stock-returns-only and
    # has no macro channel at all.
    X_train = data_processor.X_train
    asset_sums_train = X_train.sum(dim=2)

    Z_start_train, Z_end_train, valid_idx_train = data_processor.get_z_windows_train_aligned()
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

    print("Min Change", metric_train.min().item())
    print("Max Change", metric_train.max().item())
    print("Mean Change", metric_train.mean().item())
    print("Threshold:", config.hfunction.event_threshold)

    print("X_train shape", X_train.shape)

    event_asset_sums_train = asset_sums_train[mask_train]
    N_event_train = event_asset_sums_train.shape[0]

    print(f"Train set size: {X_train.shape[0]}")
    print(f"Number of events in train set: {N_event_train}")

    # Extract events from TEST set
    X_test = data_processor.X_test
    asset_sums_test = X_test.sum(dim=2)

    Z_start_test, Z_end_test, valid_idx_test = data_processor.get_z_windows_test()
    if config.hfunction.event_type == "change":
        metric_test = (Z_end_test - Z_start_test).abs()
        event_valid_test = metric_test >= config.hfunction.event_threshold
    elif config.hfunction.event_type == "absval":
        metric_test = Z_end_test.abs()
        event_valid_test = metric_test >= config.hfunction.event_threshold
    else:
        raise NotImplementedError(
            f"event_type={config.hfunction.event_type!r} not supported by the "
            "macro-based mask; only 'change' and 'absval' are implemented."
        )
    mask_test = torch.zeros(X_test.shape[0], dtype=torch.bool)
    mask_test[valid_idx_test] = event_valid_test

    print("Test dates:", data_processor.y_dates_test[0], "to", data_processor.y_dates_test[-1])
    print("Test change stats:", metric_test.min().item(), metric_test.max().item())

    event_asset_sums_test = asset_sums_test[mask_test]
    N_event_test = event_asset_sums_test.shape[0]

    print(f"Test set size: {X_test.shape[0]}")
    print(f"Number of events in test set: {N_event_test}")

    if args.skip_conditional:
        print("\nSkipping conditional generation and portfolio analysis.")
        return

    # ==================== Conditional Generation ====================
    print("\n" + "=" * 60)
    print("STEP 5: Conditional Generation")
    print("=" * 60)

    cond_generator = ConditionalGenerator(
        score_model=diffusion_model.model,
        h_model=h_trainer.model,
        diffusion_coeff_fn=diffusion_model.diffusion_coeff_fn,
        drift_coeff_fn=diffusion_model.drift_coeff_fn,
        make_vp_std_grid_fn=DiffusionModel.make_vp_std_grid,
        b_min=config.diffusion.b_min,
        b_max=config.diffusion.b_max,
        device=config.conditional.device,
        h_t_max=config.hfunction.h_t_max,
        pos_weight=getattr(h_trainer, "pos_weight", 1.0),
    )

    # Optionally train Q-model
    if (config.conditional.use_q_model or args.train_q_model) and not args.skip_qmodel_training:
        print("Training Q-model...")
        t_grid, y_grid, _ = diffusion_model.sample(
            batch_size=config.conditional.q_model_train_batch_size,
            num_steps=config.diffusion.num_steps,
            stoch=config.conditional.q_model_train_stoch,
            return_path=True,
        )
        cond_generator.train_q_model(
            t_grid=t_grid,
            y_grid=y_grid,
            n_epochs=config.conditional.q_model_epochs,
            learning_rate=config.conditional.q_model_lr,
            mini_batch_size=config.conditional.q_model_mini_batch_size,
            embed_dim=config.conditional.q_embed_dim,
            n_heads=config.conditional.q_n_heads,
            n_layers=config.conditional.q_n_layers,
            cond_dim=config.conditional.q_cond_dim,
        )
        cond_generator.save_q_model("ckpt_new/q_model.pt")

    # Use Q-model if it was trained this run or already loaded via config
    use_q_model = config.conditional.use_q_model or args.train_q_model

    # Generate conditional samples for TRAIN set events
    print(f"Generating {N_event_train} conditional samples for in-sample (train) events "
          f"(matching real event count)...")
    generated_samples_train = cond_generator.generate(
        num_samples=N_event_train,
        batch_size=config.conditional.batch_size,
        num_steps=config.conditional.num_steps,
        stoch=config.conditional.stoch,
        eta=config.conditional.eta,
        use_q_model=use_q_model,
        stop_early_steps=config.conditional.stop_early_steps,
    )
    torch.save(generated_samples_train, 'generated_samples_train.pt')

    # Generate conditional samples for TEST set events
    print(f"Generating {N_event_test} conditional samples for out-of-sample (test) events "
          f"(matching real event count)...")
    generated_samples_test = cond_generator.generate(
        num_samples=N_event_test,
        batch_size=config.conditional.batch_size,
        num_steps=config.conditional.num_steps,
        stoch=config.conditional.stoch,
        eta=config.conditional.eta,
        use_q_model=use_q_model,
        stop_early_steps=config.conditional.stop_early_steps,
    )
    torch.save(generated_samples_test, 'generated_samples_test.pt')

    # ==================== Portfolio Analysis ====================
    print("\n" + "=" * 60)
    print("STEP 6: Portfolio Analysis")
    print("=" * 60)

    portfolio_analyzer = PortfolioAnalyzer(
        data_processor=data_processor,
        window_for_cov=config.portfolio.window_for_cov,
        last_days_sum=config.portfolio.last_days_sum,
        config=config,
    )

    # ========== In-Sample (Train) Analysis ==========
    print("\n--- In-Sample (Train Set) Analysis ---")

    # Analyze generated samples for train
    print("Analyzing generated samples (in-sample)...")
    gen_mv_train, gen_rp_train, gen_avg_train = portfolio_analyzer.analyze_samples(generated_samples_train)

    # Analyze train set
    print("Analyzing train set...")
    real_mv_train, real_rp_train, real_avg_train = portfolio_analyzer.analyze_test_set(X_train, mask_train, start_weekdays=data_processor.start_weekdays_train)

    # Print statistics
    print("\n=== In-Sample Portfolio Comparison Statistics ===")
    portfolio_analyzer.summarize_statistics("GENERATED (in-sample)", gen_mv_train, gen_rp_train, gen_avg_train)
    portfolio_analyzer.summarize_statistics("REAL TRAIN", real_mv_train, real_rp_train, real_avg_train)

    # Plot comparison
    os.makedirs("results", exist_ok=True)
    plot_path_train = "results/portfolio_comparison_insample.png"
    portfolio_analyzer.plot_comparison(
        gen_mv_train, gen_rp_train, gen_avg_train,
        real_mv_train, real_rp_train, real_avg_train,
        save_path=plot_path_train
    )

    # ========== Out-of-Sample (Test) Analysis ==========
    print("\n--- Out-of-Sample (Test Set) Analysis ---")

    # Analyze generated samples for test
    print("Analyzing generated samples (out-of-sample)...")
    gen_mv_test, gen_rp_test, gen_avg_test = portfolio_analyzer.analyze_samples(generated_samples_test)

    # Analyze test set
    print("Analyzing test set...")
    real_mv_test, real_rp_test, real_avg_test = portfolio_analyzer.analyze_test_set(X_test, mask_test, start_weekdays=data_processor.start_weekdays_test)

    # Print statistics
    print("\n=== Out-of-Sample Portfolio Comparison Statistics ===")
    portfolio_analyzer.summarize_statistics("GENERATED (out-of-sample)", gen_mv_test, gen_rp_test, gen_avg_test)
    portfolio_analyzer.summarize_statistics("REAL TEST", real_mv_test, real_rp_test, real_avg_test)

    # Plot comparison
    plot_path_test = "results/portfolio_comparison_outsample.png"
    portfolio_analyzer.plot_comparison(
        gen_mv_test, gen_rp_test, gen_avg_test,
        real_mv_test, real_rp_test, real_avg_test,
        save_path=plot_path_test
    )

    # Log final results to wandb
    if use_wandb:
        import wandb

        # Log in-sample metrics
        wandb.log({
            "results/insample/gen_mv_mean": sum(gen_mv_train) / len(gen_mv_train),
            "results/insample/gen_rp_mean": sum(gen_rp_train) / len(gen_rp_train),
            "results/insample/gen_avg_mean": sum(gen_avg_train) / len(gen_avg_train),
            "results/insample/real_mv_mean": sum(real_mv_train) / len(real_mv_train),
            "results/insample/real_rp_mean": sum(real_rp_train) / len(real_rp_train),
            "results/insample/real_avg_mean": sum(real_avg_train) / len(real_avg_train),
            "results/insample/n_events": N_event_train,
        })

        # Log out-of-sample metrics
        wandb.log({
            "results/outsample/gen_mv_mean": sum(gen_mv_test) / len(gen_mv_test),
            "results/outsample/gen_rp_mean": sum(gen_rp_test) / len(gen_rp_test),
            "results/outsample/gen_avg_mean": sum(gen_avg_test) / len(gen_avg_test),
            "results/outsample/real_mv_mean": sum(real_mv_test) / len(real_mv_test),
            "results/outsample/real_rp_mean": sum(real_rp_test) / len(real_rp_test),
            "results/outsample/real_avg_mean": sum(real_avg_test) / len(real_avg_test),
            "results/outsample/n_events": N_event_test,
        })

        # Log comparison plots
        wandb.log({
            "results/portfolio_comparison_insample": wandb.Image(plot_path_train),
            "results/portfolio_comparison_outsample": wandb.Image(plot_path_test),
        })

        # Finish wandb run
        wandb.finish()

    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Conditional Diffusion Generation for Financial Time Series"
    )
    parser.add_argument(
        "--skip-diffusion-training",
        action="store_true",
        help="Skip diffusion model training and load from checkpoint",
    )
    parser.add_argument(
        "--skip-hfunction-training",
        action="store_true",
        help="Skip H-function training and load from checkpoint",
    )
    parser.add_argument(
        "--skip-qmodel-training",
        action="store_true",
        help="Skip Q-model training",
    )
    parser.add_argument(
        "--skip-conditional",
        action="store_true",
        help="Skip conditional generation and portfolio analysis",
    )
    parser.add_argument(
        "--train-q-model",
        action="store_true",
        help="Force Q-model training (overrides config.conditional.use_q_model)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )

    args = parser.parse_args()
    main(args)
