"""
Main script for Conditional Diffusion Generation in Financial Time Series
"""
import argparse
import torch
import os
from dataclasses import asdict

from config import get_default_config
from data import DataProcessor
from models import DiffusionModel, HFunctionTrainer, ConditionalGenerator
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
        winsorize_lower=config.data.winsorize_lower,
        winsorize_upper=config.data.winsorize_upper,
    )

    data_processor.process_all()
    # print(data_processor.df_z['unemp'].describe())
    # import sys; sys.exit()  # stops the script here

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
        os.makedirs("checkpoints", exist_ok=True)
        diffusion_model.save("checkpoints/diffusion_model.pt")
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
        )
        diffusion_model.load("checkpoints/diffusion_model.pt")

    # ==================== H-Function Training ====================
    if not args.skip_hfunction_training:
        print("\n" + "=" * 60)
        print("STEP 3: H-Function Training")
        print("=" * 60)

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
            reward_sharpness=config.hfunction.reward_sharpness
        )

        # Generate training paths
        print("Generating training paths...")
        t_grid, y_grid, Y_T = diffusion_model.sample(
            batch_size=config.hfunction.train_batch_size,
            num_steps=config.diffusion.num_steps,
            stoch=0.2,
            return_path=True,
        )

        # Train
        h_trainer.train(
            t_grid=t_grid,
            y_grid=y_grid,
            Y_T=Y_T,
            n_epochs=config.hfunction.n_epochs,
            batch_size=config.hfunction.train_batch_size,
            learning_rate=config.hfunction.learning_rate,
            weight_decay=config.hfunction.weight_decay,
            scheduler_patience=config.hfunction.scheduler_patience,
            scheduler_factor=config.hfunction.scheduler_factor,
            use_wandb=use_wandb,
        )

        # Save model
        h_trainer.save("checkpoints/hfunction.pt")
    else:
        print("\nSkipping H-function training, loading from checkpoint...")
        h_trainer = HFunctionTrainer(
            asset_dim=config.hfunction.asset_dim,
            time_steps=config.hfunction.time_steps,
            embed_dim=config.hfunction.embed_dim,
            event_asset_idx=config.hfunction.event_asset_idx,
            event_window=config.hfunction.event_window,
            event_threshold=config.hfunction.event_threshold,
            device=config.hfunction.device,
            constraint_mode=config.hfunction.constraint_mode,
            reward_sharpness=config.hfunction.reward_sharpness,
        )
        h_trainer.load("checkpoints/hfunction.pt")

    # ==================== Extract Events ====================
    print("\n" + "=" * 60)
    print("STEP 4: Extract Events (Train & Test)")
    print("=" * 60)

    # Extract events from TRAIN set
    X_train = data_processor.X_train
    asset_sums_train = X_train.sum(dim=2)

    last_window_train = X_train[:, -config.hfunction.event_window :, config.hfunction.event_asset_idx]
    if config.hfunction.event_type == "sum":
        metric_train = last_window_train.sum(dim=1)
        mask_train = metric_train <= config.hfunction.event_threshold
    elif config.hfunction.event_type == "change":
        metric_train = (last_window_train[:, -1] - last_window_train[:, 0]).abs()
        mask_train = metric_train >= config.hfunction.event_threshold
    elif config.hfunction.event_type == "absval":
        metric_train = last_window_train[:, -1].abs()
        mask_train = metric_train >= config.hfunction.event_threshold

    print("Min Change", metric_train.min().item())
    print("Max Change", metric_train.max().item())
    print("Mean Change", metric_train.mean().item())
    print("Threshold:", config.hfunction.event_threshold)

    print("X_train shape", X_train.shape)
    print("Sample unemployment values", X_train[0, :, 0])
    print("Sample unemployment values", X_train[0, 0, :])

    event_asset_sums_train = asset_sums_train[mask_train]
    N_event_train = event_asset_sums_train.shape[0]

    print(f"Train set size: {X_train.shape[0]}")
    print(f"Number of events in train set: {N_event_train}")

    # Extract events from TEST set
    X_test = data_processor.X_test
    asset_sums_test = X_test.sum(dim=2)

    last_window_test = X_test[:,-config.hfunction.event_window :, config.hfunction.event_asset_idx]
    if config.hfunction.event_type == "sum":
        metric_test = last_window_test.sum(dim=1)
        mask_test = metric_test <= config.hfunction.event_threshold
    elif config.hfunction.event_type == "change":
        metric_test = (last_window_test[:, -1] - last_window_test[:, 0]).abs()
        mask_test = metric_test >= config.hfunction.event_threshold
    elif config.hfunction.event_type == "absval":
        metric_test = last_window_test[:, -1].abs()
        mask_test = metric_test >= config.hfunction.event_threshold
    
    print("Test dates:", data_processor.y_dates_test[0], "to", data_processor.y_dates_test[-1])
    print("Test change stats:", metric_test.min().item(), metric_test.max().item())

    event_asset_sums_test = asset_sums_test[mask_test]
    N_event_test = event_asset_sums_test.shape[0]

    print(f"Test set size: {X_test.shape[0]}")
    print(f"Number of events in test set: {N_event_test}")

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
        constraint_mode=config.conditional.constraint_mode,
        beta=config.conditional.beta,
        in_channels=config.diffusion.in_channels,
        sample_size=config.diffusion.sample_size,
    )

    # Optionally train Q-model
    if config.conditional.use_q_model and not args.skip_qmodel_training:
        print("Training Q-model...")
        t_grid, y_grid, _ = diffusion_model.sample(
            batch_size=config.hfunction.train_batch_size,
            num_steps=config.diffusion.num_steps,
            stoch=0.2,
            return_path=True,
        )
        cond_generator.train_q_model(
            t_grid=t_grid,
            y_grid=y_grid,
            in_channels=config.diffusion.in_channels,
            out_channels=config.diffusion.out_channels,
            n_epochs=config.conditional.q_model_epochs,
            learning_rate=config.conditional.q_model_lr,
        )
        cond_generator.save_q_model("checkpoints/q_model.pt")

    # Generate conditional samples for TRAIN set events
    print(f"Generating {N_event_train} conditional samples for in-sample (train) events...")
    generated_samples_train = cond_generator.generate(
        num_samples=N_event_train,
        batch_size=config.conditional.batch_size,
        num_steps=config.conditional.num_steps,
        stoch=config.conditional.stoch,
        eta=config.conditional.eta,
        use_q_model=config.conditional.use_q_model,
    )
    torch.save(generated_samples_train, 'generated_samples_train.pt')

    # Generate conditional samples for TEST set events
    print(f"Generating {N_event_test} conditional samples for out-of-sample (test) events...")
    generated_samples_test = cond_generator.generate(
        num_samples=N_event_test,
        batch_size=config.conditional.batch_size,
        num_steps=config.conditional.num_steps,
        stoch=config.conditional.stoch,
        eta=config.conditional.eta,
        use_q_model=config.conditional.use_q_model,
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
        config = config,
    )

    # ========== In-Sample (Train) Analysis ==========
    print("\n--- In-Sample (Train Set) Analysis ---")

    # Analyze generated samples for train
    print("Analyzing generated samples (in-sample)...")
    gen_mv_train, gen_rp_train, gen_avg_train = portfolio_analyzer.analyze_samples(generated_samples_train)

    # Analyze train set
    print("Analyzing train set...")
    real_mv_train, real_rp_train, real_avg_train = portfolio_analyzer.analyze_test_set(X_train, mask_train)

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
    real_mv_test, real_rp_test, real_avg_test = portfolio_analyzer.analyze_test_set(X_test, mask_test)

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
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )

    args = parser.parse_args()
    main(args)
