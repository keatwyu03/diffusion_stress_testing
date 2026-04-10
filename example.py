"""
Example script showing how to use the cdg_finance package
"""
from config import get_default_config, Config, DataConfig, DiffusionConfig
from data import DataProcessor
from models import DiffusionModel, HFunctionTrainer, ConditionalGenerator
from utils import PortfolioAnalyzer, set_seed


def example_data_processing():
    """Example: Data processing only"""
    print("=== Data Processing Example ===\n")

    # Create data processor
    data_processor = DataProcessor(
        csv_path="Stocks_logret.csv",
        tickers=["AAPL", "AMZN", "JPM", "TSLA"],
        seq_len=64,
        test_days=700,
    )

    # Process all data
    data_processor.process_all()

    print(f"\nTraining set: {data_processor.X_train.shape}")
    print(f"Test set: {data_processor.X_test.shape}")

    return data_processor


def example_diffusion_training(data_processor):
    """Example: Train diffusion model"""
    print("\n=== Diffusion Model Training Example ===\n")

    # Create model
    model = DiffusionModel(
        in_channels=4,
        out_channels=4,
        sample_size=64,
        device="mps",
    )

    # Get training data
    train_data = data_processor.get_diffusion_data()
    print(f"Training data shape: {train_data.shape}")

    # Train (using fewer epochs for example)
    model.train(
        train_data=train_data,
        batch_size=256,
        n_epochs=100,  # Reduced for example
    )

    # Save
    model.save("checkpoints/example_diffusion.pt")

    return model


def example_sampling(diffusion_model):
    """Example: Sample from diffusion model"""
    print("\n=== Sampling Example ===\n")

    # Generate samples
    samples = diffusion_model.sample(
        batch_size=64,
        num_steps=200,
        stoch=1.0,
    )

    print(f"Generated samples shape: {samples.shape}")

    return samples


def example_custom_config():
    """Example: Using custom configuration"""
    print("\n=== Custom Configuration Example ===\n")

    # Create custom config
    config = Config(
        seed=42,
        data=DataConfig(
            csv_path="Stocks_logret.csv",
            tickers=["AAPL", "MSFT", "GOOGL"],
            test_days=500,
        ),
        diffusion=DiffusionConfig(
            n_epochs=300,
            batch_size=128,
        ),
    )

    print(f"Tickers: {config.data.tickers}")
    print(f"Test days: {config.data.test_days}")
    print(f"Diffusion epochs: {config.diffusion.n_epochs}")

    return config


def example_portfolio_analysis(data_processor, generated_samples):
    """Example: Portfolio analysis"""
    print("\n=== Portfolio Analysis Example ===\n")

    # Create analyzer
    analyzer = PortfolioAnalyzer(
        data_processor=data_processor,
        window_for_cov=54,
        last_days_sum=5,
    )

    # Analyze generated samples
    gen_mv, gen_rp, gen_avg = analyzer.analyze_samples(generated_samples)

    print(f"Generated samples analyzed: {len(gen_mv)}")
    print(f"Min-Variance mean: {sum(gen_mv) / len(gen_mv):.4f}")
    print(f"Risk-Parity mean: {sum(gen_rp) / len(gen_rp):.4f}")
    print(f"Equal-Weight mean: {sum(gen_avg) / len(gen_avg):.4f}")


if __name__ == "__main__":
    # Set seed
    set_seed(2025)

    # Run examples
    print("\n" + "=" * 60)
    print("CDG Finance Package - Usage Examples")
    print("=" * 60)

    # Example 1: Data processing
    data_proc = example_data_processing()

    # Example 2: Custom configuration
    custom_config = example_custom_config()

    # Example 3: Diffusion training (commented out by default)
    # diff_model = example_diffusion_training(data_proc)
    # samples = example_sampling(diff_model)
    # example_portfolio_analysis(data_proc, samples)

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
