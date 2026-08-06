"""Test narrowband pipeline and models."""

from pathlib import Path
import pytest


from torchsig_models.models.iq_models.xcit.xcit1d_train import xcit1d_trainer


SEED = 1234567890
DATASET_LENGTH = 16

CONFIG_FILE = Path(__file__).with_name("test_narrowband_config.yaml").absolute()


@pytest.mark.slow_no_gpu
@pytest.mark.parametrize(
    "num_epochs, config_file",
    [(2, CONFIG_FILE)],
)
def test_xcit1d_trainer_creates_expected_output_structure(
    num_epochs,
    config_file,
    narrowband_data_dir,
    tmp_path,
):
    """Test that xcit1d_trainer creates all expected directories and output files."""
    base_path = tmp_path / "xcit1d"
    pt_dir = base_path / "pt"
    metrics_dir = base_path / "metrics"

    # Run the training
    xcit1d_trainer(
        root=str(narrowband_data_dir),
        config_file=config_file,
        pt_dir=str(pt_dir),
        metrics_dir=str(metrics_dir),
        num_epochs=num_epochs,
        batch_size=4,
    )

    # Define expected directory structure relative to base_path
    expected_dirs = [
        "pt",
        "metrics",
        "metrics/train",
        "metrics/val",
        "metrics/train/conf_mats",
        "metrics/val/conf_mats",
    ]

    # Define expected files relative to base_path
    expected_files = [
        "metrics/train/accuracy.png",
        "metrics/train/f1 score.png",
        "metrics/train/loss.png",
        "metrics/train/metrics_table.csv",
        "metrics/train/precision.png",
        "metrics/train/recall.png",
        "metrics/val/accuracy.png",
        "metrics/val/f1 score.png",
        "metrics/val/loss.png",
        "metrics/val/metrics_table.csv",
        "metrics/val/precision.png",
        "metrics/val/recall.png",
    ]

    # Verify all directories exist
    for dir_path in expected_dirs:
        assert (base_path / dir_path).is_dir(), (
            f"Expected directory {dir_path} does not exist"
        )

    # Verify all files exist
    for file_path in expected_files:
        assert (base_path / file_path).is_file(), (
            f"Expected file {file_path} does not exist"
        )
