"""Script interface for running XCiT classifier training.

Example usage:
    python3 inference_nb_xcit1d.py --root=<root_dir> --config=<config_file> 
        --checkpoint-path=<chkpt_dir>
"""

import argparse
from pathlib import Path

from torchsig_models.models.iq_models.xcit.xcit1d_inference import xcit1d_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Run XCiT 1D classifier inference.")
    parser.add_argument("--root", required=True, help="Path to TorchSig dataset root.")
    parser.add_argument("--config_file", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Path to saved .ckpt file, e.g. classification_model_final.ckpt.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=32)
    args = parser.parse_args()

    accuracy = xcit1d_inference(
        root=args.root,
        config_file=args.config_file,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Overall accuracy: {accuracy:.6f}")

if __name__ == "__main__":
    main()
