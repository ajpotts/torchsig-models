"""Example script for generating a TorchSig spectrogram dataset with labeled normal and anomaly elements, where
the anomalies are spurious signals pseudorandomly injected by the Spurs SignalTransform.

Example usage:
    python3 generate_anomaly_dataset.py --root=<root> --dataset_length=256 --fft_size 256 --plot_samples
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from torchsig.datasets.datasets import TorchSigIterableDataset, StaticTorchSigDataset
from torchsig.signals.signal_types import Signal
from torchsig.transforms.transforms import SignalTransform, Spectrogram, Spurs
from torchsig.utils.data_loading import WorkerSeedingDataLoader
from torchsig.utils.defaults import TorchSigDefaults
from torchsig.utils.writer import DatasetCreator, identity_collate_fn

from torchsig_models.adapters.anomalib_utils import SpectrogramRescale


class RandomSpursAnomalies(SignalTransform):
    """SignalTransform that applies Spurs to the full sample data
    with probability p. Label spur-present samples as normal 
    (anomaly=False), and spur-absent samples as anomalous (anomaly=True). 
    """
    def __init__(
        self,
        probability: float = 0.5,
        num_spurs: tuple[int, int] = (3, 6),
        relative_power_db: tuple[float, float] = (6.0, 15.0),
        **kwargs,
    ):
        """Initialize the RandomSpursAnomalies transform.

        Args:
            probablility (float, optional): probability of inserting spurs.
                Default 0.5.
            num_spurs (tuple[int, int], optional): min, max number of spurs 
                to insert. Default (3, 6).
            relative_power_db (tuple[float, float], optional): min, max power
                of spurs relative to noise floor. Default (6.0, 15.0).
            **kwargs: Additional keyword arguments passed to the parent class.
        """ 
        super().__init__(**kwargs)
        self.probability = probability
        self.spurs = Spurs(
            num_spurs=num_spurs,
            relative_power_db=relative_power_db,
        )
        self.spurs.add_parent(self)  # inherit TorchSig seeding

    def __call__(self, signal: Signal) -> Signal:
        spurs_present = self.random_generator.random() < self.probability

        if spurs_present:
            signal = self.spurs(signal)  # insert spurs into Signal data
        
        # metadata: signal
        signal["anomaly"] = spurs_present # add anomaly label

        # metadata: components_signal
        # ensure all components have same label as Signal, 
        # whether or not spurs fall within their occupancy
        for component in signal.component_signals:
            component["anomaly"] = spurs_present

        return signal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=str, help="StaticTorchSigDataset root.")
    parser.add_argument("--dataset_length", default=2048, type=int, help="Number of items to generate for dataset. Default 2048.")
    parser.add_argument("--fft_size", default=256, type=int, help="FFT size for each spectrogram data item. Default 256.")
    parser.add_argument("--anomaly_prob", default=0.5, type=float, help="Probability of inserting anomalies into a data item. Default 0.5.")
    parser.add_argument("--plot_samples", action="store_true", help="Whether to plot several sample spectrograms after dataset generation.")
    args = parser.parse_args()

    root = args.root
    dataset_length = args.dataset_length
    fft_size = args.fft_size
    anomaly_prob = args.anomaly_prob
    plot_samples = args.plot_samples

    # metadata: wideband dataset 
    # three signal classes, one or two randomly chosen per item
    # label all "tone" signals as anomalies
    signal_generators = ["bpsk", "qpsk"] 
    seed = 12345
    target_labels = ["class_name", "start", "stop", "lower_freq", "upper_freq", "anomaly"]

    dataset_metadata = TorchSigDefaults().default_dataset_metadata
    modified_metadata = {
        "num_iq_samples_dataset": fft_size * fft_size, # square spectrogram
        "num_signals_min": 1,
        "num_signals_max": 2,        
        "fft_size": fft_size,
        "fft_stride": fft_size, # non-overlapping ffts
        "sample_rate": 10_000_000, # typical wideband sample rate
        "noise_power_db": 0.0,
        "snr_db_min": 10.0, # ensure signals are visible above noise floor in spectrogram
        "snr_db_max": 20.0,
        "cochannel_overlap_probability": 0., # no signal overlap
        "signal_duration_in_samples_min": int(fft_size * fft_size * (1 / 20)), # 5.0% of image
        "signal_duration_in_samples_max": int(fft_size * fft_size * (1 / 10)), # 10.0% of image
        "bandwidth_min": 625_000, # 1/16 sample rate
        "bandwidth_max": 1_250_000, # 1/8 sample rate
        "signal_center_freq_min": -2_500_000,
        "signal_center_freq_max": 2_499_999,
        "frequency_min": -2_500_000,
        "frequency_max": 2_499_999, 
    }
    dataset_metadata.update(modified_metadata)
    print(f"Anomaly dataset with {dataset_length} items to be written to: {root}\n")

    # create dataset with normal and anomaly items
    dataset = TorchSigIterableDataset(
        signal_generators = signal_generators,
        metadata = dataset_metadata,
        transforms = [
            RandomSpursAnomalies(
                probability=anomaly_prob,        # probability of any anomalies in item
                num_spurs=(3, 6),                # multiple spurs, when present
                relative_power_db=(10.0, 20.0),  # above the noise floor
            ),
            Spectrogram(fft_size=fft_size), # assume default (freq_bins, time_bins) output format
            SpectrogramRescale(clip_db_range=(-20, 40)) # limit and scale range
        ],
        component_transforms = [], # no component-level transforms
        target_labels = target_labels,
    )
    dataset.seed(seed)
    dataloader = WorkerSeedingDataLoader(
        dataset,
        seed=seed, 
        batch_size=64, 
        num_workers=8, 
        collate_fn=identity_collate_fn
    )
    dataset_creator = DatasetCreator(
        dataloader = dataloader,
        dataset_length = dataset_length,
        root = root,
        overwrite = True,
        multithreading = False
    )
    dataset_creator.create() # write dataset to disk

    if plot_samples:
        os.makedirs(root + "/plots", exist_ok=True)

        anomaly_dataset = StaticTorchSigDataset(
            root = root,
            target_labels = target_labels,
        )

        # plot three sample spectrograms
        for i in range(0, 3, 1):
            spectrogram_data, metadata = anomaly_dataset[i]  
            anomaly_labels_np: list = metadata[target_labels.index("anomaly")]
            anomaly_labels_bool: list = [bool(x) for x in anomaly_labels_np] # standard bool
            labels: list = any(anomaly_labels_bool) # if any anomalies, title as anomaly

            resolution = dataset_metadata["fft_size"]
            fs = dataset_metadata["sample_rate"]
            fs_bound_mhz = np.round(fs/2/1E6)
            time_bound_us = np.round(resolution*(1/fs)*1E6)
            axes_extent = [0,time_bound_us,-fs_bound_mhz,fs_bound_mhz]
            
            plt.figure(figsize=(10, 6))
            plt.imshow(spectrogram_data, extent=axes_extent, aspect='auto', origin='lower')
            #plt.title(f"Sample Spectrogram (Anomaly: {labels})")
            plt.xlabel(r'Time ($\mu$s)', fontsize=20)
            plt.ylabel("Frequency (MHz)", fontsize=20)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            #plt.colorbar(label="Intensity")
            plt.show()
            plt.savefig(f"{root}/plots/sample_{i}_anomaly_{labels}.png")
            plt.close()

  
if __name__ == "__main__":
    main()
    
