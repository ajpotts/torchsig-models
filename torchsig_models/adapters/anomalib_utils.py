"""Utilities for working with the Anomalib library."""

from pathlib import Path
from typing import Callable
import torch
import numpy as np

import pandas as pd
from torchvision.tv_tensors import Mask
from scipy.ndimage import zoom

from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.dataclasses import ImageItem
from anomalib.data.utils import LabelName
from anomalib.data.utils.split import ValSplitMode, TestSplitMode

from torchsig.datasets.datasets import StaticTorchSigDataset
from torchsig.signals.signal_types import Signal
from torchsig.transforms.metadata_transforms import MetadataTransform
from torchsig.transforms.transforms import SignalTransform
from torchsig.utils.dsp import TorchSigRealDataType


class TorchSigAnomalibDataset(AnomalibDataset):
    """AnomalibDataset adapter for StaticTorchSigDataset.

    Args:
        root (str): Root directory of the StaticTorchSigDataset.
        task (str, optional): Type of anomaly task (currently classification supported). 
            Default: "classification".
        split (str, optional): Type of data split configuration. Default: "train".
        index_bounds (list, optional): Valid current split dataset range [min, max] indices
            within full StaticTorchSigDataset dataset. Default: [0, 0].
        use_anomaly_labels (bool, optional): Maintain or remove provided anomaly labels. 
            Default: False.
        augmentations (Callable, optional): Anomalib API transforms to apply to data elements. 
            Default: None.
        torchsig_transforms (list, optional): TorchSig API transforms to apply to data 
            elements. Default: [].
        target_labels (list, optional): Labels for the StaticTorchSigDataset.
    """    
    
    def __init__(
            self, 
            root: str,
            task: str = "classification",
            split: str = "train",
            index_bounds: list = [0, 0],
            use_anomaly_labels: bool = False,
            augmentations: Callable | None = None,
            torchsig_transforms: list = [],
            target_labels: list = ["class_name", "start", "stop", "lower_freq", "upper_freq", "anomaly"],
    ) -> None:
        super().__init__(augmentations = augmentations)
        
        self.root = Path(root)
        self.split = split
        self.index_bounds = index_bounds
        self.use_anomaly_labels = use_anomaly_labels
        self.augmentations = augmentations
        self.torchsig_transforms = torchsig_transforms
        self.target_labels = target_labels

        # initialize underlying static TorchSigDataset
        self.dataset = StaticTorchSigDataset(
            root = self.root,
            transforms = self.torchsig_transforms,
            target_labels = self.target_labels,
        )
                
        # initialize samples Pandas DataFrame with placeholder 
        num_samples = len(self.dataset)
        self.placeholder_path = self.root
        self.samples = pd.DataFrame({
            "image_path":  [str(self.root)] * num_samples,      # placeholder path (exists on disk)
            "split":       [split] * num_samples,               # split information
            "label_index": [LabelName.UNKNOWN] * num_samples,   # determine labels at runtime
            "mask_path":   ["" for _ in range(num_samples)]
        })
        self.samples.attrs["task"] = task

        try:
            self.anomaly_label_index = self.target_labels.index("anomaly")
        except Exception as e: 
            self.anomaly_label_index = None # no anomaly labels provided

    def __len__(self):
        """Returns:
            int: adjusted length of dataset by provided bounds for current split
        """
        return self.index_bounds[1] - self.index_bounds[0] + 1

    def __getitem__(self, index: int) -> ImageItem:
        """Return sample in appropriate Anomalib Image format with dynamic label assignment."""
        data_element = self.dataset[self.index_bounds[0] + index] # map index to current split

        # Process TorchSig data element
        anomalies = []
        if isinstance(data_element, tuple):
            spec_np, labels = data_element  # data, list of target_labels list (each signal)

            # image-level classification task (segmentation not supported)
            if self.anomaly_label_index is not None:
                if isinstance(labels[0], list): # multiple Signals present in element (list of lists)
                    anomaly_labels: list = labels[self.anomaly_label_index] # list of np.bool_
                    for signal_label in anomaly_labels:
                        if bool(signal_label):
                            anomalies = [True]  # classify image as anomaly if 1+ anomalies present
                else: # one Signal present (single list)
                    if bool(labels[self.anomaly_label_index]):
                        anomalies = [True]  # any anomalies present in image
                    
        else:
            spec_np = data_element  # ie, no labels provided, only data

        # Convert spectrogram image to 3-channel tensor (CxHxW)
        spec = torch.from_numpy(spec_np).float()
        if spec.ndim == 2:                      # [H, W] -> [1, H, W]
            spec = spec.unsqueeze(0)

        if spec.shape[0] == 1:                  # [1, H, W] -> [3, H, W]
            spec = spec.repeat(3, 1, 1)  

        # Apply Anomalib-style augmentations here (image-only for classification)
        if self.augmentations:
            spec = self.augmentations(spec)

        H, W = spec.shape[-2], spec.shape[-1]

        # set labels and masks
        if self.samples.attrs["task"] == "segmentation": # NOT SUPPORTED
            if anomalies:
                #gt_mask = boxes_to_gt_mask((H, W), anomalies) # compute masks
                gt_mask = Mask(torch.ones((H, W), dtype=torch.uint8))
                gt_label = torch.tensor(bool(gt_mask.any()), dtype=torch.bool)
            else:
                gt_mask = Mask(torch.zeros((H, W), dtype=torch.uint8))
                gt_label = torch.tensor(False, dtype=torch.bool)
        else:
            # classification task (for masks, treat pixels in image uniformly)
            if self.use_anomaly_labels and len(anomalies) > 0:
                gt_mask = Mask(torch.ones((H, W), dtype=torch.uint8)) # all pixels anomalies
                label_val = LabelName.ABNORMAL  # anomaly present and labeled
            else: # either no anomaly, or ignoring anomalies
                gt_mask = Mask(torch.zeros((H, W), dtype=torch.uint8)) # all pixels normal
                label_val = LabelName.NORMAL    
            gt_label = torch.tensor(bool(label_val), dtype=torch.bool)

        return ImageItem( 
            image = spec,
            gt_label = gt_label,
            gt_mask = gt_mask,
            image_path = None, # NOT SUPPORTED
            mask_path = "",
        )


class TorchSigAnomalibDataModule(AnomalibDataModule):
    """AnomalibDataModule adapter using a TorchSigAnomalibDataset.

    Args:
        root (str): Root directory of the underlying TorchSigAnomalibDataset.
        train_batch_size (int, optional): Batch size for training dataloader. Default: 32.
        eval_batch_size (int, optional): Batch size for validation and test dataloaders. Default: 32.
        num_workers (int, optional): Number of workers for dataloaders. Default: 8.
        seed (int | None, optional): Random seed for reproducibility. Default: None.
        splits (tuple[float], optional): Proportions for train/val/test splits. Default: (0.70, 0.25, 0.05).
        size (int, optional): Use subset of data of this size if not None. Default: None.
        use_anomaly_labels (bool, optional): Maintain or remove provided anomaly labels. Default: False.
        augmentations (Callable, optional): Anomalib API transforms to apply to data elements. Default: None.
        transform (Callable, optional): Anomalib API transforms to apply to data elements after augmentations. Default: None.
        torchsig_transforms (list, optional): List of TorchSigDataset transforms to apply to data elements before Anomalib augmentations. Default: [].
        target_labels (list, optional): Labels for the StaticTorchSigDataset. Default: ["class_name", "start", "stop", "lower_freq", "upper_freq", "anomaly"].
    """
    def __init__(
        self,
        root: str,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        seed: int | None = None,
        splits: tuple[float] = (0.70, 0.25, 0.05),
        size: int = None,  # use data subset if not None
        use_anomaly_labels: bool = False,
        augmentations: Callable | None = None,
        transform: Callable | None = None,
        torchsig_transforms: list = [],
        target_labels: list = ["class_name", "start", "stop", "lower_freq", "upper_freq", "anomaly"],
    ):
        super().__init__(
            train_batch_size = train_batch_size,
            eval_batch_size = eval_batch_size,
            num_workers = num_workers,
            train_augmentations = augmentations,
            val_augmentations = augmentations,
            test_augmentations = augmentations,
            augmentations = augmentations,
            # already have explicit train/val/test splits, so no automatic splitting needed
            val_split_mode = ValSplitMode.NONE,
            val_split_ratio = 0.0,
            test_split_mode = TestSplitMode.NONE,
            test_split_ratio = 0.0,
            seed = seed,
        )

        self.root = Path(root)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.splits = splits
        self.size = size
        self.use_anomaly_labels = use_anomaly_labels
        self.augmentations = augmentations
        self.transform = transform
        self.torchsig_transforms = torchsig_transforms
        self.target_labels = target_labels
        
        # determine static dataset characteristics
        static_dataset = StaticTorchSigDataset(
            root=self.root,
            transforms=self.torchsig_transforms,
            target_labels=self.target_labels,
        )
        self.full_dataset_size = len(static_dataset)

        # configure [train, val, test] dataset split - deterministic, sequential data subsets        
        self.datasets = [None, None, None]
        self.dataset_sizes = [None, None, None]
        self.index_bounds = [[None, None], [None, None], [None, None]] # [min, max] indices
        if self.size is not None:
            total_size = self.size
        else:
            total_size = self.full_dataset_size

        # dataset sizes for train/val/test
        self.dataset_sizes[0] = int(self.splits[0] * total_size)            
        self.dataset_sizes[1] = int(self.splits[1] * total_size)
        self.dataset_sizes[2] = total_size - self.dataset_sizes[0] - self.dataset_sizes[1]
        
        # dataset index [min, max] for train/val/test
        self.index_bounds[0] = [0, self.dataset_sizes[0] - 1]
        self.index_bounds[1] = [self.index_bounds[0][1] + 1, self.index_bounds[0][1] + self.dataset_sizes[1]]
        self.index_bounds[2] = [self.index_bounds[1][1] + 1, self.index_bounds[1][1] + self.dataset_sizes[2]]


    def _setup(self, stage=None):
        """Configure train (0), validation (1), and test (2) datasets."""
        for i, split_name in enumerate(["train", "val", "test"]):
            self.datasets[i] = TorchSigAnomalibDataset(
                root = self.root,
                task = "classification",
                split = split_name,
                index_bounds = self.index_bounds[i],
                use_anomaly_labels = self.use_anomaly_labels,
                augmentations = self.augmentations,
                torchsig_transforms = self.torchsig_transforms,
                target_labels = self.target_labels,
            )

        self.train_data = self.datasets[0]
        self.val_data = self.datasets[1]
        self.test_data = self.datasets[2]


class AnomalyLabel(MetadataTransform):
    """Transform that adds an 'anomaly' field to each component Signal with value
    based on its class_name. Primarily for use in an Anomalib pipeline.
    """
    def __init__(self, anomaly_class_names: list = [], **kwargs):
        """Initialize AnomalyLabel transform.

        Args:
            anomaly_class_names (list, optional): List of class_names to assign 
                'anomaly' to True, otherwise assign False. Default: [].
        """
        super().__init__(required_metadata=["class_name"], **kwargs) #
        self.anomaly_class_names = list(anomaly_class_names)
        self.targets_metadata = ["anomaly"]

    def __apply__(self, signal: Signal) -> Signal:
        """Apply the AnomalyLabel transform to a Signal by adding an 'anomaly' field,
        using the anomaly_class_names list to set anomaly True or False
        
        Args:
            signal (Signal): Signal to add the 'anomaly' label.

        Returns:
            Signal: Signal with 'anomaly' label added and set.
        """
        if signal["class_name"] in self.anomaly_class_names:
            signal["anomaly"] = True
        else:
            signal["anomaly"] = False
        return signal


class SpectrogramZoom(SignalTransform):
    """Employ the scipy Zoom function to resize spectrogram data to a new shape. 
    Warning: this is a computationally expensive operation. It also is input size
    agnostic, meaning it can resize spectrograms of unconstrained input shape. 
    This may break the link between the output resized spectrogram data and the 
    signal metadata, so use with caution and verify results. 
    Primarily for use in the Anomalib pipeline when a specific data shape is required.
    """
    def __init__(
        self,
        output_shape: tuple[int, int] = (256, 256),
        **kwargs
    ):
        """Initialize SpectrogramZoom transform.
        
        Args:
            output_shape (tuple[int, int]): desired output dimensions of 
                spectrogram (height, width). Default (256, 256).
        """
        super().__init__(
            required_metadata=[],
            data_dtype=TorchSigRealDataType,
            **kwargs
        )
        self.output_shape = output_shape

    def __apply__(self, signal: Signal) -> Signal:
        """Apply scipy zoom to Signal spectrogram data.

        Args:
            signal (Signal): input signal containing spectrogram data to be resized.
        Returns:
            Signal: output signal with resized spectrogram data.
        """
        in_shape = signal.data.shape

        # zoom expects (height, width) order for 2D data, so calculate zoom factors accordingly
        signal.data = zoom(
            signal.data,
            zoom=(self.output_shape[0]/in_shape[0], self.output_shape[1]/in_shape[1]),
            order=2,
            mode="nearest",
            prefilter=True,
        )

        # Signal metadata may no longer be accurate after resizing, 
        # so consider whether to update or remove metadata fields 
        return signal


class SpectrogramRescale(SignalTransform):
    """Rescales a real-valued spectrogram in dB to [0.0, 1.0] range, with pre-clipping.

    Intended for taking the output of `Spectrogram` and similar data formats, 
    usually preparing data for vision/anomaly models (e.g., anomalib).

    Two modes:
      1) Fixed range (recommended): clip_db_range=(lo, hi) then scale linearly:
            y = clip(x, lo, hi)
            y = (y - lo) / (hi - lo)
      2) Per-sample min-max: clip_db_range=None:
            y = (x - min(x)) / (max(x) - min(x))

    Notes:
        - Enforces TorchSigRealDataType on output via SignalTransform's dtype handling.
        - Does not update Signal metadata.
    """

    def __init__(
        self,
        clip_db_range: tuple[float, float] | None,
        eps: float = 1e-8,
        **kwargs
    ):
        """Initialize SpectrogramRescale transform.
        
        Args:
            clip_db_range (Tuple[float,float] | None): If provided, values are clipped
                to (lo, hi) before scaling. If None (default), per-sample min-max is used.
            eps (float): Small constant to avoid division-by-zero. Default: 1e-8.
        """
        super().__init__(
            required_metadata=[],
            data_dtype=TorchSigRealDataType,
            **kwargs
        )
        self.clip_db_range = clip_db_range
        self.eps = eps

    def __apply__(self, signal: Signal) -> Signal:
        """Apply rescaling to spectrogram data in the Signal.
        
        Args:
            signal (Signal): Input Signal containing spectrogram [log10 dB] data to be rescaled.
        
        Returns:
            Signal: Output Signal with rescaled spectrogram data [0., 1.].
        """
        x = (signal.data).astype(np.float32, copy=False)

        if self.clip_db_range is not None: # universal scaling
            lo, hi = self.clip_db_range
            x = np.clip(x, lo, hi) # limit db range
            x = (x - lo) / (hi - lo + self.eps)  # [0., 1.] 
        else: # note per-image scaling (not across all dataset)
            x = x - float(np.max(x)) # full scale set at 0.0
            x_min = float(np.min(x))
            x = (x - x_min) / (0.0 - x_min + self.eps) # [0., 1.]

        x = np.clip(x, 0.0, 1.0).astype(np.float32)       
        signal.data = x

        # Note: metadata may no longer be accurate after rescaling, so consider whether
        # to update or remove metadata fields
        return signal