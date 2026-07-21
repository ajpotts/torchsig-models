""" Plot a confusion matrix image from a stored csv file.

Usage:
    python3 conf_matrix_plotting.py <csv> <image> [--class_list CLASS_LIST] 

Raises:
    ValueError: Invalid class_list provided.
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from torchsig.signals.signal_lists import TorchSigSignalLists


def main():
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    parser = argparse.ArgumentParser()

    # Make the root directories for the dataset
    parser.add_argument("csv", type=str, help="Path to confusion matrix csv file.")
    parser.add_argument("image", type=str, help="Path to saved image.")
    parser.add_argument("--class_list", type=str, default='all', help="Signal list type.")
    args = parser.parse_args()
    conf_mat_path = args.csv
    image_save_path = args.image

    list_type = args.class_list
    class_list = None
    if list_type == "all":
        class_list = TorchSigSignalLists.all_signals
    elif list_type == "family":
        class_list = TorchSigSignalLists.family_list
    elif list_type == "fsk":
        class_list = TorchSigSignalLists.fsk_signals
    elif list_type == "ofdm":
        class_list = TorchSigSignalLists.ofdm_signals
    elif list_type == "constellation":
        class_list = TorchSigSignalLists.constellation_signals
    elif list_type == "am":
        class_list = TorchSigSignalLists.am_signals
    elif list_type == "fm":
        class_list = TorchSigSignalLists.fm_signals
    elif list_type == "lfm":
        class_list = TorchSigSignalLists.lfm_signals
    elif list_type == "chirp":
        class_list = TorchSigSignalLists.chirpss_signals
    elif list_type == "tone":
        class_list = TorchSigSignalLists.tone_signals
    else:
        raise ValueError("Unrecognized list name")

    data = np.genfromtxt(conf_mat_path, delimiter=',', dtype = int)

    fig, ax = plt.subplots(figsize = (20,20))
    im = plt.imshow(data, cmap = 'viridis')

    ax.set_xlabel('Ground Truth Class')
    ax.set_xticks(range(len(class_list)))
    ax.set_xticklabels(class_list, rotation = 90)

    ax.set_ylabel('Predicted Class')
    ax.set_yticks(range(len(class_list)))
    ax.set_yticklabels(class_list)

    plt.show()

    plt.savefig(image_save_path)

if __name__ == "__main__":
    main()



