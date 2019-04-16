# This script performs spoken digit recognition using Gaussian Mixture Models
#
# Ning Ma (n.ma@sheffield.ac.uk)
# 20/02/2018

# =================================================================
# GENERAL IMPORTS
# =================================================================
import numpy as np
import scipy.io.wavfile as wav
import os, pickle
from sklearn.mixture import GaussianMixture
import speechtech.frontend as feat

# =================================================================
# GLOBAL VARIABLES
# =================================================================
DIGITS = ['z', 'o', '1', '2', '3', '4', '5', '6', '7', '8', '9']
NUM_DIGITS = len(DIGITS)
DATA_DIR = 'data'
TRAIN_LIST = '{0}/flists/flist_train.txt'.format(DATA_DIR)
TEST_LIST = '{0}/flists/flist_test.txt'.format(DATA_DIR)
NUM_MIXTURES = 7  # number of Gaussian mixtures per GMM


# =================================================================
# TRAINING ROUTINE
# =================================================================
def train_GMMs(file_list, feature_type='fbank'):
    """Performs GMM training giveing a list of utterances

    Args:
        file_list: A list of utterances for training GMMs
        feature_type: Acoustic feature type ('fbank' or 'mfcc', default 'fbank')

    Returns
        A list of trained GMMs
    """
    print('Training feature type: {}'.format(feature_type))
    print('Number of utterances: {}'.format(len(file_list)))

    gmm_set = [GaussianMixture(n_components=NUM_MIXTURES, covariance_type='diag',
                               init_params='kmeans', max_iter=100) for gmm_id in range(NUM_DIGITS)]

    for gmm_id in range(NUM_DIGITS):
        gmm_label = DIGITS[gmm_id]
        all_features = np.array([])

        # Accumulate training data for each model
        for fn in file_list:
            # Obtain the label for the data
            label = os.path.splitext(os.path.basename(fn))[0][0]
            if gmm_label == label:
                # Compute features
                fs_hz, signal = wav.read(fn)
                features = feat.compute_features[feature_type](signal, fs_hz)
                all_features = np.concatenate((all_features, features)) if all_features.size else features

        # Train GMM
        print('Training digit model {}: features {}'.format(gmm_label, all_features.shape))
        gmm_set[gmm_id].fit(all_features)

    return gmm_set


# =================================================================
# EVALUATION ROUTINE
# =================================================================
def plot_confusions(target_labels, rec_labels):
    """Plot the confusion matrix

    Args:
        target_labels: A list of target labels
        rec_labels: A list of recognition output labels
    """
    mat = confusion_matrix(target_labels, rec_labels)
    sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=DIGITS, yticklabels=DIGITS, cmap="YlGnBu")
    plt.xlabel('predicted label')
    plt.ylabel('true label')


def eval_GMMs(gmm_set, file_list, feature_type='fbank'):
    """Performs GMM evaluation giveing a list of GMMs and utterances

    Args:
        file_list: A list of utterances for training GMMs
        feature_type: Acoustic feature type ('fbank' or 'mfcc', default 'fbank')

    Returns
        wer: Word error rate
        target_labels: Target reference labels
        rec_labels: Recognition output labels
    """
    print('Evaluation feature type: {}'.format(feature_type))
    print('Number of utterances = {}'.format(len(file_list)))

    num_files = len(file_list)
    target_labels = []
    rec_labels = []
    for fn in file_list:
        # Obtain the label for the data
        target_labels = np.append(target_labels, os.path.splitext(os.path.basename(fn))[0][0])

        # Compute features
        fs_hz, signal = wav.read(fn)
        features = feat.compute_features[feature_type](signal, fs_hz)

        # Maximum likelihood classification based on average likelihood scores
        # across each utterance
        scores = [gmm_set[gmm_id].score(features) for gmm_id in range(NUM_DIGITS)]
        rec_labels = np.append(rec_labels, DIGITS[np.argmax(scores)])

    # Compute word error rate (for isolated digit recognition there are only substitution errors)
    wer = np.mean(target_labels != rec_labels)
    print('Word error rate is {0:0.2f}%\n'.format(wer * 100))

    return wer, target_labels, rec_labels


# =================================================================
# MAIN FUNCTION
# =================================================================
def main():
    # Read training set list
    with open(TRAIN_LIST) as f:
        flist_train = f.readlines()
    flist_train = [x.strip() for x in flist_train]

    # Read test set list
    with open(TEST_LIST) as f:
        flist_test = f.readlines()
    flist_test = [x.strip() for x in flist_test]

    # -------------------------
    # FBANK
    # -------------------------

    # Train GMMs
    gmm_set = train_GMMs(flist_train, feature_type='fbank')

    # Evaluation
    wer, target_labels, rec_labels = eval_GMMs(gmm_set, flist_test, feature_type='fbank')

    # -------------------------
    # MFCC
    # -------------------------

    # Train GMMs
    gmm_set = train_GMMs(flist_train, feature_type='mfcc')

    # Evaluation
    wer, target_labels, rec_labels = eval_GMMs(gmm_set, flist_test, feature_type='mfcc')


# =================================================================
if __name__ == '__main__':
    main()
