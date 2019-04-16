# Test script for Viterbi decoding
#
# Ning Ma (n.ma@sheffield.ac.uk)
# 20/02/2018

# =================================================================
# GENERAL IMPORTS
# =================================================================
import numpy as np
import scipy.io.wavfile as wav
import os, pickle
import speechtech.frontend as feat


# =================================================================
# GLOBAL VARIABLES
# =================================================================
DIGITS = ['z', 'o', '1', '2', '3', '4', '5', '6', '7', '8', '9']
NUM_DIGITS = len(DIGITS)
DATA_DIR = 'data'
TRAIN_LIST = '{0}/flists/flist_train.txt'.format(DATA_DIR)
TEST_LIST = '{0}/flists/flist_test.txt'.format(DATA_DIR)

# HMM parameters
NUM_STATES = 10  # number of HMM states
NUM_MIXTURES = 3  # number of Gaussian mixtures per state
FEATURE_TYPE = 'mfcc'  # 'fbank' or 'mfcc'
MODEL_DIR = 'models'
MODEL_FILE = os.path.join(MODEL_DIR, "hmmfile_{0}_{1}states_{2}mix".format(FEATURE_TYPE, NUM_STATES, NUM_MIXTURES))


# =================================================================
# MAIN FUNCTION 
# =================================================================
def main():

    # Load trained HMM set
    if not os.path.exists(MODEL_FILE):
        print('HMM file cannot be found: {0}'.format(MODEL_FILE))
        exit()
    with open(MODEL_FILE, 'rb') as f:
        hmm_set = pickle.load(f)
    print('Loaded HMM file: {0}'.format(MODEL_FILE))

    # Read test set list
    with open(TEST_LIST, 'r') as f:
        flist = f.readlines()
    flist = [x.strip() for x in flist]

    # Choose the first utterance for testing
    fn = flist[0]

    # Obtain the target label
    lab = os.path.splitext(os.path.basename(fn))[0][0]
    print('==== Testing digit "{0}" ({1})\n'.format(lab, fn))

    # Choose the target HMM and a wrong HMM for comparison
    target_hmm_id = DIGITS.index(lab)
    wrong_hmm_id = (target_hmm_id + 1) % NUM_DIGITS

    # Compute features
    fs_hz, signal = wav.read(fn)
    features = feat.compute_features[FEATURE_TYPE](signal, fs_hz)

    # For the target HMM
    log_prob, state_seq = hmm_set[target_hmm_id].viterbi_decoding(features)
    print('---- HMM "{0}": log_prob = {1}\nstate_seq = {2}\n'.format(DIGITS[target_hmm_id], log_prob, state_seq))

    # For the wrong HMM
    log_prob, state_seq = hmm_set[wrong_hmm_id].viterbi_decoding(features)
    print('---- HMM "{0}": log_prob = {1}\nstate_seq = {2}\n'.format(DIGITS[wrong_hmm_id], log_prob, state_seq))

# =================================================================
if __name__ == '__main__':
    main()

