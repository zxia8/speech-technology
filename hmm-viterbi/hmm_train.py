# This script performs Viterbi-style training for Hidden Markov Models
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
from hmm import HMM

# =================================================================
# GLOBAL VARIABLES
# =================================================================
DIGITS = ['z', 'o', '1', '2', '3', '4', '5', '6', '7', '8', '9']
NUM_DIGITS = len(DIGITS)
DATA_DIR = 'data'
TRAIN_LIST = '{0}/flists/flist_train.txt'.format(DATA_DIR)
TEST_LIST = '{0}/flists/flist_test.txt'.format(DATA_DIR)

# HMM TRAINING
NUM_STATES = 10  # number of HMM states
NUM_MIXTURES = 3  # number of Gaussian mixtures per state
NUM_ITERATIONS = 20  # number of training iterations
FEATURE_TYPE = 'mfcc'  # 'fbank' or 'mfcc'
MODEL_DIR = 'models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
MODEL_FILE = os.path.join(MODEL_DIR, "hmmfile_{0}_{1}states_{2}mix".format(FEATURE_TYPE, NUM_STATES, NUM_MIXTURES))


# =================================================================
# TRAINING ROUTINE 
# =================================================================
def viterbi_train(hmm, feature_list):
    """Performs Viterbi HMM training given a list of features

	Args:
		hmm: The HMM to be trained
		feature_list: A list of acoustic feature sequences 

	Returns
		hmm: The trained HMM
	"""

    ndim = feature_list[0].shape[1] # feature dimension
    num_seqs = len(feature_list) # number of obs sequences

    # ------------------------------------------------------
    # Initialise HMM with uniform state segmentations
    # ------------------------------------------------------
    # For accumulating state obs across each obs sequence
    state_obs = [np.empty((0, ndim)) for s in range(hmm.num_states)]
    for n in range(num_seqs):
        # ----- Get uniform state segmentations
        segs = np.array_split(np.arange(len(feature_list[n])), hmm.num_states)

        # ----- Accumulate state obs based on uniform state segmentations
        for s in range(hmm.num_states):
            state_obs[s] = np.concatenate((state_obs[s], feature_list[n][segs[s], :]), axis=0)

    # ----- Initialise output pdfs based on uniform segmentations
    for s in range(hmm.num_states):
        hmm.states[s].fit(state_obs[s])

    # ------------------------------------------------------
    # Perform Viterbi training iteratively
    # ------------------------------------------------------
    for iter in range(NUM_ITERATIONS):

        total_errors = 0

        # ------------------------------------------------------
        # Accumulate statistics using Viterbi alignment
        # ------------------------------------------------------
        # For accumulating state obs across each obs sequence
        state_obs = [np.empty((0, ndim)) for s in range(hmm.num_states)]
        # For accumulating state transitions
        state_trans = np.zeros(hmm.num_states)

        for n in range(num_seqs):
            # ----- Get the Viterbi alignment
            log_prob, state_seq = hmm.viterbi_decoding(feature_list[n])
            total_errors += log_prob

            # ----- Accumulate statistics based on the alignment
            # For each state, accumulate assigned observations and state 
            # transitions
            # ====>>>>
            # ====>>>> FILL WITH YOUR CODE HERE
            # ====>>>>

        # ------------------------------------------------------
        # Update output pdfs and transition probabilities
        # ------------------------------------------------------
        # ----- Update output pdfs
        for s in range(hmm.num_states):
            hmm.states[s].fit(state_obs[s])

        # ----- Update transition probabilities (remember to take log)
        # For left-to-right no-skip HMMs, each state has only one transition to
        # other states per obs sequence and the rest are all self-transitions
        for s in range(hmm.num_states):
            hmm.log_transp[s, s] = np.log((state_trans[s] - num_seqs) / state_trans[s])
            if s + 1 < hmm.num_states:
                hmm.log_transp[s, s+1] = np.log(num_seqs / state_trans[s])


        print('Iteration {0}: log likelihood = {1}'.format(iter+1, total_errors))

    return hmm


def train_HMMs(file_list, feature_type='fbank'):
    """Performs Viterbi HMM training giveing a list of utterances

	Args:
		file_list: A list of utterances for training GMMs
		feature_type: Acoustic feature type ('fbank' or 'mfcc', default 'fbank')

	Returns
		A list of trained HMMs
	"""
    print('==== Training feature type: {}'.format(feature_type))
    print('Number of utterances: {}'.format(len(file_list)))

    hmm_set = [HMM(num_states=NUM_STATES, num_mixtures=NUM_MIXTURES) for hmm_id in range(NUM_DIGITS)]

    for hmm_id in range(NUM_DIGITS):

        hmm_label = DIGITS[hmm_id]
        feature_list = []

        # Accumulate training data for each model
        for fn in file_list:
            # Obtain the label for the data
            label = os.path.splitext(os.path.basename(fn))[0][0]
            if hmm_label == label:
                # Compute features
                fs_hz, signal = wav.read(fn)
                features = feat.compute_features[feature_type](signal, fs_hz)
                # Put all utterances for the same digit in a list
                feature_list.append(features)

        # Train each HMM
        print('\n---- Training digit model "{0}" with {1} utterances'.format(hmm_label, len(feature_list)))
        viterbi_train(hmm_set[hmm_id], feature_list)

    return hmm_set


# =================================================================
# MAIN FUNCTION 
# =================================================================
def main():
    # Read training set list
    with open(TRAIN_LIST) as f:
        flist = f.readlines()
    flist = [x.strip() for x in flist]

    # Train HMMs
    hmm_set = train_HMMs(flist, feature_type=FEATURE_TYPE)

    # Store models
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(hmm_set, f)

    print('---- All done. HMMs are saved in "{0}"'.format(MODEL_FILE))

# =================================================================
if __name__ == '__main__':
    main()
