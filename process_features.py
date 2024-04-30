import os
import pickle
import argparse
import numpy as np
import json

# feature processing routines, specific to each submission
import auxiliary_scripts.feature_processor_main as proc

# this is the path that contains files specific to your submission
auxiliary_files_path = 'auxiliary_files'

# read the raw reports in standardize format and convert them to the feature vectors the model expects
def write_feature_vectors(dataset_path, features_filepath):

    # PCA learned on the training set for dimensionality reduction
    with open(os.path.join(auxiliary_files_path, 'pca.pkl'), 'rb') as fp:
        pca = pickle.load(fp)

    # the token dictionary that contains the top-10k most frequent tokens (to eliminate the rare tokens)
    with open(os.path.join(auxiliary_files_path, 'tokens_dict.pkl'), 'rb') as fp:
        kept_tokens_dict = pickle.load(fp)


    filenames = os.listdir(dataset_path)

    report_indices = []
    features = []
    for file in filenames:

        cur_index = int(file.split('.')[0])

        with open(os.path.join(dataset_path, file), 'r') as fp:
            raw_trace = json.load(fp)
        
        cur_feats = proc.get_featurized_input_ngrams([raw_trace], kept_tokens_dict, pca)
        features.append(cur_feats)
        report_indices.append(cur_index)

    report_indices = np.asarray(report_indices)

    assert np.all(np.sort(report_indices) == np.arange(len(filenames)))

    features = np.vstack(features)[np.argsort(report_indices)]
    
    print(features.shape)

    # write the feature vectors to a file as an np array
    with open(features_filepath, 'wb') as fp:
        np.save(fp, features)

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Malware Detection In the Wild Leaderboard (https://malwaredetectioninthewild.github.io) Example Submission for Feature Processing.')

    parser.add_argument('--dataset_path', type=str, help='Path to the folder input traces for inference are located', default='./dataset')

    parser.add_argument('--features_filepath', type=str, help='Path to write the processed features in a format that is expected by the model.', default='./features.pkl')

    args = parser.parse_args()

    print(args)

    write_feature_vectors(args.dataset_path, args.features_filepath)