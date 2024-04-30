import os
import argparse
import numpy as np

from auxiliary_scripts.utils import load_model

# this is the path that contains files specific to your submission
auxiliary_files_path = 'auxiliary_files'


def write_results(model, features_filepath, results_filepath):

    # load the feature vectors file written by the process_features script
    with open(features_filepath, 'rb') as fp:
        feature_vectors = np.load(fp)

    # make inference to get prediction scores from the model, we care about the malware-ness score 
    # ([:,0] gives the benign-ness score and [:,1 ] gives the malwareness score in our model implementation)
    probs = model.predict_proba(feature_vectors)[:,1].tolist()

    # write the results to a json file whose path was passed as an argument to the script
    with open(results_filepath, 'w') as fp:
        fp.write("\n".join(map(str, probs)))
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Malware Detection In the Wild Leaderboard (https://malwaredetectioninthewild.github.io) Example Submission Format for Inference.')

    parser.add_argument('--device', type=str, help='Device for performing model inference on feature vectors (GPU or CPU)', default='cuda')

    parser.add_argument('--features_filepath', type=str, help='Path to load the processed features in a format that is expected by the model (must be written by process_features.py script).', default='./features.pkl')

    parser.add_argument('--results_filepath', type=str, help='File path to the file where the predicted malware probabilities for each trace in the dataset will be written as a dictionary (filenames are the keys and the scores are the values)', default='./results.pkl')

    args = parser.parse_args()


    model_path = os.path.join(auxiliary_files_path, 'model.dat')

    model = load_model(model_path, device=args.device, train=False)

    write_results(model, args.features_filepath, args.results_filepath)