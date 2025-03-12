import argparse
import os
import torch
import pickle
import json
import numpy as np
import time

from auxiliary_scripts.utils import load_model
from pathlib import Path

# feature processing routines for this submission
import auxiliary_scripts.feature_processor_main as proc

# The path that contains files specific to the example submission
# these are included in the image
auxiliary_files_path = 'auxiliary_files'

# the directory to save the temporary files created by the submission
temp_dir = './temp'

Path(temp_dir).mkdir(exist_ok=True)

# read the raw reports in standardized format and convert them to the feature vectors the model expects
def feature_processing(input_dir):

    # PCA learned on the training set for dimensionality reduction
    with open(os.path.join(auxiliary_files_path, 'pca.pkl'), 'rb') as fp:
        pca = pickle.load(fp)

    # the token dictionary that contains the top-10k most frequent tokens (to eliminate the rare tokens)
    with open(os.path.join(auxiliary_files_path, 'tokens_dict.pkl'), 'rb') as fp:
        kept_tokens_dict = pickle.load(fp)

    filenames = os.listdir(input_dir)

    report_indices = []
    features = []
    for file in filenames:

        cur_index = int(file.split('.')[0])

        with open(os.path.join(input_dir, file), 'r') as fp:
            raw_trace = json.load(fp)
        
        cur_feats = proc.get_featurized_input_ngrams([raw_trace], kept_tokens_dict, pca)
        features.append(cur_feats)
        report_indices.append(cur_index)

    report_indices = np.asarray(report_indices)

    # list of sorted filenames: 1.json, 2.json,.......<num_files>.json
    assert np.all(np.sort(report_indices) == np.arange(1, len(filenames)+1))

    # order the extracted features matrix based on the filenames
    features = np.vstack(features)[np.argsort(report_indices)]
    
    print(f'Shape of the extracted features: {features.shape}')

    # write the feature vectors to a file as an np array
    with open(os.path.join(temp_dir, 'features.npy'), 'wb') as fp:
        np.save(fp, features)

    # Write the results (predicted probabilities) to the result_{file_index}.txt file in the output_dir
def write_probs_to_results_file(output_dir, probs, model_index=None):
    # each line in the file correspond to a trace file
    # Line 1 -> Prediction score on 1.json
    # Line 2 -> Prediction score on 2.json

    # if no model_index is provided, save the results as results.txt
    save_fname = f'results_{model_index}.txt' if (model_index and model_index > 0) else 'results.txt'

    with open(os.path.join(output_dir, save_fname), 'w') as fp:
        fp.write("\n".join(map(str, probs)))

def get_model_probs():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_path = os.path.join(auxiliary_files_path, 'model.dat')

    model = load_model(model_path, device=device, train=False)

    # load the feature vectors file written by the process_features script into the output_dir
    with open(os.path.join(temp_dir, 'features.npy'), 'rb') as fp:
        feature_vectors = np.load(fp)

    # make inference to get prediction scores from the model, we care about the malware-ness score 
    # ([:,0] gives the benign-ness score and [:,1 ] gives the malwareness score in our model implementation)
    probs = model.predict_proba(feature_vectors)[:,1].tolist()

    return probs

def get_random_probs(input_dir):
    filenames = os.listdir(input_dir)
    num_files = len(filenames)

    probs = np.random.rand(num_files)
    return probs


def main():
    arg_desc = 'Malware Detection In the Wild Leaderboard Example Submission.'
    parser = argparse.ArgumentParser(description=arg_desc)

    input_help = 'The directory where the input traces for inference are located. (1.json, 2.json...)'
    output_help = 'The directory to write the results.txt file that contains the predictions '

    parser.add_argument('TestFiles', type=str, help=input_help)
    parser.add_argument('OutputFiles', type=str, help=output_help)
    
    # Parse command-line arguments
    args = parser.parse_args()
    input_dir = args.TestFiles
    output_dir = args.OutputFiles

    # Print the directories
    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")

    # Check if the input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        exit(1)
    else:
        print(f"Input directory '{input_dir}' exists.")

    # Check if the input directory exists
    if not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' does not exist.")
        exit(1)
    else:
        print(f"Output directory '{output_dir}' exists.")
    
    # INCLUDE THIS SO OUR LEADERBOARD CAN REPORT YOUR RUNTIME INFORMATION ACCURATELY
    # TIME YOUR FEATURE PROCESSING STEP SEPARATELY
    start = time.time()
    feature_processing(input_dir)
    end = time.time()

    # PRINT THIS LINE TO STDOUT
    print(f'FEATURE PROCESSING TOOK {end-start} SECONDS.')

    # inference step from the actual model
    model_probs = get_model_probs()

    # The output probabilities from the model will be written to results_1.txt
    write_probs_to_results_file(output_dir, model_probs, model_index=1)


    random_probs = get_random_probs(input_dir)
    # The random probabilities will be written to results_2.txt
    write_probs_to_results_file(output_dir, random_probs, model_index=2)


    # No model_index is provided
    # The output probabilities from the model will be written to results.txt
    write_probs_to_results_file(output_dir, model_probs, model_index=None)

    # We will perform evaluation on results.txt, results_1.txt and results.2.txt

    print("\nExecution completed successfully.")

if __name__ == "__main__":
    main()
