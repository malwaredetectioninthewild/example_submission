import numpy as np

from .tokenization import tokenize_report
from .featurization import filter_report_w_vocabulary, filtered_report_to_sequence, sequences_to_ngrams
from .vectorization import ngram_sequence_to_feature_counts

def get_featurized_input_ngrams(reports, kept_tokens_info, pca):
    
    fcounts_all = []

    for report in reports:
        # tokenize the report
        trep = tokenize_report(report, add_depth=False, concat=False, return_token_types=True)

        # use the dictionary to remove rare tokens
        filtered_report = filter_report_w_vocabulary(trep, kept_tokens_info)

        # flatten the report into a sequences of tokens (one sequence for each action type in the report)
        sequence = filtered_report_to_sequence(filtered_report)

        # extract the n-grams from the token sequence
        feats = sequences_to_ngrams([sequence], n=2)

        #  get bag-of-ngrams from the sequnce of n-grams using feature hashing
        fcounts = ngram_sequence_to_feature_counts(feats, 14)

        # convert the n-grams counts to log scale
        fcounts_scaled = np.log2(fcounts+1)

        fcounts_all.append(fcounts_scaled)

    # apply PCA to the n-gram-counts vector
    fcounts_transformed = pca.transform(np.vstack(fcounts_all))
    
    return fcounts_transformed
