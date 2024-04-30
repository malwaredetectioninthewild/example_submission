## The example submission for the **[Malware Detection In the Wild Leaderboard](https://malwaredetectioninthewild.github.io/)**

You are expected to provide three files as part of your submission (you can also submit auxiliary files and scripts needed for your submission):

* **requirements.txt** A list of required packages (e.g., PyTorch) to create the necessary Conda environment for running your submission. This repository includes an example of this file.

* **process_features.py**: Reads all the JSON traces (*0.json*, *1.json*, *2.json*...) in the *dataset_path* directory, converts them to a feature matrix expected by your model, and saves this feature matrix as a file at *features_filepath* path. This repository includes an example of this script. An example command that will be used on your submission is:

`python process_features.py --dataset_path example_testing_data --features_filepath features.npy`

* **inference.py**: Reads the feature matrix generated by **process_features.py** from *features_filepath* path, loads your model and other auxiliary files needed by your submission to produce prediction scores for each trace read from *dataset_path*. These predictions are saved to a text file at the *results_filepath* path. In the output file, each line corresponds to one trace: the first line is the score for *0.json*, the second line is for *1.json*, and so on. This repository includes an example of this script. An example command that will be used on your submission is:

`python inference.py --features_filepath features.npy --results_filepath results.txt --device cuda`

Based on this submission format, we have provided a small dataset (*example_testing_data*), auxiliary scripts, and files to demonstrate how to structure a submission. Please download and extract the *auxiliary_files* needed to reproduce this example from this [Dropbox link](https://www.dropbox.com/scl/fi/b37ovcwz53psot2rz1r85/auxiliary_files.zip?rlkey=x5g39stfpxc6enuuhukbeydy0&st=r3jkka0z&dl=1).


This repository also includes **features.npy** that is produced by our included **process_features.py** (the feature matrix accepted by our model as input); and **results.txt** as an example valid output produced by our included **inference.py**. If you're trying to reproduce our example submission, please compare your results against these files.

Finally, please visit our [data description repository](https://github.com/malwaredetectioninthewild/explore_data) to learn about the standardized trace format.

<h3> Contact </h3>

*Contact information will be released when the leaderboard is officially launched*
