## The documentation for the submission format for  **[Malware Detection In the Wild Leaderboard](https://malwaredetectioninthewild.github.io/)** 

### [See our SaTML 2025 paper](https://arxiv.org/abs/2405.06124)

### Before You Start

Go to [the Github repository for our data](https://github.com/malwaredetectioninthewild/explore_data) and request access to the **Train Data**.

Once you obtain the dataset and a `<team_identifier>` tag from this request, you are ready to develop.

## Submission Instructions

We accept submissions in Docker image format. 

First build your Docker image using `docker build` (with `-t <team_identifier>` to set the image name). 

Next, package the image as a `.tar` file using the following command: 

`docker save -o <team_identifier>.tar <team_identifier>`

This `<team_identifier>.tar` file will be your submission.

### Sending Your Submission

To make a submission, email [Yigitcan Kaya](https://yigitcankaya.github.io) at **yigitcan at ucsb dot edu**. 

Use the subject line `MalwareITW Submission: <team_identifier>` for your emails and include a Google Drive download link for your image's `.tar` file in the email's body.

Each team can only make one submission a week.

### Submission Execution

After receiving your submission, we will execute the following commands:

1) To load your image from the submitted `.tar` file:

`docker load -i /path/<team_identifier>.tar`

2) To get the prediction results (saved as `results.txt` under `OutputFiles`) on the JSON testing trace files under `TestFiles`:

`docker run --rm --network none --gpus all  -v /host/path/TestFiles:/TestFiles/ -v /host/path/OutputFiles:/OutputFiles/ <team_identifier> /TestFiles /OutputFiles`

### Computation Limits

Your submission's image will be run in a non-networked container with access to a CUDA-compatible GPU (e.g., for using PyTorch). 

The container will be given the following resources:

* 4 Processor Cores
* 32 GB RAM
* 10 GB Storage Space (for temporary files of your submission)
* CUDA-compatible GPU with 16 GB VRAM

Our complete testing data contains around 400K trace files. We will timeout and terminate your container **4 hours** after `docker run` is called.

### Recording Submission Run-time Information

We will record the total run-time of your submission using the following Linux command:

`time (docker run... > submission_output.txt 2>&1) 2> timing_info.txt`

Please see `ExampleSubmissionFiles/main.py:135` for instructions to print your feature processing run-time information to stdout, which we will use to report your feature processing and inference times separately in our leaderboard. When this information is missing, we will assume that the whole run-time of the container was for inference.

### Including Multiple Models Within the Same Submission

We will use the `results.txt` file your container creates under `OutputFiles` to perform our evaluation. This file contains your model's prediction scores on each JSON trace under `TestFiles`

If your container creates multiple results files, named as `results_1.txt`, `results_2.txt`, and so on, we will perform the evaluation on each, individually.

This allows participants to perform, for example, hyper-parameter tuning for their approaches without needing to make multiple submissions.

If a submission creates multiple results files, we will email the submitter team the evaluation results for each and list the best one in our leaderboard.

For each submission, at most 50 results files (indexed from 1 to 50) will be used for evaluation.

### Example Submission

We included a `Dockerfile` to build our example submission image.

We also included an example `TestFiles` directory that contains 10 JSON trace files.

The source code for this example is under `ExampleSubmissionFiles`.

To build our image, first download and extract the auxiliary files (the model file and PCA file for feature reduction) from this [Dropbox link](https://www.dropbox.com/scl/fi/17ra6gackdzs7ehgfmkk3/ExampleSubmissionFiles.zip?rlkey=a4ysw9uxe776c09u3sukxn1cm&st=h5qn2pth&dl=0) and create the `auxiliary_files` directory under `ExampleSubmissionFiles`.

Then, run `docker build` in the same directory as our `Dockerfile`.

The results files our example submission creates when successfully run on the included `TestFiles` are under the `OutputFiles` directory.

Please see `ExampleSubmissionFiles/main.py` for instructions on how to create multiple results files for evaluation.

Please note how our example submission uses batching for feature extraction (`ExampleSubmissionFiles/main.py::26`) and for inference (`ExampleSubmissionFiles/main.py::90`). This ensures that our submission is both runtime efficient and doesn't use excessive GPU or system memory, which can cause a submission to fail.


### Example Evaluation Report

For each created results file under `OutputFiles`, we will perform evaluation on both our endpoint and sandbox test datasets and generate a performance report that will be send to the submitter.

You can see `ExampleEvaluationReport.json` for an example performance report that you will receive.

Please contact us if you need more metrics or information in these reports to assist your development.
