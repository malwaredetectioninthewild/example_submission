# Start from the specified base image
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app

# Install dependencies
RUN pip install scikit-learn==1.4.2 scipy==1.13.0

# Copy the submission files into the container
COPY ExampleSubmissionFiles/ /app/

# Set the default command to run the script
ENTRYPOINT ["python", "/app/main.py"]
