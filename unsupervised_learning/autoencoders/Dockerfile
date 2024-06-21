# Use the official TensorFlow Docker image as the base image
FROM tensorflow/tensorflow:2.15.0

# Set environment variables to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    python3-pip \
    python3-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install TensorFlow and other necessary Python packages
RUN pip3 install \
    tensorflow==2.15 \
    tensorflow-metal \
    matplotlib \
    numpy

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Default command to run when starting the container
CMD ["bash"]
