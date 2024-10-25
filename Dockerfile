# Use the official Ubuntu image as the base
FROM ubuntu:latest

# Set the working directory
WORKDIR /app

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Install required packages
RUN apt-get update && apt-get install -y wget curl

# Copy your files into the container
COPY . /app

# Install Miniconda on x86 or ARM platforms
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
    echo "Unsupported architecture: $arch"; \
    exit 1; \
    fi && \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh

# Create a new conda environment
RUN conda env create -n fairness-sim-ot-lab -f environment.yml -y

# Expose port 8888 for Jupyter Notebook
EXPOSE 8888

CMD ["bash", "lab_start.sh"]