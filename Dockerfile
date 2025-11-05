# Start from the desired NVIDIA base image
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Install Python and clean up apt lists to keep image size small
RUN apt-get update && apt-get install -y python3 python3-pip cmake curl 
# Set the working directory
WORKDIR /workspace
