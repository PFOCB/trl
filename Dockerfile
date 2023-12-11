# Use the official NVIDIA CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set the working directory
WORKDIR /app

# Install basic dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    wget \
    sudo

# Install Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    pip \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA cuDNN libraries
# RUN apt-get update && apt-get install -y \
#     libcudnn8 \
#     libcudnn8-dev \
#     && rm -rf /var/lib/apt/lists/*

# Set the default Python version
RUN ln -s /usr/bin/python3 /usr/bin/python

# RUN pip install torch
COPY ./trl .
RUN pip install -r requirements.txt
# RUN pip install peft scipy bitsandbytes
# RUN pip install torch torchvision accelerate
# # scipy peft butsandbytes

RUN mkdir "./vicuna-7b-llama2"
COPY ["model weights/Vicuna 7b llama2", "vicuna-7b-llama2/"]

RUN mkdir "./dataset"
COPY ["ChatsDataset/", "dataset/"]
RUN pip install datasets 
RUN pip install SentencePiece
###TODO: we need to change the path of the model inside inference.py and output-47k/adapter_config.json

# TODO:
# The inputs are:
# - instruction for the intent
# - user input
# - system message



ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# RUN sudo apt update && sudo apt install nvidia-cuda-toolkit -y

CMD ["python", "examples/scripts/sft_trainer.py", "--model_name", "./vicuna-7b-llama2", "--dataset_name", "./dataset/run2", "--load_in_8bit", "--use_peft", "--batch_size", "8", "--gradient_accumulation_steps", "1", "--num_train_epochs", "1"]

# ENTRYPOINT ["python"]

# python examples/scripts/sft_trainer.py --model_name "/home/g1-s23/dev/Vicuna 7b llama2" --dataset_name "/home/koko/dataset" --load_in_8bit --use_peft --batch_size 8 --gradient_accumulation_steps 1 --num_train_epochs 1

# CMD  ["examples/scripts/sft_trainer.py", "--model_name", "/home/g1-s23/dev/Vicuna 7b llama2", "--dataset_name", "/home/koko/dataset", "--load_in_8bit", "--use_peft", "--batch_size", "8", "--gradient_accumulation_steps", "1", "--num_train_epochs", "1"]

# python examples/scripts/sft_trainer.py --model_name ./vicuna-7b-llama2 --dataset_name ./dataset/run2 --load_in_8bit --use_peft --batch_size 8 --gradient_accumulation_steps 1 --num_train_epochs 1


