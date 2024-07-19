# Denoising AutoEncoder and Variational AutoEncoder with ResNet-style Architecture

This project involves training two types of autoencoders, a Denoising AutoEncoder (DAE) and a Denoising Variational AutoEncoder (DVAE), using ResNet-style architectures. The primary goals are to design and implement the models, ensure the inclusion of residual connections, and visualize the embeddings using 3D t-SNE plots after every 10 epochs.

## Project Structure

- `data/`: Contains the dataset used for training and testing.
- `models/`: Contains the implementations of the DAE and DVAE.
- `notebooks/`: Jupyter notebooks for experimentation and visualization.
- `scripts/`: Python scripts for training the models and plotting embeddings.
- `results/`: Directory to save model checkpoints, loss logs, and t-SNE plots.
- `README.md`: Project documentation.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- scikit-learn
- matplotlib
- seaborn





## Models

### ResNet-style Denoising AutoEncoder (DAE)

- Encoder and decoder follow ResNet-style architecture.
- Residual connections are added after every two convolutional or convolutional-batchnorm layers.

### ResNet-style Denoising Variational AutoEncoder (DVAE)

- Encoder and decoder follow ResNet-style architecture.
- Residual connections are added after every two convolutional or convolutional-batchnorm layers.
- Variational inference is used in the encoder to sample embeddings.

## Results

- Model checkpoints, loss logs, and t-SNE plots will be saved.

