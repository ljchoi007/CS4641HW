# ============================================================
# HW5 Coding: Neural Networks, Embeddings, & Zero-Shot Similarity
#
# Description:
# This assignment covers the fundamentals of building, training,
# and using deep learning models. You will implement a Multi-Layer
# Perceptron (MLP), a feature extraction pipeline, a standard
# training loop, and a zero-shot similarity classifier.
# ============================================================

from __future__ import annotations
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

def build_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Module:
    """
    Build a multi-layer perceptron for classification or regression.

    Architecture Requirements:
    Linear -> ReLU -> ... -> Linear -> ReLU -> Linear (output).
    Note: There should be NO activation function after the final linear layer.

    Instructions:
    1. Iterate through hidden_dims to create pairs of nn.Linear and nn.ReLU.
    2. Use nn.Sequential to wrap your list of layers into a single module.
    """
    pre_outputs = [input_dim] + hidden_dims
    layers = []
    #hidden layers with a relu
    for i in range(len(pre_outputs) - 1):
        layers.append(nn.Linear(pre_outputs[i], pre_outputs[i+1]))
        layers.append(nn.ReLU())
    #output layer
    layers.append(nn.Linear(pre_outputs[len(pre_outputs) - 1], output_dim))
    return nn.Sequential(*layers)


def embed_data(images: torch.Tensor, model: nn.Module) -> np.ndarray:
    """
    Run a PyTorch model on a batch of images to extract 'embeddings'.

    PyTorch Steps:
    1. model.eval(): Put the model in evaluation mode (disables Dropout/BatchNorm).
    2. torch.no_grad(): Prevents PyTorch from calculating gradients, saving significant memory and time.
    3. Return as NumPy: Move the result to the CPU and convert it to a NumPy array.

    Parameters
    ----------
    images : torch.Tensor
        Input batch of images.
    model : torch.nn.Module
        The model used to extract features.

    Returns
    -------
    embeddings : np.ndarray (dtype float32)
    """
    model.eval()
    with torch.no_grad():
        embeds = model(images)
    to_cpu = embeds.cpu()
    return to_cpu.numpy().astype('float32')


def training_step(
    model: nn.Module,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    loss_fn: nn.Module,
    lr: float,
) -> float:
    """
    Perform exactly one gradient-descent step on a mini-batch using SGD.

    PyTorch Steps:
    1. Training Mode: model.train() to put the model in training mode.
    2. Initialize Optimizer: Create the optimizer with torch.optim.SGD(model.parameters(), lr=lr).
    3. Zero Gradients: Clear old gradients with optimizer.zero_grad() so they don't accumulate.
    4. Forward Pass: Pass batch_x through the model to get predictions.
    5. Compute Loss: Use loss_fn(outputs, batch_y) to calculate the model's error.
    6. Backward Pass: Call loss.backward() to compute the gradient for every parameter.
    7. Update Parameters: Call optimizer.step() to update the parameters in the direction of the gradient.
    8. Return result: Move the loss result to the CPU and detach it from the graph.

    Parameters
    ----------
    model : torch.nn.Module
    batch_x : torch.Tensor
        Batch of inputs for the model.
    batch_y : torch.Tensor
        Batch of targets (shape/type expected by loss_fn).
    loss_fn : torch.nn.Module
    lr : float
        Learning rate (step size) for SGD.

    Returns
    -------
    loss_value : float
        The scalar loss after the update step.
    """

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()
    outputs = model(batch_x)
    loss = loss_fn(outputs, batch_y)
    loss.backward()
    optimizer.step()
    scalar_loss = loss.item()
    return scalar_loss


def zero_shot_classify(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Predict classes by matching image embeddings to the most similar text description.

    Instructions:
    1. Use NumPy to calculate Cosine Similarity between image and text vectors.
       Similarity = (A · B) / (||A|| * ||B||).
    2. For each image embedding, compare it against all K text embeddings.
    3. Pick the index (0 to K-1) that maximizes this similarity.

    Parameters
    ----------
    image_embeddings : np.ndarray (N, D)
    text_embeddings : np.ndarray (K, D)

    Returns
    -------
    predictions : np.ndarray (N,) - The index (0 to K-1) for each image.
    """
    predictions = np.zeros((image_embeddings.shape[0]), dtype=float)
    for i in range(image_embeddings.shape[0]):
        cur_img = image_embeddings[i]
        dots = text_embeddings @ cur_img
        norms = np.linalg.norm(text_embeddings, axis=1) * np.linalg.norm(cur_img)
        cos_sims = dots / norms
        predictions[i] = np.argmax(cos_sims)
    return predictions
