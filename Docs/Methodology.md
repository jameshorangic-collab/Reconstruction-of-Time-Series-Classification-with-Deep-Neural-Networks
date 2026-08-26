# Reconstruction Methodology Notes

This file documents the components implemented for the from-scratch NumPy reconstruction of the Multi-Layer Perceptron (MLP), along with the training and evaluation behavior used in the experiments.

## Network Architecture

The MLP follows this sequence:

`Input -> Dropout(0.1) -> Linear(500) -> ReLU -> Dropout(0.2) -> Linear(500) -> ReLU -> Dropout(0.2) -> Linear(500) -> ReLU -> Dropout(0.3) -> Linear(numberOfClasses) -> Softmax`

For Adiac, the example configuration uses an input length of 176 and 37 output classes. The no-dropout version preserves this architecture while removing all four dropout operations from the forward pass and backpropagation.

## Random Seed and Reproducibility

Two random sources are controlled at the beginning of training. Python's `random.seed(seed)` controls the training-sample shuffle, while the custom `setSeed(seed)` function resets the shared NumPy random-number generator used for layer initialization and dropout. Together, these controls provide deterministic reproducibility within this reconstruction when the same seed and configuration are used. They do not imply identical random-number trajectories to historical Keras/TensorFlow implementations.

## Data Preprocessing and Normalization

The first value of each row is treated as the class label and the remaining values as the time-series input. A single global mean and standard deviation are calculated from all input values in the training set. Each training input is normalized as:

`(x - meanOfTrain) / standardDeviationOfTrain`

The test set is normalized using the same mean and standard deviation calculated from the training set. Test-set statistics are not used for normalization.

## Mini-Batching and Shuffling

Training samples are shuffled before every epoch. The shuffled samples are divided into mini-batches of 16 and are run together through both the forward pass and backpropagation. Both training scripts run for 5000 epochs.

## Linear Layer

The custom `Linear` layer implements a fully connected transformation:

`output = W @ x + b`

Weights are initialized with Glorot/Xavier uniform initialization using the limit:

`sqrt(6 / (inputSize + outputSize))`

Biases begin at zero.

## ReLU Activation

`max(0, x)`

## Dropout

The faithful reconstruction uses inverted dropout. A binary mask is sampled for each forward pass, dropped activations are set to zero, and surviving activations are divided by `1 - dropoutRate`.

The dropout rates are:

- Input: `0.1`
- After first hidden ReLU: `0.2`
- After second hidden ReLU: `0.2`
- After third hidden ReLU: `0.3`

The same mask and scaling are applied to the gradient during the corresponding backward pass. Dropout is not applied during evaluation.

## Softmax

The `Softmax` function converts the output logits into class probabilities. Before exponentiation, the maximum logit in each sample is subtracted for numerical stability.

## Cross-Entropy Loss

Training uses categorical cross-entropy:

`-sum(target * log(prediction + epsilon))`

The loss is accumulated across all mini-batches and divided by the number of samples to produce the training loss for each epoch. `epsilon = 1e-8` is added inside the logarithm for numerical stability.

For the softmax-cross-entropy combination, the initial gradient used for backpropagation is:

`prediction - targetBatch`

## Adadelta Optimizer

The reconstruction includes a from-scratch implementation of Adadelta. Each weight matrix and bias vector has its own optimizer instance and therefore its own running state.

The optimizer uses:

- `rho = 0.95`
- `epsilon = 1e-8`

It maintains exponentially weighted moving averages of squared gradients and squared updates, scales the current gradient using their root-mean-square values, and applies the resulting update to the parameter.

## Learning-Rate Reduction

The training scripts use the learning-rate sequence:

`1.0 -> 0.5 -> 0.25 -> 0.125 -> 0.1`

The scheduler monitors training loss with a patience of 200 epochs. An epoch counts as a scheduler improvement only when the loss improves by more than `1e-4`. The learning rate decreases in the provided order when the patience threshold is reached.

## Model Checkpointing

Model checkpointing monitors training loss independently from the learning-rate scheduler. A checkpoint is saved whenever the current epoch loss is strictly smaller than `bestCheckpointLoss`. Unlike the scheduler, checkpointing does not require an improvement greater than `1e-4`. As a result, the saved weights correspond to the lowest training loss observed during the run, while `schedulerBestLoss` separately controls learning-rate reduction.

## Additional SGD Implementation

`SGD.py` contains a basic stochastic-gradient-descent optimizer that updates a parameter with `parameter -= learningRate * gradient`. It was implemented as a supporting optimizer but is not used in the faithful MLP experiments, which use Adadelta.

## Results

Experimental outputs and multi-seed summaries are recorded separately in [`Results.md`](./Results.md).
