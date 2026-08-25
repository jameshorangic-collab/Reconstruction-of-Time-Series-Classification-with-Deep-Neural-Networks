# Reconstruction Methodology Notes

This file documents the components implemented for the from-scratch NumPy reconstruction of the Multi-Layer Perceptron (MLP), along with the training and evaluation behavior used in the experiments.

## Network Architecture

The faithful MLP follows this sequence:

`Input -> Dropout(0.1) -> Linear(500) -> ReLU -> Dropout(0.2) -> Linear(500) -> ReLU -> Dropout(0.2) -> Linear(500) -> ReLU -> Dropout(0.3) -> Linear(numberOfClasses) -> Softmax`

For Adiac, the example configuration uses an input length of 176 and 37 output classes. The no-dropout ablation preserves the same dense architecture while removing all four dropout operations.

## Random Seed and Reproducibility

Two random sources are controlled at the beginning of training. Python's `random.seed(seed)` controls the order produced by `random.shuffle`, while `setSeed(seed)` resets the shared NumPy random-number generator used by the custom layers. This shared generator controls both Glorot weight initialization and dropout masks.

## Data Preprocessing and Normalization

The first value of each row is treated as the class label and the remaining values as the time-series input. A single global mean and standard deviation are calculated from all input values in the training set. Each training input is normalized as:

`(x - meanOfTrain) / standardDeviationOfTrain`

The test set is normalized using the same mean and standard deviation calculated from the training set. Test-set statistics are not used for normalization.

This reconstruction does **not** implement a Batch Normalization (BatchNorm) layer. The normalization above is input-data preprocessing, separate from the neural-network layers.

## Mini-Batching and Shuffling

Training samples are shuffled before every epoch using Python's `random.shuffle`. The shuffled samples are divided into mini-batches of 16. Inputs are arranged in `features x batch` form, and class labels are converted into one-hot target vectors.

## Linear Layer

The custom `Linear` layer implements a fully connected transformation:

`output = W @ x + b`

Weights are initialized with Glorot/Xavier uniform initialization using the limit:

`sqrt(6 / (inputSize + outputSize))`

Biases begin at zero. During backpropagation, the layer calculates the average weight gradient and mean bias gradient across the mini-batch, then returns the gradient with respect to the previous layer.

## ReLU Activation

The custom ReLU activation performs:

`max(0, x)`

During backpropagation, incoming gradients are passed through locations where the stored input was greater than zero and are set to zero elsewhere.

## Dropout

The faithful reconstruction uses inverted dropout. A binary mask is sampled for each forward pass, dropped activations are set to zero, and surviving activations are divided by `1 - dropoutRate`.

The dropout rates are:

- Input: `0.1`
- After first hidden ReLU: `0.2`
- After second hidden ReLU: `0.2`
- After third hidden ReLU: `0.3`

The same mask and scaling are applied to the gradient during the corresponding backward pass. Because inverted dropout performs its scaling during training, dropout is not applied during evaluation.

## Softmax

The custom `Softmax` converts the output logits into class probabilities. Before exponentiation, the maximum logit in each sample is subtracted for numerical stability. The exponentials are then divided by their column-wise sum.

## Cross-Entropy Loss

Training uses categorical cross-entropy:

`-sum(target * log(prediction + epsilon))`

The loss is accumulated across all mini-batches and divided by the number of samples to produce the training loss for each epoch. `epsilon = 1e-8` is added inside the logarithm for numerical stability.

For the softmax-cross-entropy combination, the initial gradient used for backpropagation is:

`prediction - targetBatch`

## Backpropagation

Gradients are propagated through the network in the reverse order of the forward pass. Each layer stores the information it needs during its forward operation, then uses that stored state in its backward operation. In the faithful model, gradients also pass through each matching dropout mask in reverse order.

## Adadelta Optimizer

The reconstruction includes a from-scratch implementation of Adadelta. Each weight matrix and bias vector has its own optimizer instance and therefore its own running state.

The optimizer uses:

- `rho = 0.95`
- `epsilon = 1e-8`

It maintains exponentially weighted moving averages of squared gradients and squared updates, scales the current gradient using their root-mean-square values, and applies the resulting update to the parameter.

## Learning-Rate Reduction

The training scripts use the learning-rate sequence:

`1.0 -> 0.5 -> 0.25 -> 0.125 -> 0.1`

The scheduler monitors training loss with a patience of 200 epochs. An epoch counts as a scheduler improvement only when the loss improves by more than `1e-4`. Once patience is exhausted, the next learning rate in the sequence is applied. The minimum learning rate is `0.1`.

## Model Checkpointing

Checkpointing and learning-rate scheduling use separate best-loss variables. A checkpoint is saved whenever the current training loss is strictly lower than the previous checkpoint loss, even if the improvement is smaller than `1e-4`.

The saved `.npz` checkpoint contains the weights and biases for all four linear layers. This preserves the model state corresponding to the minimum observed training loss.

## Training Configuration

The reconstruction trains for 5000 epochs with a mini-batch size of 16. The faithful training script includes the four dropout operations described above. `Train_Without_Dropout.py` is a controlled ablation that removes dropout while leaving the remaining training procedure unchanged.

The no-dropout model is not the faithful architecture; it is used to test how sensitive the observed behavior is to dropout regularization.

## Evaluation

`Test.py` recreates the dense/ReLU network, loads the saved checkpoint parameters, and normalizes the test data using statistics from the training set. Dropout is omitted during testing, as required by the inverted-dropout implementation.

For each test sample, the predicted class is the index of the largest softmax probability. This is compared with the one-hot encoded true class, and final accuracy is calculated as the number of correct predictions divided by the total number of test samples.

## Additional SGD Implementation

`SGD.py` contains a basic stochastic-gradient-descent optimizer that updates a parameter with `parameter -= learningRate * gradient`. It was implemented as a supporting optimizer but is not used in the faithful MLP experiments, which use Adadelta.

## Results

Experimental outputs and multi-seed summaries are recorded separately in [`Results.md`](./Results.md).
