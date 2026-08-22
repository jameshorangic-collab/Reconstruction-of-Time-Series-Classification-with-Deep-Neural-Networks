from Layers import Linear, ReLU, Dropout, Softmax, setSeed
from Adadelta import Adadelta
import numpy as np
import random

seed = 10
random.seed(seed)
setSeed(seed)

# File selection
trainFile = "Adiac_TRAIN.txt"
numberOfClasses = 37
inputLength = 176
weightsFile = "mlp_weights_dropout.npz"

# Optimizer
rho = 0.95
epsilon = 1e-8

learningRates = [1.0, 0.5, 0.25, 0.125, 0.1]
learningRatesIndex = 0
learningRate = learningRates[learningRatesIndex]

sinceImprovement = 0
patience = 200
thresholdImprovement = 1e-4

optW1 = Adadelta(learningRate, rho, epsilon)
optW2 = Adadelta(learningRate, rho, epsilon)
optW3 = Adadelta(learningRate, rho, epsilon)
optW4 = Adadelta(learningRate, rho, epsilon)

optB1 = Adadelta(learningRate, rho, epsilon)
optB2 = Adadelta(learningRate, rho, epsilon)
optB3 = Adadelta(learningRate, rho, epsilon)
optB4 = Adadelta(learningRate, rho, epsilon)

# Training configuration and tracking
lossTotal = 0
sample = 0
bestCheckpointLoss = np.inf
schedulerBestLoss = np.inf
epochs = 5000
batchSize = 16

# Normalize using statistics calculated only from the training set
arrOfLines = []
labelsOfLines = []

with open(trainFile, "r", encoding="utf-8") as file:
    for line in file:
        hold = line.strip().split(",")

        labelsOfLines.append(hold[0])

        inputs = np.array(hold[1:], dtype=float)
        arrOfLines.append(inputs)

meanOfTrain = np.mean(arrOfLines)
standardDeviationOfTrain = np.std(arrOfLines)
normalizedTrainFile = "normalized_" + trainFile

with open(normalizedTrainFile, "w", encoding="utf-8") as file:
    for i in range(len(arrOfLines)):
        normalizedLine = (arrOfLines[i] - meanOfTrain) / standardDeviationOfTrain

        file.write(labelsOfLines[i] + "," + ",".join(map(str, normalizedLine)) + "\n")

# Create network
dropout1 = Dropout(0.1)
layer1 = Linear(inputLength, 500)
relu1 = ReLU()
dropout2 = Dropout(0.2)
layer2 = Linear(500, 500)
relu2 = ReLU()
dropout3 = Dropout(0.2)
layer3 = Linear(500, 500)
relu3 = ReLU()
dropout4 = Dropout(0.3)
layer4 = Linear(500, numberOfClasses)

# Training
for i in range(1, epochs + 1):

    with open(normalizedTrainFile, "r", encoding="utf-8") as file:
        shuffledLines = file.readlines()

    # Shuffle training samples before batching
    random.shuffle(shuffledLines)

    for start in range(0, len(shuffledLines), batchSize):
        lines = shuffledLines[start:start + batchSize]

        inputBatch = []
        targetBatch = []

        for line in lines:
            hold = line.strip().split(",")
            inputValues = [float(num) for num in hold[1:]]
            inputBatch.append(inputValues)

            target = np.zeros(numberOfClasses)
            target[int(float(hold[0])) - 1] = 1
            targetBatch.append(target)

        inputBatch = np.transpose(np.array(inputBatch))
        targetBatch = np.transpose(np.array(targetBatch))

        # Forward pass
        d1 = dropout1.forward(inputBatch)
        z1 = layer1.forward(d1)
        a1 = relu1.forward(z1)

        d2 = dropout2.forward(a1)
        z2 = layer2.forward(d2)
        a2 = relu2.forward(z2)

        d3 = dropout3.forward(a2)
        z3 = layer3.forward(d3)
        a3 = relu3.forward(z3)

        d4 = dropout4.forward(a3)
        z4 = layer4.forward(d4)

        prediction = Softmax.forward(z4)
        gradient = prediction - targetBatch

        lossTotal += -np.sum(targetBatch * np.log(prediction + epsilon))
        sample += targetBatch.shape[1]

        # Backward pass
        gradient = layer4.backward(gradient)
        gradient = dropout4.backward(gradient)

        gradient = relu3.backward(gradient)
        gradient = layer3.backward(gradient)
        gradient = dropout3.backward(gradient)

        gradient = relu2.backward(gradient)
        gradient = layer2.backward(gradient)
        gradient = dropout2.backward(gradient)

        gradient = relu1.backward(gradient)
        gradient = layer1.backward(gradient)
        gradient = dropout1.backward(gradient)

        # Optimizer
        optW1.step(layer1.W, layer1.dW)
        optB1.step(layer1.b, layer1.db)

        optW2.step(layer2.W, layer2.dW)
        optB2.step(layer2.b, layer2.db)

        optW3.step(layer3.W, layer3.dW)
        optB3.step(layer3.b, layer3.db)

        optW4.step(layer4.W, layer4.dW)
        optB4.step(layer4.b, layer4.db)

    # Record epoch loss and training state
    loss = lossTotal / max(sample, 1)
    print("Epoch:", i, "Loss:", loss, "LR:", learningRate, "Since improvement:", sinceImprovement)

    # ModelCheckpoint behavior: save on any new minimum training loss
    if loss < bestCheckpointLoss:
        bestCheckpointLoss = loss

        np.savez(
            weightsFile,
            layer1_W=layer1.W,
            layer1_b=layer1.b,
            layer2_W=layer2.W,
            layer2_b=layer2.b,
            layer3_W=layer3.W,
            layer3_b=layer3.b,
            layer4_W=layer4.W,
            layer4_b=layer4.b
        )

    # ReduceLROnPlateau behavior: require a minimum improvement of 1e-4
    if loss < schedulerBestLoss - thresholdImprovement:
        schedulerBestLoss = loss
        sinceImprovement = 0
    else:
        sinceImprovement += 1

        if sinceImprovement >= patience and learningRatesIndex < (len(learningRates) - 1):
            learningRatesIndex += 1
            learningRate = learningRates[learningRatesIndex]

            optW1.learningRate = learningRate
            optW2.learningRate = learningRate
            optW3.learningRate = learningRate
            optW4.learningRate = learningRate

            optB1.learningRate = learningRate
            optB2.learningRate = learningRate
            optB3.learningRate = learningRate
            optB4.learningRate = learningRate

            sinceImprovement = 0

    lossTotal = 0
    sample = 0
