from Layers import Linear, ReLU, Softmax
import numpy as np

# File selection (Adiac is used as example)
testFile = "Adiac_TEST.txt"
trainFile = "Adiac_TRAIN.txt"
numberOfClasses = 37
inputLength = 176
weightsFile = "mlp_weights_dropout.npz"

correct = 0
total = 0

# Normalize test data using statistics calculated only from the training set
trainInputs = []

with open(trainFile, "r", encoding="utf-8") as file:
    for line in file:
        hold = line.strip().split(",")
        trainInputs.append(np.array(hold[1:], dtype=float))

meanOfTrain = np.mean(trainInputs)
standardDeviationOfTrain = np.std(trainInputs)
normalizedTestFile = "normalized_" + testFile

testInputs = []
testLabels = []

with open(testFile, "r", encoding="utf-8") as file:
    for line in file:
        hold = line.strip().split(",")

        testLabels.append(hold[0])
        testInputs.append(np.array(hold[1:], dtype=float))

with open(normalizedTestFile, "w", encoding="utf-8") as file:
    for i in range(len(testInputs)):
        normalizedLine = (
            testInputs[i] - meanOfTrain
        ) / standardDeviationOfTrain

        file.write(
            testLabels[i]
            + ","
            + ",".join(map(str, normalizedLine))
            + "\n"
        )

# Recreate network architecture
layer1 = Linear(inputLength, 500)
relu1 = ReLU()

layer2 = Linear(500, 500)
relu2 = ReLU()

layer3 = Linear(500, 500)
relu3 = ReLU()

layer4 = Linear(500, numberOfClasses)

# Load trained parameters
weights = np.load(weightsFile)

layer1.W = weights["layer1_W"]
layer1.b = weights["layer1_b"]

layer2.W = weights["layer2_W"]
layer2.b = weights["layer2_b"]

layer3.W = weights["layer3_W"]
layer3.b = weights["layer3_b"]

layer4.W = weights["layer4_W"]
layer4.b = weights["layer4_b"]

# Compare predicted and true class indices
def accuracy(prediction, target):
    return int(np.argmax(prediction) == np.argmax(target))

# Evaluate network on normalized test set
with open(normalizedTestFile, "r", encoding="utf-8") as file:
    for line in file:
        hold = line.strip().split(",")

        target = np.zeros((numberOfClasses, 1))
        target[int(float(hold[0])) - 1] = 1

        x = np.array(
            [float(value) for value in hold[1:]]
        ).reshape(inputLength, 1)

        # Forward pass
        z1 = layer1.forward(x)
        a1 = relu1.forward(z1)

        z2 = layer2.forward(a1)
        a2 = relu2.forward(z2)

        z3 = layer3.forward(a2)
        a3 = relu3.forward(z3)

        z4 = layer4.forward(a3)
        prediction = Softmax.forward(z4)

        correct += accuracy(prediction, target)
        total += 1

accuracyPercent = float(correct) / total
