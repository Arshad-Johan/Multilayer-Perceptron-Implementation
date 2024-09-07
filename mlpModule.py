import numpy as np

class MLP:
    def __init__(self, nInput, nHidden, nOutput):
        self.nInput = nInput
        self.nHidden = nHidden
        self.nOutput = nOutput
        self.WList = [np.random.rand(nHidden, nInput+1), np.random.rand(nOutput, nHidden+1)]
        self.sigmoid = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))

    def compute2Arg(self, v1, v2):
        return self.compute([v1, v2])[0]

    def compute(self, inputValues):
        inputValues = np.append(1, inputValues)
        self.lastInput = inputValues
        self.lastHiddenOutput = self.sigmoid(np.dot(self.WList[0], self.lastInput))
        self.lastHiddenOutput = np.append(1, self.lastHiddenOutput)
        self.lastOutput = self.sigmoid(np.dot(self.WList[1], self.lastHiddenOutput))
        return self.lastOutput

class Backpropagation:
    def __init__(self, mlp, alpha, regularization):
        self.alpha = alpha
        self.regularization = regularization
        self.mlp = mlp
        self.DeltaList = [np.zeros_like(w) for w in mlp.WList]

    def iterate(self, inputValues, outputValues):
        for x, y in zip(inputValues, outputValues):
            self.mlp.compute(x)
            delta3 = self.mlp.lastOutput - y

            # Gradient for the output layer
            Od3 = np.dot(self.mlp.WList[1].T, delta3)
            gz2 = self.mlp.lastHiddenOutput[1:] * (1 - self.mlp.lastHiddenOutput[1:])
            delta2 = delta3 @ self.mlp.WList[1][:, 1:] * gz2

            self.DeltaList[1] += np.outer(delta3, self.mlp.lastHiddenOutput)
            self.DeltaList[0] += np.outer(delta2, self.mlp.lastInput)

        m = len(inputValues)
        self.DeltaList[0] = (self.DeltaList[0] / m) + (self.regularization * self.mlp.WList[0])
        self.DeltaList[1] = (self.DeltaList[1] / m) + (self.regularization * self.mlp.WList[1])
        self.mlp.WList[0] -= self.alpha * self.DeltaList[0]
        self.mlp.WList[1] -= self.alpha * self.DeltaList[1]