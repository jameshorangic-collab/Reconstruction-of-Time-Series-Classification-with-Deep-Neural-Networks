class SGD:
    def __init__(self, learningRate):
        self.learningRate = learningRate
      
    def step(self, parameter, gradient):
        parameter -= self.learningRate * gradient
