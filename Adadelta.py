import numpy as np

class Adadelta:
  
  def __init__(self, learningRate, rho, epsilon):
      self.learningRate = learningRate
      self.rho = rho
      self.epsilon = epsilon
      self.pastGradient = 0
      self.pastUpdate = 0
      
  def step(self, parameter, gradient):
      #Calculate moving average of squared gradients
      self.pastGradient = (
        self.rho*self.pastGradient 
        +(1 - self.rho)*(np.square(gradient))
      )

      #Scale by using past RMS and update
      update = (
        np.sqrt(self.pastUpdate + self.epsilon)
        /np.sqrt(self.pastGradient + self.epsilon)
      )*gradient
    
      parameter -= self.learningRate*update

      #Save update for moving average
      self.pastUpdate = (
        self.rho*self.pastUpdate
        +(1 - self.rho)*np.square(update)
      )
