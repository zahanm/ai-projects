
import util, random, math, sys
from math import exp, log
from util import Counter

import itertools
import re

############################################################
# Feature extractors: a feature extractor should take a raw input x (tuple of
# tokens) and add features to the featureVector (Counter) provided.

def basicFeatureExtractor(x):
  url, title = x
  featureVector = util.Counter()

  # For each token in the URL, add an indicator feature
  for token in url.split("/"):
    featureVector['url:' + token] += 1

  return featureVector

URL_PATTERN = re.compile(
  r"""
  \w{3,5}:\/\/ # domain
  (?P<host>[\w\-\.]+) # host
  (?P<path>(\/[\w\-]*)*) # path
  """, re.X)
FULLNAME_PATTERN = re.compile(
  r"""
  [\w\-]+\s+ # firstname
  \w\.\s+ # middle initial
  [\w\-]+\s* # lastname
  """, re.X)
def isNotSpace(s):
  return len(s.strip()) > 0

def customFeatureExtractor(x):
  url, title = x
  featureVector = util.Counter()

  # url
  # ---
  m = URL_PATTERN.match(url)

  if m != None:
    for subdomain in filter(isNotSpace, m.group('host').split('.')):
      featureVector['url:' + subdomain] += 1

    for subpath in filter(isNotSpace, m.group('path').split('/')):
      featureVector['url:' + subpath] += 1

  for token in url.split('/'):
    featureVector['url:' + token] += 1

  if '~' in url:
    featureVector['url:tilde'] += 1
  if ':' in url:
    featureVector['url:colon'] += 1

  # title
  # -----
  for word in filter(isNotSpace, re.split(r"\s+", title)):
    featureVector['title:' + word] += 1

  if FULLNAME_PATTERN.match(title):
    featureVector['title:fullname'] += 1

  return featureVector

############################################################
# You should implement the logistic, hinge, and squared loss.
# Each function takes a featureVector phi(x), output, y, weights, w and returns
# either the value of the loss at that point or the gradient of the loss at
# that point.

"""
The logistic loss, for a given weight vector.
@param featureVector: The featurized representation of a training example
@param y: The true value of the example (in our case, +/- 3)
@param weights: The weight vector assigning a weight to every feature
@return The scalar value of the logistic loss.
"""
def logisticLoss(featureVector, y, weights):
  "*** YOUR CODE HERE (around 2 lines of code expected) ***"
  dotp = 0.0
  for fKey in featureVector.iterkeys():
    dotp += featureVector[fKey] * weights[fKey]
  return log( 1 + exp(-dotp * y) )

"""
The gradient of the logistic loss with respect to the weight vector.
@param featureVector: The featurized representation of a training example
@param y: The true value of the example (in our case, +/- 1)
@param weights: The weight vector assigning a weight to every feature
@return The gradient [vector] of the logistic loss, with respect to w,
        the weights we are learning.
"""
def logisticLossGradient(featureVector, y, weights):
  "*** YOUR CODE HERE (around 3 lines of code expected) ***"
  dotp = 0.0
  for fKey in featureVector.iterkeys():
    dotp += featureVector[fKey] * weights[fKey]
  grad = Counter()
  for fKey in featureVector.iterkeys():
    grad[fKey] = -featureVector[fKey] * y / (1 + exp(dotp * y))
  return grad

"""
The hinge loss, for a given weight vector.
@param featureVector: The featurized representation of a training example
@param y: The true value of the example (in our case, +/- 1)
@param weights: The weight vector assigning a weight to every feature
@return The scalar value of the hinge loss.
"""
def hingeLoss(featureVector, y, weights):
  "*** YOUR CODE HERE (around 2 lines of code expected) ***"
  dotp = 0.0
  for fKey in featureVector.iterkeys():
    dotp += featureVector[fKey] * weights[fKey]
  return max(1 - dotp * y, 0.0)

"""
The gradient of the hinge loss with respect to the weight vector.
@param featureVector: The featurized representation of a training example
@param y: The true value of the example (in our case, +/- 1)
@param weights: The weight vector assigning a weight to every feature
@return The gradient [vector] of the hinge loss, with respect to w,
        the weights we are learning.
        You should not worry about the case when the hinge loss is exactly 1
"""
def hingeLossGradient(featureVector, y, weights):
  "*** YOUR CODE HERE (around 3 lines of code expected) ***"
  dotp = 0.0
  for fKey in featureVector.iterkeys():
    dotp += featureVector[fKey] * weights[fKey]
  grad = Counter()
  for fKey in featureVector.iterkeys():
    if dotp*y < 1:
      grad[fKey] = (-featureVector[fKey] * y)
    else:
      grad[fKey] = 0.0
  return grad

"""
The squared loss, for a given weight vector.
@param featureVector: The featurized representation of a training example
@param y: The true value of the example (in our case, +/- 1)
@param weights: The weight vector assigning a weight to every feature
@return The scalar value of the squared loss.
"""
def squaredLoss(featureVector, y, weights):
  "*** YOUR CODE HERE (around 2 lines of code expected) ***"
  dotp = 0.0
  for fKey in featureVector.iterkeys():
    dotp += featureVector[fKey] * weights[fKey]
  return 0.5 * (dotp - y) ** 2

"""
The gradient of the squared loss with respect to the weight vector.
@param featureVector: The featurized representation of a training example
@param y: The true value of the example (in our case, +/- 1)
@param weights: The weight vector assigning a weight to every feature
@return The gradient [vector] of the squared loss, with respect to w,
        the weights we are learning.
"""
def squaredLossGradient(featureVector, y, weights):
  "*** YOUR CODE HERE (around 2 lines of code expected) ***"
  dotp = 0.0
  for fKey in featureVector.iterkeys():
    dotp += featureVector[fKey] * weights[fKey]
  grad = Counter()
  for fKey in featureVector.iterkeys():
    grad[fKey] = (dotp - y) * featureVector[fKey]
  return grad

class StochasticGradientLearner():
  def __init__(self, featureExtractor):
    self.featureExtractor = util.memoizeById(featureExtractor)

  """
  This function takes a list of training examples and performs stochastic
  gradient descent to learn weights.
  @param trainExamples: list of training examples (you should only use this to
                        update weights).
                        Each element of this list is a list whose first element
                        is the input, and the second element, and the second
                        element is the true label of the training example.
  @param validationExamples: list of validation examples (just to see how well
                             you're generalizing)
  @param loss: function that takes (x, y, weights) and returns a number
               representing the loss.
  @param lossGradient: function that takes (x, y, weights) and returns the
                       gradient vector as a counter.
                       Recall that this is a function of the featureVector,
                       the true label, and the current weights.
  @param options: various parameters of the algorithm
     * initStepSize: the initial step size
     * stepSizeReduction: the t-th update should have step size:
                          initStepSize / t^stepSizeReduction
     * numRounds: make this many passes over your training data
     * regularization: the 'lambda' term in L2 regularization
  @return No return value, but you should set self.weights to be a counter with
          the new weights, after learning has finished.
  """
  def learn(self, trainExamples, validationExamples, loss, lossGradient, options):
    self.weights = util.Counter()
    random.seed(42)

    # You should go over the training data numRounds times.
    # Each round, go through all the examples in some random order and update
    # the weights with respect to the gradient.
    for r in xrange(0, options.numRounds):
      random.shuffle(trainExamples)
      numUpdates = 0  # Should be incremented with each example and determines the step size.

      # Loop over the training examples and update the weights based on loss and regularization.
      # If your code runs slowly, try to explicitly write out the dot products
      # in the code here (e.g., "for key,value in counter: counter[key] += ---"
      # rather than "counter * other_vector")
      for x, y in trainExamples:
        numUpdates += 1
        "*** YOUR CODE HERE (around 7 lines of code expected) ***"
        featureVector = self.featureExtractor(x)
        lossGrad = lossGradient(featureVector, y, self.weights)
        stepSize = options.initStepSize / (numUpdates ** options.stepSizeReduction)
        if (options.regularization > 0):
          c = float(options.regularization) / len(trainExamples)
          for fKey in self.weights.iterkeys():
            if self.weights[fKey] > 0 or lossGrad[fKey] > 0:
              self.weights[fKey] -= stepSize * (lossGrad[fKey] + c * self.weights[fKey])
        else:
          for fKey, fLoss in lossGrad.iteritems():
            self.weights[fKey] -= stepSize * fLoss

      # Compute the objective function.
      # Here, we have split the objective function into two components:
      # the training loss, and the regularization penalty.
      # The objective function is the sum of these two values
      trainLoss = 0  # Training loss
      regularizationPenalty = 0  # L2 Regularization penalty
      "*** YOUR CODE HERE (around 5 lines of code expected) ***"
      for x, y in trainExamples:
        featureVector = self.featureExtractor(x)
        trainLoss += loss(featureVector, y, self.weights)
      for weight in self.weights.itervalues():
        if weight > 0:
          regularizationPenalty += weight ** 2
      regularizationPenalty *= 0.5 * options.regularization
      self.objective = trainLoss + regularizationPenalty

      # See how well we're doing on our actual goal (error rate).
      trainError = util.getClassificationErrorRate(trainExamples, self.predict, 'train', options.verbose, self.featureExtractor, self.weights)
      validationError = util.getClassificationErrorRate(validationExamples, self.predict, 'validation', options.verbose, self.featureExtractor, self.weights)

      print "Round %s/%s: objective = %.2f = %.2f + %.2f, train error = %.4f, validation error = %.4f" % (r+1, options.numRounds, self.objective, trainLoss, regularizationPenalty, trainError, validationError)

    # Print out feature weights
    out = open('weights', 'w')
    for f, v in sorted(self.weights.items(), key=lambda x: -x[1]):
      print >> out, f + "\t" + str(v)
    out.close()

  """
  Classify a new input into either +1 or -1 based on the current weights
  (self.weights). Note that this function should be agnostic to the loss
  you are using for training.
  You may find the following fields useful:
    self.weights: Your current weights
    self.featureExtractor(): A function which takes a datum as input and
                             returns a featurized version of the datum.
  @param x An input example, not yet featurized.
  @return +1 or -1
  """
  def predict(self, x):
    "*** YOUR CODE HERE (around 3 lines of code expected) ***"
    featureVector = self.featureExtractor(x)
    pred = 0.0
    for fKey in featureVector.iterkeys():
      pred += featureVector[fKey] * self.weights[fKey]
    return 1 if pred >= 0 else -1

# After you have tuned your parameters, set the hyperparameter options:
# featureExtractor, loss, initStepSize, stepSizeReduction, numRounds, regularization, etc.
# The autograder will call this function before calling learn().
def setTunedOptions(options):
  "*** YOUR CODE HERE (around 6 lines of code expected) ***"

if __name__ == '__main__':
  util.runLearner(sys.modules[__name__], sys.argv[1:])
