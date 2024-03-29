
import inspect, sys

# Return the error rate on examples when using predict.
# featureExtractor, if specified is used for debugging.
def getClassificationErrorRate(examples, predict, displayName=None, verbose=0, featureExtractor=None, weights=None):
  numMistakes = 0
  # For each example, make a prediction.
  for x, y in examples:
    predicted_y = predict(x)
    if y != predicted_y:
      if verbose > 0:
        featureVector = featureExtractor(x)
        margin = (featureVector * weights) * y
        print "%s error (true y = %s, predicted y = %s, margin = %s): x = %s" % (displayName, y, predicted_y, margin, x)
        for f, v, w in sorted([(f, v, weights[f]) for f, v in featureVector.items()], key = lambda fvw: fvw[1]*fvw[2]):
          print "  %-30s : %s * %.2f = %.2f" % (f, v, w, v * w)
      numMistakes += 1
  return 1.0 * numMistakes / len(examples)

def readExamples(path):
  # path is a CSV file, each line contains label (+1 or -1), followed by a list of tokens.
  # Return list of examples; each example is a (x,y) pair.
  import csv
  examples = []
  for row in csv.reader(open(path)):
    x = tuple(row[1:])
    y = int(row[0])
    if not (y == -1 or y == +1):
      raise "Invalid output label (only binary classification supported): %s" % y
    examples.append((x, y))
  print "Read %d examples from %s" % (len(examples), path)
  return examples

def runLearner(module, args):
  # Parse command-line arguments
  from optparse import OptionParser
  parser = OptionParser()
  def default(str):
    return str + ' [Default: %default]'
  parser.add_option('-f', '--featureExtractor', dest='featureExtractor', type='string',
                    help=default('Which feature extractor to use (basic or custom)'), default="basic")
  parser.add_option('-l', '--loss', dest='loss', type='string',
                    help=default('Which loss function to use (logistic, hinge, or squared)'), default="logistic")
  parser.add_option('-i', '--initStepSize', dest='initStepSize', type='float',
                    help=default('the initial step size'), default=1)
  parser.add_option('-s', '--stepSizeReduction', dest='stepSizeReduction', type='float',
                    help=default('How much to reduce the step size [0, 1]'), default=0.5)
  parser.add_option('-R', '--numRounds', dest='numRounds', type='int',
                    help=default('Number of passes over the training data'), default=10)
  parser.add_option('-r', '--regularization', dest='regularization', type='float',
                    help=default('The lambda in L2 regularization'), default=0)
  parser.add_option('-d', '--dataset', dest='dataset', type='string',
                    help=default('Prefix of dataset to load (files are <prefix>.{train,validation}.csv)'), default='toy')
  parser.add_option('-v', '--verbose', dest='verbose', type='int',
                    help=default('Verbosity level'), default=0)
  parser.add_option('-u', '--setTunedOptions', dest='setTunedOptions',
                    help=default('Whether to used the tuned options'), default=False)
  options, extra_args = parser.parse_args(args)
  if len(extra_args) != 0:
    print "Ignoring extra arguments:", extra_args

  # Read data
  trainExamples = readExamples(options.dataset + '.train.csv')
  validationExamples = readExamples(options.dataset + '.validation.csv')

  if options.setTunedOptions:
    print "Using tuned options"
    module.setTunedOptions(options)

  # Set the loss
  loss = None
  lossGradient = None
  if options.loss == 'squared':
    loss, lossGradient = module.squaredLoss, module.squaredLossGradient
  elif options.loss == 'logistic':
    loss, lossGradient = module.logisticLoss, module.logisticLossGradient
  elif options.loss == 'hinge':
    loss, lossGradient = module.hingeLoss, module.hingeLossGradient
  else:
    raise "Unknown loss function: " + options.loss

  # Set the feature extractor
  featureExtractor = None
  if options.featureExtractor == 'basic':
    featureExtractor = module.basicFeatureExtractor
  elif options.featureExtractor == 'custom':
    featureExtractor = module.customFeatureExtractor
  else:
    raise "Unknown feature extractor: " + options.featureExtractor

  # Learn a model and evaluate
  learner = module.StochasticGradientLearner(featureExtractor)
  learner.learn(trainExamples, validationExamples, loss, lossGradient, options)
  return (learner, options)

############################################################

# Return a version of a single function f which memoizes based on ID.
def memoizeById(f):
  cache = {}
  def memf(x):
    i = id(x)
    if i not in cache:
      y = cache[i] = f(x)
    return cache[i]
  return memf

def raiseNotDefined():
  print "Method not implemented: %s" % inspect.stack()[1][3]    
  sys.exit(1)

class Counter(dict):
  """
  A counter keeps track of counts for a set of keys.
  
  The counter class is an extension of the standard python
  dictionary type.  It is specialized to have number values  
  (integers or floats), and includes a handful of additional
  functions to ease the task of counting data.  In particular, 
  all keys are defaulted to have value 0.  Using a dictionary:
  
  a = {}
  print a['test']
  
  would give an error, while the Counter class analogue:
    
  >>> a = Counter()
  >>> print a['test']
  0

  returns the default 0 value. Note that to reference a key 
  that you know is contained in the counter, 
  you can still use the dictionary syntax:
    
  >>> a = Counter()
  >>> a['test'] = 2
  >>> print a['test']
  2
  
  This is very useful for counting things without initializing their counts,
  see for example:
  
  >>> a['blah'] += 1
  >>> print a['blah']
  1
  
  The counter also includes additional functionality useful in implementing
  the classifiers for this assignment.  Two counters can be added,
  subtracted or multiplied together.  See below for details.  They can
  also be normalized and their total count and arg max can be extracted.
  """
  def __getitem__(self, idx):
    self.setdefault(idx, 0)
    return dict.__getitem__(self, idx)

  def incrementAll(self, keys, count):
    """
    Increments all elements of keys by the same count.
    
    >>> a = Counter()
    >>> a.incrementAll(['one','two', 'three'], 1)
    >>> a['one']
    1
    >>> a['two']
    1
    """
    for key in keys:
      self[key] += count
  
  def argMax(self):
    """
    Returns the key with the highest value.
    """
    if len(self.keys()) == 0: return None
    all = self.items()
    values = [x[1] for x in all]
    maxIndex = values.index(max(values))
    return all[maxIndex][0]
  
  def sortedKeys(self):
    """
    Returns a list of keys sorted by their values.  Keys
    with the highest values will appear first.
    
    >>> a = Counter()
    >>> a['first'] = -2
    >>> a['second'] = 4
    >>> a['third'] = 1
    >>> a.sortedKeys()
    ['second', 'third', 'first']
    """
    sortedItems = self.items()
    compare = lambda x, y:  sign(y[1] - x[1])
    sortedItems.sort(cmp=compare)
    return [x[0] for x in sortedItems]
  
  def totalCount(self):
    """
    Returns the sum of counts for all keys.
    """
    return sum(self.values())
  
  def normalize(self):
    """
    Edits the counter such that the total count of all
    keys sums to 1.  The ratio of counts for all keys
    will remain the same. Note that normalizing an empty 
    Counter will result in an error.
    """
    total = float(self.totalCount())
    if total == 0: return
    for key in self.keys():
      self[key] = self[key] / total
      
  def divideAll(self, divisor):
    """
    Divides all counts by divisor
    """
    divisor = float(divisor)
    for key in self:
      self[key] /= divisor

  def copy(self):
    """
    Returns a copy of the counter
    """
    return Counter(dict.copy(self))
  
  def __mul__(self, y):
    if not isinstance(y, Counter):
      # Return the scalar y multiplied by every element of result.
      result = Counter()
      for key in self:
        result[key] = y * self[key]
      return result

    """
    Multiplying two counters gives the dot product of their vectors where
    each unique label is a vector element.
    
    >>> a = Counter()
    >>> b = Counter()
    >>> a['first'] = -2
    >>> a['second'] = 4
    >>> b['first'] = 3
    >>> b['second'] = 5
    >>> a['third'] = 1.5
    >>> a['fourth'] = 2.5
    >>> a * b
    14
    """
    sum = 0
    x = self
    if len(x) > len(y):
      x,y = y,x
    for key in x:
      if key not in y:
        continue
      sum += x[key] * y[key]      
    return sum
      
  def __radd__(self, y):
    """
    Adding another counter to a counter increments the current counter
    by the values stored in the second counter.
    
    >>> a = Counter()
    >>> b = Counter()
    >>> a['first'] = -2
    >>> a['second'] = 4
    >>> b['first'] = 3
    >>> b['third'] = 1
    >>> a += b
    >>> a['first']
    1
    """ 
    for key, value in y.items():
      self[key] += value   
      
  def __add__( self, y ):
    """
    Adding two counters gives a counter with the union of all keys and
    counts of the second added to counts of the first.
    
    >>> a = Counter()
    >>> b = Counter()
    >>> a['first'] = -2
    >>> a['second'] = 4
    >>> b['first'] = 3
    >>> b['third'] = 1
    >>> (a + b)['first']
    1
    """
    addend = Counter()
    for key in self:
      if key in y:
        addend[key] = self[key] + y[key]
      else:
        addend[key] = self[key]
    for key in y:
      if key in self:
        continue
      addend[key] = y[key]
    return addend
    
  def __sub__( self, y ):
    """
    Subtracting a counter from another gives a counter with the union of all keys and
    counts of the second subtracted from counts of the first.
    
    >>> a = Counter()
    >>> b = Counter()
    >>> a['first'] = -2
    >>> a['second'] = 4
    >>> b['first'] = 3
    >>> b['third'] = 1
    >>> (a - b)['first']
    -5
    """      
    addend = Counter()
    for key in self:
      if key in y:
        addend[key] = self[key] - y[key]
      else:
        addend[key] = self[key]
    for key in y:
      if key in self:
        continue
      addend[key] = -1 * y[key]
    return addend
