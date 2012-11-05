# inference.py
# ------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import random
import busters
import game
import math
import itertools

# Constants
# ---------
MAX_DIST_DELTA = 7

class InferenceModule:
  """
  An inference module tracks a belief distribution over a ghost's location.
  This is an abstract class, which you should not modify.
  """

  ############################################
  # Useful methods for all inference modules #
  ############################################

  def __init__(self, ghostAgent):
    "Sets the ghost agent for later access"
    self.ghostAgent = ghostAgent
    self.index = ghostAgent.index

  def getPositionDistribution(self, gameState):
    """
    Returns a distribution over successor positions of the ghost from the given gameState.

    You must first place the ghost in the gameState, using setGhostPosition below.
    """
    ghostPosition = gameState.getGhostPosition(self.index) # The position you set
    actionDist = self.ghostAgent.getDistribution(gameState)
    dist = util.Counter()
    for action, prob in actionDist.items():
      successorPosition = game.Actions.getSuccessor(ghostPosition, action)
      dist[successorPosition] = prob
    return dist

  def setGhostPosition(self, gameState, ghostPosition):
    """
    Sets the position of the ghost for this inference module to the specified
    position in the supplied gameState.
    """
    conf = game.Configuration(ghostPosition, game.Directions.STOP)
    gameState.data.agentStates[self.index] = game.AgentState(conf, False)
    return gameState

  def observeState(self, gameState):
    "Collects the relevant noisy distance observation and pass it along."
    distances = gameState.getNoisyGhostDistances()
    if len(distances) >= self.index: # Check for missing observations
      obs = distances[self.index - 1]
      self.observe(obs, gameState)

  def initialize(self, gameState):
    "Initializes beliefs to a uniform distribution over all positions."
    # The legal positions do not include the ghost prison cells in the bottom left.
    self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    self.initializeUniformly(gameState)

  ######################################
  # Methods that need to be overridden #
  ######################################

  def initializeUniformly(self, gameState):
    "Sets the belief state to a uniform prior belief over all positions."
    pass

  def observe(self, observation, gameState):
    "Updates beliefs based on the given distance observation and gameState."
    pass

  def elapseTime(self, gameState):
    "Updates beliefs for a time step elapsing from a gameState."
    pass

  def getBeliefDistribution(self):
    """
    Returns the agent's current belief state, a distribution over
    ghost locations conditioned on all evidence so far.
    """
    pass

class ExactInference(InferenceModule):
  """
  The exact dynamic inference module should use forward-algorithm
  updates to compute the exact belief function at each time step.
  """

  def initializeUniformly(self, gameState):
    "Begin with a uniform distribution over ghost positions."
    self.beliefs = util.Counter()
    for p in self.legalPositions: self.beliefs[p] = 1.0
    self.beliefs.normalize()

  def observe(self, observation, gameState):
    """
    Updates beliefs based on the distance observation and Pacman's position.

    The noisyDistance is the estimated manhattan distance to the ghost you are tracking.

    The emissionModel below stores the probability of the noisyDistance for any true
    distance you supply.  That is, it stores P(noisyDistance | TrueDistance).

    self.legalPositions is a list of the possible ghost positions (you
    should only consider positions that are in self.legalPositions).
    """
    noisyDistance = observation
    emissionModel = busters.getObservationDistribution(noisyDistance)
    pacmanPosition = gameState.getPacmanPosition()

    """
    calculate normalizing factor for p(trueDistance)
    Int[ exp(-|x|), x from -d to d ]
    = Int[ exp(x), x from -d to 0 ] + Int[ exp(-x), x from 0 to d ]
    = exp(0) - exp(-d) + exp(0) - exp(-d)
    = 2 * ( 1 - exp(-d) )
    """
    # pTrueNorm = 2 * (1 - math.exp(-MAX_DIST_DELTA))
    # above was having numerical errors
    # pTrueNorm = sum(map(lambda x: math.exp(-abs(x)), range(-7, 8)))

    for pos in self.beliefs:
      trueDistance = util.manhattanDistance(pos, pacmanPosition)
      """
      use bayes rule to get
      P( true | noisy ) = P( noisy | true ) * P( true ) / P( noisy )
      We take care of the denominator by normalizing afterwords
      """
      if emissionModel[trueDistance] > 0 and self.beliefs[pos] > 0:
        # no need to normalize since it's by a constant
        pTrue = math.exp( -abs(trueDistance - noisyDistance) )
        self.beliefs[pos] = self.beliefs[pos] * emissionModel[trueDistance] * pTrue
      else:
        self.beliefs[pos] = 0

    self.beliefs.normalize()

  def elapseTime(self, gameState):
    """
    Update self.beliefs in response to a time step passing from the current state.

    The transition model is not entirely stationary: it may depend on Pacman's
    current position (e.g., for DirectionalGhost).  However, this is not a problem,
    as Pacman's current position is known.

    In order to obtain the distribution over new positions for the
    ghost, given its previous position (oldPos) as well as Pacman's
    current position, use this line of code:

      newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))

    Note that you may need to replace "oldPos" with the correct name
    of the variable that you have used to refer to the previous ghost
    position for which you are computing this distribution.

    newPosDist is a util.Counter object, where for each position p in self.legalPositions,

    newPostDist[p] = Pr( ghost is at position p at time t + 1 | ghost is at position oldPos at time t )

    (and also given Pacman's current position).  You may also find it useful to loop over key, value pairs
    in newPosDist, like:

      for newPos, prob in newPosDist.items():
        ...

    As an implementation detail (with which you need not concern
    yourself), the line of code above for obtaining newPosDist makes
    use of two helper methods provided in InferenceModule above:

      1) self.setGhostPosition(gameState, ghostPosition)
          This method alters the gameState by placing the ghost we're tracking
          in a particular position.  This altered gameState can be used to query
          what the ghost would do in this position.

      2) self.getPositionDistribution(gameState)
          This method uses the ghost agent to determine what positions the ghost
          will move to from the provided gameState.  The ghost must be placed
          in the gameState with a call to self.setGhostPosition above.
    """

    oldBeliefs = self.beliefs
    self.beliefs = util.Counter()

    for oldPos, oldProb in oldBeliefs.iteritems():
      if oldProb > 0:
        distribution = \
          self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))
        # there isn't a multiplyAll() function
        distribution.divideAll(1.0 / oldProb)
        self.beliefs += distribution

    self.beliefs.normalize()

  def getBeliefDistribution(self):
    return self.beliefs

class ParticleFilter(InferenceModule):
  """
  A particle filter for approximately tracking a single ghost.

  Useful helper functions will include random.choice, which chooses
  an element from a list uniformly at random, and util.sample, which
  samples a key from a Counter by treating its values as probabilities.
  """

  def initializeUniformly(self, gameState, numParticles=300):
    "Initializes a list of particles."
    self.numParticles = numParticles
    # really using it as a Integer Count for particles
    self.particles = util.Counter()
    for i in xrange(self.numParticles):
      self.particles[ random.choice(self.legalPositions) ] += 1
    # proposal distribution
    unifProposal = CounterFromIterable(self.legalPositions)
    unifProposal.normalize()
    self.proposals = None
    self.sampledCounts = None

  def observe(self, observation, gameState):
    """
    Update beliefs based on the given distance observation.

    Want to model P(noisy|true)
    """
    noisyDistance = observation
    emissionModel = busters.getObservationDistribution(noisyDistance)
    pacmanPos = gameState.getPacmanPosition()

    weighted = util.Counter()

    # check if jailed
    if noisyDistance == 999:
      jailLoc = (2 * (self.index - 1) + 1, 1)
      weighted[jailLoc] += 1
    else:
      for oldPos, counts in self.sampledCounts.iteritems():
        for pos in counts:
          trueDistance = util.manhattanDistance(pacmanPos, pos)
          delta = abs(trueDistance - noisyDistance)
          if emissionModel[trueDistance] > 0 and delta <= MAX_DIST_DELTA:
            # no need to normalize by constant
            pTrue = math.exp( -delta )
            weighted[pos] += \
              counts[pos] * emissionModel[trueDistance] * pTrue / self.proposals[oldPos][pos]

    weighted.normalize()

    if len(weighted) == 0:
      # reinitialize probs
      self.particles = util.Counter()
      for i in xrange(self.numParticles):
        self.particles[ random.choice(self.legalPositions) ] += 1
    else:
      # resample particles with replacement
      self.particles = util.Counter()
      for n in xrange(self.numParticles):
        p = util.sample(weighted)
        self.particles[p] += 1

  def elapseTime(self, gameState):
    """
    Update beliefs for a time step elapsing.

    As in the elapseTime method of ExactInference, you should use:

      newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))

    to obtain the distribution over new positions for the ghost, given
    its previous position (oldPos) as well as Pacman's current
    position.

    Want to model P(T2|T1)
    """

    self.sampledCounts = {}
    self.proposals = {}

    for oldPos, oldNumParticles in self.particles.iteritems():
      if oldNumParticles > 0:
        pretendState = self.setGhostPosition(gameState, oldPos)
        # dist[p] = Pr( ghost is at position p at time t + 1 | ghost is at
        #   position oldPos at time t )
        posDist = self.getPositionDistribution(pretendState)
        self.proposals[oldPos] = posDist
        self.sampledCounts[oldPos] = nSampleCounterWR(posDist, oldNumParticles)

  def getBeliefDistribution(self):
    """
    Return the agent's current belief state, a distribution over
    ghost locations conditioned on all evidence and time passage.
    """
    return util.normalize(self.particles)

class MarginalInference(InferenceModule):
  "A wrapper around the JointInference module that returns marginal beliefs about ghosts."

  def initializeUniformly(self, gameState):
    "Set the belief state to an initial, prior value."
    if self.index == 1: jointInference.initialize(gameState, self.legalPositions)
    jointInference.addGhostAgent(self.ghostAgent)

  def observeState(self, gameState):
    "Update beliefs based on the given distance observation and gameState."
    if self.index == 1: jointInference.observeState(gameState)

  def elapseTime(self, gameState):
    "Update beliefs for a time step elapsing from a gameState."
    if self.index == 1: jointInference.elapseTime(gameState)

  def getBeliefDistribution(self):
    "Returns the marginal belief over a particular ghost by summing out the others."
    jointDistribution = jointInference.getBeliefDistribution()
    dist = util.Counter()
    for t, prob in jointDistribution.items():
      dist[t[self.index - 1]] += prob
    return dist

class JointParticleFilter:
  "JointParticleFilter tracks a joint distribution over tuples of all ghost positions."

  def initialize(self, gameState, legalPositions, numParticles = 600):
    "Stores information about the game, then initializes particles."
    self.numGhosts = gameState.getNumAgents() - 1
    self.numParticles = numParticles
    self.ghostAgents = []
    self.legalPositions = legalPositions
    self.initializeParticles()

  def initializeParticles(self):
    "Initializes particles randomly.  Each particle is a tuple of ghost positions."
    self.particles = util.Counter()
    for i in xrange(self.numParticles):
      pos = \
        [ random.choice(self.legalPositions) for ghost in xrange(self.numGhosts) ]
      self.particles[tuple(pos)] += 1
    self.sampledCounts = None
    self.proposals = None

  def addGhostAgent(self, agent):
    "Each ghost agent is registered separately and stored (in case they are different)."
    self.ghostAgents.append(agent)

  def elapseTime(self, gameState):
    """
    Samples each particle's next state based on its current state and the gameState.

    To loop over the ghosts, use:

      for i in range(self.numGhosts):
        ...

    Then, assuming that "i" refers to the (0-based) index of the
    ghost, to obtain the distributions over new positions for that
    single ghost, given the list (prevGhostPositions) of previous
    positions of ALL of the ghosts, use this line of code:

      newPosDist = getPositionDistributionForGhost(setGhostPositions(gameState, prevGhostPositions),
                                                   i + 1, self.ghostAgents[i])

    Note that you may need to replace "prevGhostPositions" with the
    correct name of the variable that you have used to refer to the
    list of the previous positions of all of the ghosts, and you may
    need to replace "i" with the variable you have used to refer to
    the index of the ghost for which you are computing the new
    position distribution.

    As an implementation detail (with which you need not concern
    yourself), the line of code above for obtaining newPosDist makes
    use of two helper functions defined below in this file:

      1) setGhostPositions(gameState, ghostPositions)
          This method alters the gameState by placing the ghosts in the supplied positions.

      2) getPositionDistributionForGhost(gameState, ghostIndex, agent)
          This method uses the supplied ghost agent to determine what positions
          a ghost (ghostIndex) controlled by a particular agent (ghostAgent)
          will move to in the supplied gameState.  All ghosts
          must first be placed in the gameState using setGhostPositions above.
          Remember: ghosts start at index 1 (Pacman is agent 0).

          The ghost agent you are meant to supply is self.ghostAgents[ghostIndex-1],
          but in this project all ghost agents are always the same.
    """
    self.proposals = [ util.Counter() ] * self.numGhosts
    self.sampledCounts = {}

    for oldAssign, oldNumParticles in self.particles.iteritems():
      if oldNumParticles <= 0:
        continue
      sampleParticles = [ tuple() ] * oldNumParticles
      for g in xrange(self.numGhosts):
        pretendState = setGhostPositions(gameState, oldAssign)
        dist = getPositionDistributionForGhost(pretendState, g + 1, self.ghostAgents[g])
        self.proposals[g][oldAssign[g]] = dist
        samples = nSampleCounterWR(dist, oldNumParticles, aslist=True)
        for i in xrange(len(samples)):
          sampleParticles[i] = sampleParticles[i] + (samples[i], )
      self.sampledCounts[oldAssign] = CounterFromIterable(sampleParticles)

    self.particles = util.Counter()
    for k, v in self.sampledCounts.iteritems():
      self.particles = self.particles + v

  def observeState(self, gameState):
    """
    Resamples the set of particles using the likelihood of the noisy observations.

    As in elapseTime, to loop over the ghosts, use:

      for i in range(self.numGhosts):
        ...

    A correct implementation will handle two special cases:
      1) When a ghost is captured by Pacman, all particles should be updated so
         that the ghost appears in its prison cell, position (2 * i + 1, 1),
         where "i" is the 0-based index of the ghost.

         You can check if a ghost has been captured by Pacman by
         checking if it has a noisyDistance of 999 (a noisy distance
         of 999 will be returned if, and only if, the ghost is
         captured).

      2) When all particles receive 0 weight, they should be recreated from the
          prior distribution by calling initializeParticles.
    """

    pacmanPos = gameState.getPacmanPosition()
    noisyDistances = gameState.getNoisyGhostDistances()
    if len(noisyDistances) < self.numGhosts: return
    emissionModels = [busters.getObservationDistribution(dist) for dist in noisyDistances]

    weighted = util.Counter()
    jailed = [ noisy == 999 for noisy in noisyDistances ]

    for oldAssign, counts in self.sampledCounts.iteritems():
      for assign, oldCount in counts.iteritems():
        if oldCount <= 0:
          continue
        emissions = [ 1.0 ] * self.numGhosts
        proposals = [ 1.0 ] * self.numGhosts
        for g in xrange(self.numGhosts):
          if jailed[g]:
            # if jailed
            jailLocation = (2 * g + 1, 1)
            assign = assign[:g] + (jailLocation,) + assign[g+1:]
            emissions[g] = 1.0
            proposals[g] = 1.0
            continue
          # roaming free
          trueDistance = util.manhattanDistance(pacmanPos, assign[g])
          delta = abs(trueDistance - noisyDistances[g])
          if emissionModels[g][trueDistance] > 0 and delta <= MAX_DIST_DELTA:
            # no need to normalize by constant
            pTrue = math.exp( -delta )
            emissions[g] = emissionModels[g][trueDistance] * pTrue
            proposals[g] = self.proposals[g][oldAssign[g]][assign[g]]
          else:
            emissions[g] = 0.0
            proposals[g] = 1.0
        weighted[assign] += oldCount * listProduct(emissions) / listProduct(proposals)
      weighted.normalize()

    if len(weighted) == 0:
      # reinitialize probs
      self.initializeParticles()
    else:
      # resample particles with replacement
      self.particles = util.Counter()
      for n in xrange(self.numParticles):
        p = util.sample(weighted)
        self.particles[p] += 1

  def getBeliefDistribution(self):
    return util.normalize(self.particles)

# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()

def getPositionDistributionForGhost(gameState, ghostIndex, agent):
  """
  Returns the distribution over positions for a ghost, using the supplied gameState.
  """
  ghostPosition = gameState.getGhostPosition(ghostIndex)
  actionDist = agent.getDistribution(gameState)
  dist = util.Counter()
  for action, prob in actionDist.items():
    successorPosition = game.Actions.getSuccessor(ghostPosition, action)
    dist[successorPosition] = prob
  return dist

def setGhostPositions(gameState, ghostPositions):
  "Sets the position of all ghosts to the values in ghostPositionTuple."
  for index, pos in enumerate(ghostPositions):
    conf = game.Configuration(pos, game.Directions.STOP)
    gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
  return gameState

"""
Find possible successor states for ghost positions assignment

"""
def possibleGhostMoves(assignment, numGhosts, legalPositions):
  possible = [ set() ] * numGhosts
  for g in xrange(numGhosts):
    for delta in xrange(MAX_DIST_DELTA + 1):
      for split in xrange(delta + 1):
        pos = (assignment[g][0] + split, assignment[g][1] + delta - split)
        if pos in legalPositions: possible[g].add(pos)
        pos = (assignment[g][0] - split, assignment[g][1] - delta + split)
        if pos in legalPositions: possible[g].add(pos)
  # returns the cartesian product of each possible ghost position
  return itertools.product(*possible)

# returns Counter or list after sampling n items from counts
def nSampleCounterWR(counts, n, aslist = False):
  if counts.totalCount() != 1:
    counts = util.normalize(counts)
  pairs = counts.items()
  keys = [ k for k,v in pairs ]
  values = [ v for k,v in pairs ]
  sampled = util.nSample(values, keys, n)
  if aslist:
    return sampled
  return CounterFromIterable(sampled)

# create a Counter from an Iterable
def CounterFromIterable(items):
  c = util.Counter()
  for i in items: c[i] += 1
  return c

def listProduct(l):
  r = 1
  for i in l:
    r *= i
  return r
