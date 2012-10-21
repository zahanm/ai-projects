# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util, math, itertools

from game import Agent

##############################
# Reflex Agent
##############################

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFoodGrid = currentGameState.getFood()
    oldCapsules = currentGameState.getCapsules()
    oldGhostStates = currentGameState.getGhostStates()
    newGhostStates = successorGameState.getGhostStates()

    """
    Features to consider:
    * score = score after moving
    * df = distance to nearest food
    * dg[] = distances to each ghost
    * dc = distance to nearest capsule

    Weighted combination for final evaluation
    score = w[0] * score + w[1] * -df + w[2] * sum(dg) + w[3] * -dc
    """

    w = [ 1.0, 5.0, 50.0, 100.0 ]

    score = 0

    score += w[0] * successorGameState.getScore()

    oldFood = betterGridToList(oldFoodGrid)
    if len(oldFood) > 0:
      score += w[1] * -distanceToNearest(newPos, oldFood)

    for oldState, newState in itertools.izip(oldGhostStates, newGhostStates):
      if oldState.scaredTimer > 0:
        score += w[2] * 20.0 / max(distance(newPos, oldState.getPosition()), 0.01)
      elif newState.getPosition() == newPos:
        return float("-inf")
      else:
        score += w[2] * -1.0 / max(distance(newPos, newState.getPosition()), 0.01)

    if len(oldCapsules) > 0:
      score += w[3] * 1.0 / max(distanceToNearest(newPos, oldCapsules), 0.01)

    return score

##############################
# Multi-Agent Searchers
##############################

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

##############################
# Minimax Agent
##############################

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    def minimax(state, agent, depth):
      if state.isWin() or state.isLose():
        return (0, Directions.STOP)
      if agent == state.getNumAgents():
        agent = 0
        depth -= 1
        if depth <= 0:
          return (0, Directions.STOP)
      currentScore = self.evaluationFunction(state)
      if agent == 0:
        # pacman's move
        v, maxaction = float("-inf"), Directions.STOP
        for action in state.getLegalActions(agent):
          nextState = state.generateSuccessor(agent, action)
          reward = self.evaluationFunction(nextState) - currentScore
          mnm = minimax(nextState, agent + 1, depth)
          if (reward + mnm[0]) > v:
            v = reward + mnm[0]
            maxaction = action
        return (v, maxaction)
      else:
        # it's a ghost's move
        v, minaction = float("inf"), Directions.STOP
        for action in state.getLegalActions(agent):
          nextState = state.generateSuccessor(agent, action)
          reward = self.evaluationFunction(nextState) - currentScore
          mnm = minimax(nextState, agent + 1, depth)
          if (reward + mnm[0]) < v:
            v = reward + mnm[0]
            minaction = action
        return (v, minaction)
    v, maxaction = minimax(gameState, 0, self.depth)
    return maxaction

##############################
# Alpha-Beta Agent
##############################

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    def minimaxAlphaBeta(state, agent, depth, alpha, beta):
      if state.isWin() or state.isLose():
        return (0, Directions.STOP)
      if agent == state.getNumAgents():
        agent = 0
        depth -= 1
        if depth <= 0:
          return (0, Directions.STOP)
      currentScore = self.evaluationFunction(state)
      if agent == 0:
        # pacman's move
        v, maxaction = float("-inf"), Directions.STOP
        for action in state.getLegalActions(agent):
          nextState = state.generateSuccessor(agent, action)
          reward = self.evaluationFunction(nextState) - currentScore
          mnm = minimaxAlphaBeta(nextState, agent + 1, depth, alpha, beta)
          if (reward + mnm[0]) > v:
            v = reward + mnm[0]
            maxaction = action
          if v >= beta:
            return (v, maxaction)
          if v > alpha:
            alpha = v
        return (v, maxaction)
      else:
        # it's a ghost's move
        v, minaction = float("inf"), Directions.STOP
        for action in state.getLegalActions(agent):
          nextState = state.generateSuccessor(agent, action)
          reward = self.evaluationFunction(nextState) - currentScore
          mnm = minimaxAlphaBeta(nextState, agent + 1, depth, alpha, beta)
          if (reward + mnm[0]) < v:
            v = reward + mnm[0]
            minaction = action
          if v <= alpha:
            return (v, minaction)
          if v < beta:
            beta = v
        return (v, minaction)
    v, maxaction = minimaxAlphaBeta(gameState, 0, self.depth,
      float("-inf"), float("inf"))
    return maxaction

##############################
# Expectimax Agent
##############################

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    def expectimax(state, agent, depth):
      if state.isWin() or state.isLose():
        return (0, Directions.STOP)
      if agent == state.getNumAgents():
        agent = 0
        depth -= 1
        if depth <= 0:
          return (0, Directions.STOP)
      currentScore = self.evaluationFunction(state)
      if agent == 0:
        # pacman's move
        v, maxaction = float("-inf"), Directions.STOP
        for action in state.getLegalActions(agent):
          nextState = state.generateSuccessor(agent, action)
          reward = self.evaluationFunction(nextState) - currentScore
          mnm = expectimax(nextState, agent + 1, depth)
          if (reward + mnm[0]) > v:
            v = reward + mnm[0]
            maxaction = action
        return (v, maxaction)
      else:
        # it's a ghost's move
        # taking expectation over uniform distribution = (1 / size) * value
        v = 0.0
        actions = state.getLegalActions(agent)
        for action in actions:
          nextState = state.generateSuccessor(agent, action)
          reward = self.evaluationFunction(nextState) - currentScore
          mnm = expectimax(nextState, agent + 1, depth)
          v += float(reward + mnm[0]) / len(actions)
        return (v, None)
    v, maxaction = expectimax(gameState, 0, self.depth)
    return maxaction

####################################
# State based Evalutation Function
####################################

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

##############################
# Helpers
# - graph traversal
##############################

def distance(origin, dest):
  """
  Uses simple manhatten distance for now
  Will use A* search in the future
  """
  return util.manhattanDistance(origin, dest)

def distanceToNearest(origin, dests):
  """
  Linear search algorithm for now
  Will use something smarter, A* again?
  """
  return min([ distance(origin, d) for d in dests ])

def betterGridToList(grid):
  l = []
  for i in xrange(grid.width):
    for j in xrange(grid.height):
      if grid[i][j]:
        l.append((i,j))
  return l
