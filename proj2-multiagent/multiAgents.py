# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util


from game import Agent

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newGhostPos = successorGameState.getGhostPositions()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #Compute the distance to nearest food
        foodDist = float('inf')
        for f in newFood:
            if manhattanDistance(newPos, f) < foodDist:
                foodDist = manhattanDistance(newPos, f)
        #Avoid ghosts
        for g in newGhostPos:
            if (manhattanDistance(newPos, g) < 1.5):
                return -float('inf')

        return successorGameState.getScore() + 1.0/foodDist


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

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        action, score = self.miniMax(gameState, 0, 0)
        return action
        util.raiseNotDefined()

    def miniMax(self, gameState, agentInd, depth):
        if agentInd == gameState.getNumAgents():
            agentInd = 0
            depth += 1
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return (None, self.evaluationFunction(gameState))
        if agentInd == 0:
            return self.maxValue(gameState, agentInd, depth)
        else:
            return self.minValue(gameState, agentInd, depth)
    
    def maxValue(self, gameState, agentInd, depth):
        v = (None, -float('inf'))
        for action in gameState.getLegalActions(agentInd):
            _, score = self.miniMax(gameState.generateSuccessor(agentInd, action), agentInd + 1, depth)
            if score > v[1]:
                v = (action, score)
        return v

    def minValue(self, gameState, agentInd, depth):
        v = (None, float('inf'))
        for action in gameState.getLegalActions(agentInd):
            _, score = self.miniMax(gameState.generateSuccessor(agentInd, action), agentInd + 1, depth)
            if score < v[1]:
                v = (action, score)
        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        action, score = self.alphaBeta(gameState, 0, 0, -float('inf'), float('inf'))
        return action
        util.raiseNotDefined()

    def alphaBeta(self, gameState, agentInd, depth, alpha, beta):
        if agentInd == gameState.getNumAgents():
            agentInd = 0
            depth += 1
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return (None, self.evaluationFunction(gameState))
        if agentInd == 0:
            return self.maxValue(gameState, agentInd, depth, alpha, beta)
        else:
            return self.minValue(gameState, agentInd, depth, alpha, beta)
    
    def maxValue(self, gameState, agentInd, depth, alpha, beta):
        v = (None, -float('inf'))
        for action in gameState.getLegalActions(agentInd):
            _, score = self.alphaBeta(gameState.generateSuccessor(agentInd, action), agentInd + 1, depth, alpha, beta)
            if score > v[1]:
                v = (action, score)
            if v[1] > beta:
                return v
            alpha = max(alpha,v[1])
        return v

    def minValue(self, gameState, agentInd, depth, alpha, beta):
        v = (None, float('inf'))
        for action in gameState.getLegalActions(agentInd):
            _, score = self.alphaBeta(gameState.generateSuccessor(agentInd, action), agentInd + 1, depth, alpha, beta)
            if score < v[1]:
                v = (action, score)
            if v[1] < alpha:
                return v
            beta = min(beta,v[1])
        return v


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
        "*** YOUR CODE HERE ***"
        action, score = self.Expectimax(gameState, 0, 0)
        return action
        util.raiseNotDefined()

    def Expectimax(self, gameState, agentInd, depth):
        if agentInd == gameState.getNumAgents():
            agentInd = 0
            depth += 1
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return (None, self.evaluationFunction(gameState))
        if agentInd == 0:
            return self.maxValue(gameState, agentInd, depth)
        else:
            return self.expValue(gameState, agentInd, depth)
    
    def maxValue(self, gameState, agentInd, depth):
        v = (None, -float('inf'))
        for action in gameState.getLegalActions(agentInd):
            _, score = self.Expectimax(gameState.generateSuccessor(agentInd, action), agentInd + 1, depth)
            if score > v[1]:
                v = (action, score)
        return v

    def expValue(self, gameState, agentInd, depth):
        v = 0
        legalActions = gameState.getLegalActions(agentInd)
        for action in legalActions:
            _, score = self.Expectimax(gameState.generateSuccessor(agentInd, action), agentInd + 1, depth)
            v += score / len(legalActions)
        return (None, v)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: I make use of four factors to improve the evaluation function, which are:
      foodNumFactor: related to the number of food remaining
      foodDistFactor: related to the distance to the closest food
      capsDistFactor: related to the distance to the closest capsules
      ghostDistFactor: related to the distance to the closest ghost
    I use these four factors and four corressponding coefficients to compute the new evaluation
    function.
    """
    "*** YOUR CODE HERE ***"
    evaluation = currentGameState.getScore()
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostPos = currentGameState.getGhostPositions()
    newCaps = currentGameState.getCapsules()
    #number of food remaining
    foodNum = currentGameState.getNumFood()
    #diatance to the nearest food
    foodDist = float('inf')
    for f in newFood:
        foodDist = min(foodDist, manhattanDistance(newPos, f))
    #distance to the nearest ghost
    ghostDist = float('inf')
    for g in newGhostPos:
        ghostDist = min(ghostDist, manhattanDistance(newPos, g))
    #distance to the nearest capsules
    capsDist = float('inf')
    for c in newCaps:
        capsDist = min(capsDist, manhattanDistance(newPos, c))

    #compute the evaluation factors
    foodNumFactor = 1/(foodNum + 1)
    foodDistFactor = 1/(foodDist + 1)
    if ghostDist <= 1:
        ghostDistFactor = -float('inf')
    else:
        ghostDistFactor = ghostDist
    ghostDistFactor = -1/(ghostDistFactor + 1)
    capsDistFactor = 1/(capsDist + 1)

    #compute the evaluation function
    evaluation = evaluation + 1.2*foodNumFactor + 1.5*foodDistFactor + 3*capsDistFactor + 1.5*ghostDistFactor
    
    return evaluation
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
