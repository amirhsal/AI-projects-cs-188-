# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()
        self.qValuesOfStateAction = {}

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for i in range(self.iterations):
            vVal = util.Counter()
            for state in self.mdp.getStates():
                possibleActions = self.mdp.getPossibleActions(state)
                if self.mdp.isTerminal(state) or len(possibleActions) == 0:
                    self.values[state] = 0
                else:
                    qVal = []
                    for action in possibleActions:
                        qVal.append(self.computeQValueFromValues(state, action))
                    vVal[state] = max(qVal)
            self.values = vVal
                #print("!!!!!!!!!!!!!!!! self.value: ",self.values )"""

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        qValue = 0
        for transition in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, transition[0])
            nextStateValueStar = self.values[transition[0]]
            qValue += (transition[1]) * (reward + (self.discount * nextStateValueStar))
        return qValue


        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        legalActions = self.mdp.getPossibleActions(state)
        if self.mdp.isTerminal(state) or len(legalActions) == 0:
            return None
        vVal = util.Counter()
        for action in legalActions:
            vVal[action] = self.computeQValueFromValues(state, action)
                #print("best Action $$$$$$$$$$$$$$$$: ", action)
        return vVal.argMax()

        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """


        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        index = 0
        stateList = self.mdp.getStates()
        numOfState = len(stateList)
        while index < self.iterations:
            state = stateList[index % numOfState]
            vVal = util.Counter()
            possibleActions = self.mdp.getPossibleActions(state)
            if self.mdp.isTerminal(state) or len(possibleActions) == 0:
                self.values[state] = 0
            else:
                qVal = []
                for action in possibleActions:
                    qVal.append(self.computeQValueFromValues(state, action))
                self.values[state] = max(qVal)
            index += 1


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # computing the highest Q-value of a state among its possible actions
        def maxQvalue(state):
            qVal = []
            if self.mdp.isTerminal(state):
                return 0
            for action in self.mdp.getPossibleActions(state):
                qVal.append(self.getQValue(state, action))
            return max(qVal)



        #computing the predecessors
        predecessors = {}
        stateList = self.mdp.getStates()
        for state in stateList:
            predecessors[state] = set()
        for state in stateList:
            for action in self.mdp.getPossibleActions(state):
                transition = self.mdp.getTransitionStatesAndProbs(state, action)
                for preState, prob in transition:
                    if prob != 0:
                        #print("IIIIIIIIII:", preState)
                        predecessors[preState].add(state)
                        #print("OOOOOOOOOOOO:", predecessors)

        #_______________________________________
        pq = util.PriorityQueue()
        if not self.mdp.isTerminal(state):
            for state in self.mdp.getStates():
                diff = abs(self.values[state] - maxQvalue(state))
                pq.push(state, -diff)
        for i in range(self.iterations):
            if pq.isEmpty():
                break
            s = pq.pop()
            self.values[s] = maxQvalue(s)
            #print("OOOOOOOOOOOO:", predecessors[s])
            for pred in predecessors[s]:
                diff = abs(self.values[pred] - maxQvalue(pred))
                if diff > self.theta:
                    pq.update(pred, -diff)
