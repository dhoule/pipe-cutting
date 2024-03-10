import copy
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd

from alns import ALNS
from alns.accept import HillClimbing
from alns.select import RouletteWheel
from alns.stop import MaxIterations

# %matplotlib inline

# Used to seed the random number generater
SEED = 5432
# The beams/pipes available
OPTIMAL_BEAMS = 74

# The first line lists the number of lines for beam/pipe orders.
# The second line is the length of the available beams/pipes. Each
# following line is an order of (length, amount) tuples.
with open('pipesNeeded.txt') as file:
    data = file.readlines()

NUM_LINES = int(data[0])
BEAM_LENGTH = int(data[1])


# Beams to be cut from the available beams
BEAMS = [float(length)
         for datum in data[2:]
         for length, amount in [datum.strip().split()]
         for _ in range(int(amount))]
   
print("Each available beam is of length:", BEAM_LENGTH)
print("Number of beams to be cut (orders):", len(BEAMS))

class CspState:
    """
    Solution state for the CSP problem. It has two data members, assignments
    and unassigned. Assignments is a list of lists, one for each beam in use.
    Each entry is another list, containing the ordered beams cut from this 
    beam. Each such sublist must sum to at most BEAM_LENGTH. Unassigned is a
    list of ordered beams that are not currently assigned to one of the
    available beams.
    """

    def __init__(self, assignments, unassigned=None):
        self.assignments = assignments
        self.unassigned = []
        
        if unassigned is not None:
            self.unassigned = unassigned

    def copy(self):
        """
        Helper method to ensure each solution state is immutable.
        """
        return CspState(copy.deepcopy(self.assignments),
                        self.unassigned.copy())

    def objective(self):
        """
        Computes the total number of beams in use.
        """
        return len(self.assignments)

    def plot(self):
        """
        Helper method to plot a solution.
        """
        _, ax = plt.subplots(figsize=(12, 6))
        
        ax.barh(np.arange(len(self.assignments)), 
                [sum(assignment) for assignment in self.assignments], 
                height=1)

        ax.set_xlim(right=BEAM_LENGTH)
        ax.set_yticks(np.arange(len(self.assignments), step=10))

        ax.margins(x=0, y=0)

        ax.set_xlabel('Usage')
        ax.set_ylabel('Beam (#)')

        # plt.draw_if_interactive()
        plt.show()

    # Returns the assigned cutting of each beam/pipe that is to be used
    def get_assignments(self):
      return self.assignments

def wastage(assignment):
    """
    Helper method that computes the wastage on a given beam assignment.
    """
    return BEAM_LENGTH - sum(assignment)

# The percentage of beams/pipes to destroy to find a solution
degree_of_destruction = 0.25

def beams_to_remove(num_beams):
    return float(num_beams * degree_of_destruction)

def random_removal(state, random_state):
    """
    Iteratively removes randomly chosen beam assignments.
    """
    state = state.copy()

    for _ in range(round(beams_to_remove(state.objective()))):
        idx = random_state.randint(state.objective())
        state.unassigned.extend(state.assignments.pop(idx))

    return state

def worst_removal(state, random_state):
    """
    Removes beams in decreasing order of wastage, such that the
    poorest assignments are removed first.
    """
    state = state.copy()

    # Sort assignments by wastage, worst first
    state.assignments.sort(key=wastage, reverse=True)

    # Removes the worst assignments
    for _ in range(round(beams_to_remove(state.objective()))):
        state.unassigned.extend(state.assignments.pop(0))

    return state

def greedy_insert(state, random_state):
    """
    Inserts the unassigned beams greedily into the first fitting
    beam. Shuffles the unassigned ordered beams before inserting.
    """
    random_state.shuffle(state.unassigned)

    while len(state.unassigned) != 0:
      beam = state.unassigned.pop(0)
      # the Else clause is always done after the end of he For loop
      for assignment in state.assignments:
        if beam <= wastage(assignment):
          assignment.append(beam)
          break
      else:
        state.assignments.append([beam])

    return state

def minimal_wastage(state, random_state):
    """
    For every unassigned ordered beam, the operator determines
    which beam would minimise that beam's waste once the ordered
    beam is inserted.
    """
    def insertion_cost(assignment, beam):  # helper method for min
        if beam <= wastage(assignment):
            return wastage(assignment) - beam

        return float("inf")

    while len(state.unassigned) != 0:
        beam = state.unassigned.pop(0)

        assignment = min(state.assignments,
                         key=partial(insertion_cost, beam=beam))

        if beam <= wastage(assignment):
            assignment.append(beam) 
        else:
            state.assignments.append([beam])

    return state

rnd_state = rnd.RandomState(SEED)

state = CspState([], BEAMS.copy())
init_sol = greedy_insert(state, rnd_state)

print("Initial solution has objective value:", init_sol.objective())

# init_sol.plot()

alns = ALNS(rnd_state)

alns.add_destroy_operator(random_removal)
alns.add_destroy_operator(worst_removal)

alns.add_repair_operator(greedy_insert)
alns.add_repair_operator(minimal_wastage)

accept = HillClimbing()
select = RouletteWheel([3, 2, 1, 0.5], 0.8, 2, 2)
stop = MaxIterations(5_000)

result = alns.iterate(init_sol, select, accept, stop)
solution = result.best_state
objective = solution.objective()

_, ax = plt.subplots(figsize=(12, 6))
result.plot_objectives(ax=ax)

figure = plt.figure("operator_counts", figsize=(12, 6))
figure.subplots_adjust(bottom=0.15, hspace=.5)
result.plot_operator_counts(figure, title="Operator diagnostics")

print("Heuristic solution has objective value:", solution.objective())

solution.plot()

obj = solution.objective()
print("Number of beams used is {0}, which is {1} more than the optimal value {2}."
      .format(obj, obj - OPTIMAL_BEAMS, OPTIMAL_BEAMS))
print("The cuts for each beam/pipe needed are as follows:")
# Print the cuts per beam/pipe
print(solution.get_assignments())
