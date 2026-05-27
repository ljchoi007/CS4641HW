# ============================================================
# HW4 Coding: Bayesian Networks + Metropolis–Hastings
#
# Students must implement the following three functions:
#   - marginal_log_probability(bn, assignment)
#   - conditional_log_probability(bn, target, evidence)
#   - metropolis_hastings(log_probability, transition, rng, T)
#
# The autograder will only call these three functions.
# ============================================================
import math
import numpy as np
import itertools
from sklearn.datasets import fetch_20newsgroups

def marginal_log_probability(bn, assignment) -> float:
    """
    Compute the marginal log probability of a (possibly partial) assignment
    in a Bayesian Network.

    Inputs
    ------
    bn : SimpleBayesNet
        Refer to student_report.ipynb for the definition of this BN object and its attributes.
        It has the following attributes:
        1.  bn.variables : list of strings
            Names of the random variables, e.g. ["A", "B", "C"].
        2.  bn.parents : dict from string to list of strings
            For each variable name X, bn.parents[X] is a list of the
            names of its parent variables (possibly an empty list).
            Example: {"A": [], "B": ["A"], "C": ["B"]}.
        3.  bn.domains : dict from string to list
            For each variable name X, bn.domains[X] is a list of all
            possible values that X can take.  The values can be 0/1,
            ints, or small strings.
            Example: {"A": [0, 1], "B": [0, 1], "C": [0, 1]}.
        4.  bn.cpts : nested dictionaries encoding the conditional
            probability tables (CPTs).

    assignment
        A Python dict mapping variable names (strings) to values.
        Example: {"A": 1, "C": 0}.
        This can specify only a subset of variables; the rest must be
        marginalized out (summed over all their possible values).

    Output
    ------
    log_p : float
        log P(X1 = x1, ..., Xk = xk)
        where (X1, ..., Xk) and values come from `assignment`.
        All other variables in bn.variables are marginalized out
        (summed over all their possible values).
    """
    #get unassigned variables
    remaining_rvs = bn.variables.copy()
    cartesian_domains = []
    for rv in bn.variables:
        if rv in assignment:
            cartesian_domains.append([assignment[rv]])
        else:
            cartesian_domains.append(bn.domains[rv])
    #cartesian over the assigned variables to get probabilities that i'll need to sum
    probabilities = list(itertools.product(*cartesian_domains))
    joint_probabilities = dict.fromkeys(probabilities, 1)
    #loop thru list to calculate
    while (len(remaining_rvs) > 0):
        remove_list = []
        for rv in remaining_rvs:
            #print(rv, bn.parents[rv])
            parents_present = True
            for parent in bn.parents[rv]:
                if (parent in remaining_rvs) and (not parent in remove_list):
                    parents_present = False
                    break
            if (parents_present == False):
                continue
            
            #parents are present, and we can continue
            #for unassigned vars, we need to multiply the likelihood in the tuple
            for combination in joint_probabilities:
                cur_prob = 1
                cond_list = []
                for parent in bn.parents[rv]:
                    var_index = bn.variables.index(parent)
                    cond_list.append(combination[var_index])
                cond_tuple = tuple(cond_list)
                rv_index = bn.variables.index(rv)
                joint_probabilities[combination] *= bn.cpts[rv][cond_tuple][combination[rv_index]]
                #print("rv:", rv, ", cond_tuple:", cond_tuple, ", cartesian:", combination, ",cur prob", bn.cpts[rv][cond_tuple][combination[rv_index]])
            remove_list.append(rv)
        for rv in remove_list:
            remaining_rvs.remove(rv)
    total_prob = 0
    for prob in joint_probabilities:
        total_prob += joint_probabilities[prob]
    #print(math.log(total_prob))
    return math.log(total_prob)

def conditional_log_probability(
    bn,
    target,
    evidence,
) -> float:
    """
    Compute log P(target | evidence) in a Bayesian Network, reusing
    marginal_log_probability.

    Inputs
    ------
    bn
        Same BN object / attribute structure as in marginal_log_probability.

    target
        A dict mapping variable names to values for the event A whose
        probability you care about, e.g. {"Rain": 1}.

    evidence
        A dict mapping variable names to values for the conditioning
        event B, e.g. {"WetGrass": 1}.

    Output
    ------
    log_p : float
        log P(target | evidence) = log P(target, evidence) - log P(evidence).
    """
    target_and_evidence = target.copy()
    target_and_evidence.update(evidence)
    p_both = marginal_log_probability(bn, target_and_evidence)
    p_evidence = marginal_log_probability(bn, evidence)
    return (p_both - p_evidence)


def metropolis_hastings(
    log_probability,
    transition,
    rng,
    T: int,
):
    """
    Run the Metropolis–Hastings Markov chain for T steps.

    Inputs
    ------
    log_probability : function
        Maps a state x to log P(x).
    transition : function
        Proposal function:
          - If called with state=None, returns an initial random state.
          - Otherwise, given current state x, returns a random proposed new state x'.
    rng : numpy.random.Generator
        Random number generator used for accept/reject decisions.
        You should use rng.uniform(0.0, 1.0) for draws in [0, 1).
    T : int
        Number of steps / samples to return.

    Output
    ------
    samples : list
        List of length T containing the states of the Markov chain
        (including the initial state at t=0).

    Algorithm
    ---------
      1. Initialize x_0 = transition(None) and compute log P(x_0).
      2. For t = 1, ..., T-1:
           - Propose x' = transition(x_{t-1}).
           - Compute log_alpha = log P(x') - log P(x_{t-1}).
           - Set alpha = min(1, exp(log_alpha)).
           - Draw u ~ Uniform(0, 1) using rng.uniform(0.0, 1.0).
           - If u < alpha, accept: x_t = x'; else reject: x_t = x_{t-1}.
      3. Return [x_0, x_1, ..., x_{T-1}].
    """
    x_list = []
    #step 1
    x_0 = transition(None)
    logx_0 = log_probability(x_0)
    x_list.append(x_0)
    prev_x = x_0

    for i in range(1, T):
        x_prime = transition(prev_x)
        log_alpha = log_probability(x_prime) - log_probability(prev_x)
        alpha = min(1, math.exp(log_alpha))
        rand_u = rng.uniform(0.0, 1.0)
        if (rand_u < alpha):
            #accept x_t = x'
            prev_x = x_prime
            x_list.append(x_prime)
        else:
            #reject (x_t = x_{t-1})
            x_list.append(prev_x)
    return x_list


class SimpleBayesNet:
    """Minimal Bayesian Network container used in this notebook."""

    def __init__(self, variables, parents, domains, cpts):
        self.variables = variables
        self.parents = parents
        self.domains = domains
        self.cpts = cpts
if __name__ == '__main__':
    # Load a small 2-class subset of 20 Newsgroups just for illustration
    categories = ["rec.sport.hockey", "rec.sport.baseball"]
    print("before")
    example_data = fetch_20newsgroups(
        subset="train",
        categories=categories,
        remove=("headers", "footers", "quotes"),
    )

    print("Categories:", example_data.target_names)
    print("Number of example documents:", len(example_data.data))
    print("\nSample documents (truncated):\n")

    for i in range(3):
        label = example_data.target_names[example_data.target[i]]
        text = example_data.data[i].strip().replace("\n", " ")
        print(f"Example {i+1} - Label: {label}")
        print(text[:400] + ("..." if len(text) > 400 else ""))
        print("-" * 80)