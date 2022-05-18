import random

import numpy as np

from ..do_not_touch.mdp_env_wrapper import Env1
from ..do_not_touch.result_structures import ValueFunction, PolicyAndValueFunction

from ..to_do.line_world_mdp import *
from ..to_do.grid_world_mdp import *

np.set_printoptions(suppress=True)


def policy_evaluation_on_line_world() -> ValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    S, A, R, p, gamma, threshold, pi, V = reset_line_world()
    while True:
        delta = 0
        for s in S:
            v = np.copy(V[s])
            V[s] = 0
            for a in A:
                for s_p in S:
                    for r_idx, r in enumerate(R):
                        V[s] += pi[s, a] * p[s, a, s_p, r_idx] * (r + gamma * V[s_p])
            delta = max(delta, abs(v - V[s]))
        if delta < threshold:
            break
    return V


def policy_iteration_on_line_world() -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    S, A, R, p, gamma, threshold, pi, V = reset_line_world()
    while True:
        # Policy evaluation
        while True:
            delta = 0
            for s in S:
                v = np.copy(V[s])
                V[s] = 0
                for a in A:
                    for s_p in S:
                        for r_idx, r in enumerate(R):
                            V[s] += pi[s, a] * p[s, a, s_p, r_idx] * (r + gamma * V[s_p])
                delta = max(delta, abs(v - V[s]))

            if delta < threshold:
                break

        # Policy improvement
        policy_stable = True
        for s in S:
            old_policy = np.copy(pi[s, :])
            best_action = -9999
            num_best_a = -1
            for a in A:
                action_value = 0
                for s_p in S:
                    for r_idx, r in enumerate(R):
                        action_value += p[s, a, s_p, r_idx] * (r + gamma * V[s_p])
                if action_value > best_action:
                    best_action = action_value
                    num_best_a = a
            pi[s, :] = 0.0
            pi[s, num_best_a] = 1.0
            if not np.array_equal(old_policy, pi[s]):
                policy_stable = False
        if policy_stable:
            break
    return PolicyAndValueFunction(pi, V)


def value_iteration_on_line_world() -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    S, A, R, p, gamma, threshold, pi, V = reset_line_world()
    while True:
        delta = 0
        for s in S:
            v = V[s]
            max_a = -1
            for a in A:
                action_value = 0
                for s_p in S:
                    for r_idx, r in enumerate(R):
                        action_value += p[s, a, s_p, r_idx] * (r + gamma * V[s_p])
                        if action_value > V[s]:
                            V[s] = action_value
                            max_a = a
            pi[s, :] = 0.0
            pi[s, max_a] = 1.0
            delta = max(delta, np.abs(v - V[s]))
        if delta < threshold:
            break
    return PolicyAndValueFunction(pi, V)


def policy_evaluation_on_grid_world() -> ValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    S, A, R, p, gamma, threshold, pi, V = reset_grid_world()
    while True:
        delta = 0
        for s in S:
            v = np.copy(V[s])
            V[s] = 0
            for a in A:
                for s_p in S:
                    for r_idx, r in enumerate(R):
                        V[s] += pi[s, a] * p[s, a, s_p, r_idx] * (r + gamma * V[s_p])
            delta = max(delta, abs(v - V[s]))

        if delta < threshold:
            break
    return V


def policy_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    S, A, R, p, gamma, threshold, pi, V = reset_grid_world()
    while True:
        # Policy evaluation
        while True:
            delta = 0
            for s in S:
                v = np.copy(V[s])
                V[s] = 0
                for a in A:
                    for s_p in S:
                        for r_idx, r in enumerate(R):
                            V[s] += pi[s, a] * p[s, a, s_p, r_idx] * (r + gamma * V[s_p])
                delta = max(delta, abs(v - V[s]))

            if delta < threshold:
                break

        # Policy improvement
        policy_stable = True
        for s in S:
            old_policy = np.copy(pi[s, :])
            best_action = -9999
            num_best_a = -1
            for a in A:
                action_value = 0
                for s_p in S:
                    for r_idx, r in enumerate(R):
                        action_value += p[s, a, s_p, r_idx] * (r + gamma * V[s_p])
                if action_value > best_action:
                    best_action = action_value
                    num_best_a = a
            pi[s, :] = 0.0
            pi[s, num_best_a] = 1.0
            if not np.array_equal(old_policy, pi[s]):
                policy_stable = False
        if policy_stable:
            break
    return PolicyAndValueFunction(pi, V)


def value_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    S, A, R, p, gamma, threshold, pi, V = reset_grid_world()
    while True:
        delta = 0
        for s in S:
            v = V[s]
            max_a = -1
            for a in A:
                action_value = 0
                for s_p in S:
                    for r_idx, r in enumerate(R):
                        action_value += p[s, a, s_p, r_idx] * (r + gamma * V[s_p])
                        if action_value > V[s]:
                            V[s] = action_value
                            max_a = a
            print(a)
            pi[s, :] = 0.0
            pi[s, max_a] = 1.0
            delta = max(delta, np.abs(v - V[s]))
        if delta < threshold:
            break
    print(pi)
    exit()
    return PolicyAndValueFunction(pi, V)


def policy_evaluation_on_secret_env1() -> ValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    env = Env1()
    V = np.zeros((len(env.states(),)))
    pi = np.zeros((len(env.states()), len(env.actions())))
    pi[:] = 1/len(env.actions())
    while True:
        delta = 0
        for s in env.states():
            v = np.copy(V[s])
            V[s] = 0
            for a in env.actions():
                for s_p in env.states():
                    for r_idx, r in enumerate(env.rewards()):
                        V[s] += pi[s, a] * env.transition_probability(s, a, s_p, r_idx) * (r + 0.999999 * V[s_p])
            delta = max(delta, abs(v - V[s]))
        if delta < 0.000001:
            break
    return V


def policy_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()
    V = np.zeros(len(env.states()))
    pi = np.zeros((len(env.states()), len(env.actions())))
    pi[:] = 1 / len(env.actions())
    while True:
        # policy evaluation
        while True:
            delta = 0
            for s in env.states():
                v = np.copy(V[s])
                V[s] = 0
                for a in env.actions():
                    for s_p in env.states():
                        for r_idx, r in enumerate(env.rewards()):
                            V[s] += pi[s, a] * env.transition_probability(s, a, s_p, r_idx) * (r + 0.999999 * V[s_p])
                delta = max(delta, abs(v - V[s]))
            if delta < 0.000001:
                break

        # policy improvement
        policy_stable = True
        for s in env.states():
            old_policy = np.copy(pi[s, :])
            best_action = -9999
            num_best_a = -1
            for a in env.actions():
                action_value = 0
                for s_p in env.states():
                    for r_idx, r in enumerate(env.rewards()):
                        action_value += env.transition_probability(s, a, s_p, r_idx) * (r + 0.999999 * V[s_p])
                if action_value > best_action:
                    best_action = action_value
                    num_best_a = a
                pi[s, :] = 0.0
                pi[s, num_best_a] = 1.0
            if not np.array_equal(old_policy, pi[s]):
                policy_stable = False
        if policy_stable:
            break
    return PolicyAndValueFunction(pi, V)


def value_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Prints the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()
    V = np.zeros(len(env.states()))
    pi = np.zeros((len(env.states()), len(env.actions())))
    pi[:] = 1 / len(env.actions())
    while True:
        delta = 0
        for s in env.states():
            v = V[s]
            max_a = -1
            for a in env.actions():
                action_value = 0
                for s_p in env.states():
                    for r_idx, r in enumerate(env.rewards()):
                        action_value += env.transition_probability(s, a, s_p, r_idx) * (r + 0.999999 * V[s_p])
                        if action_value > V[s]:
                            V[s] = action_value
                            max_a = a
            pi[s, :] = 0.0
            pi[s, max_a] = 1.0
            delta = max(delta, np.abs(v - V[s]))
        if delta < 0.000001:
            break
    return PolicyAndValueFunction(pi, V)


def demo():
    '''
    print(policy_evaluation_on_line_world())
    print(policy_iteration_on_line_world())
    print(value_iteration_on_line_world())
    '''
    # print(policy_evaluation_on_grid_world().reshape(5, 5))
    # print(policy_iteration_on_grid_world())
    print(value_iteration_on_grid_world())
    exit()

    print(policy_evaluation_on_secret_env1())
    print(policy_iteration_on_secret_env1())
    print(value_iteration_on_secret_env1())
