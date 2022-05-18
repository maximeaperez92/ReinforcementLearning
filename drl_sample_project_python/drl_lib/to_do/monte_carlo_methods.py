import random

import numpy as np
from tqdm import tqdm

from ..do_not_touch.result_structures import PolicyAndActionValueFunction
from ..do_not_touch.single_agent_env_wrapper import Env2
from ..to_do.tic_tac_toe_env import TicTacToe
from ..to_do.visualisation import plot_evolution_tic_tac_toe, plot_evolution_secret_env, test_policy_on_env


def monte_carlo_es_on_tic_tac_toe_solo(
        nb_iter: int,
        gamma: float,
        play_first: bool,
        p_random: float,
        decay_random: float) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    env = TicTacToe(play_first)
    pi = {}
    q = {}
    returns = {}

    stats = []

    for it in tqdm(range(nb_iter)):
        p_random *= decay_random
        start_over = env.reset_random(p_random)

        S = []
        A = []
        R = []

        if it % 100 == 0 and it != 0:
            stats.append([env.nb_wins, env.nb_losses, env.nb_draws])

        # Generate an episode
        while not start_over and not env.is_game_over():
            s = env.state_id()
            available_actions = env.available_action_ids()
            S.append(s)
            if s not in pi:
                pi[s] = {}
                q[s] = {}
                returns[s] = {}
                for a in available_actions:
                    pi[s][a] = 1.0 / len(available_actions)
                    q[s][a] = 0.0
                    returns[s][a] = []
            if (play_first and env.player_turn == 'o') or (not play_first and env.player_turn == 'x'):
                chosen_action = random.choice(available_actions)
            else:
                chosen_action = np.random.choice(
                    list(pi[s].keys()),
                    1,
                    False,
                    p=list(pi[s].values())
                )[0]

            A.append(chosen_action)
            old_score = env.get_score()
            env.act_with_action_id(chosen_action)
            r = env.get_score() - old_score
            R.append(r)
        G = 0
        for t in reversed(range(len(S))):
            G = gamma * G + R[t]
            s_t = S[t]
            a_t = A[t]
            found = False
            for p_s, p_a in zip(S[:t], A[:t]):
                if s_t == p_s and a_t == p_a:
                    found = True
                    break
            if found:
                continue
            returns[s_t][a_t].append(G)
            q[s_t][a_t] = np.mean(returns[s_t][a_t])
            ind_best_a = list(q[s_t].keys())[np.argmax(list(q[s_t].values()))]
            pi[s_t] = {ind_best_a: 1.0}
    print(len(pi))
    print(p_random)
    plot_evolution_tic_tac_toe(stats)
    exit()
    return PolicyAndActionValueFunction(pi, q)


def on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo(
        nb_iter: int,
        gamma: float,
        play_first: bool,
        epsilon: float
) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy
    and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    pi = {}
    q = {}
    returns = {}
    stats = []

    env = TicTacToe(play_first)

    for it in tqdm(range(nb_iter)):
        env.reset()
        S = []
        A = []
        R = []
        if it % 100 == 0 and it != 0:
            stats.append([env.nb_wins, env.nb_losses, env.nb_draws])
        while not env.is_game_over():
            s = env.state_id()
            S.append(s)
            available_actions = env.available_action_ids()
            if s not in pi:
                pi[s] = {}
                q[s] = {}
                returns[s] = {}
                for a in available_actions:
                    pi[s][a] = 1.0 / len(available_actions)
                    # q[s][a] = 0.0
                    q[s][a] = 0
                    returns[s][a] = []
                    # returns[s][a].append(len(available_actions) * 100)

            if (play_first and env.player_turn == 'o') or (not play_first and env.player_turn == 'x'):
                chosen_action = random.choice(available_actions)
            else:
                nb_random = random.uniform(0, 1)
                if nb_random < epsilon:
                    chosen_action = np.random.choice(
                        list(pi[s].keys()),
                        1,
                        False,
                    )[0]
                else:
                    chosen_action = np.random.choice(
                        list(pi[s].keys()),
                        1,
                        False,
                        p=list(pi[s].values())
                    )[0]

            A.append(chosen_action)
            old_score = env.get_score()
            env.act_with_action_id(chosen_action)
            r = env.get_score() - old_score
            R.append(r)

        G = 0
        for t in reversed(range(len(S))):
            G = gamma * G + R[t]
            s_t = S[t]
            a_t = A[t]
            found = False
            for p_s, p_a in zip(S[:t], A[:t]):
                if s_t == p_s and a_t == p_a:
                    found = True
                    break
            if found:
                continue
            returns[s_t][a_t].append(G)
            q[s_t][a_t] = np.mean(returns[s_t][a_t])
            optimal_a_t = list(q[s_t].keys())[np.argmax(list(q[s_t].values()))]
            available_actions_t_count = len(q[s_t])
            for a_key, q_s_a in q[s_t].items():
                if a_key == optimal_a_t:
                    pi[s_t][a_key] = 1 - epsilon + epsilon / available_actions_t_count
                else:
                    pi[s_t][a_key] = epsilon / available_actions_t_count
    plot_evolution_tic_tac_toe(stats)
    return PolicyAndActionValueFunction(pi, q)


def off_policy_monte_carlo_control_on_tic_tac_toe_solo(
        nb_iter: int,
        gamma: float,
        play_first: bool,
        epsilon: float
) -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    q = {}
    c = {}
    pi = {}
    b = {}
    stats = []
    env = TicTacToe(play_first)
    for it in tqdm(range(nb_iter)):
        env.reset()
        if it % 100 == 0 and it != 0:
            stats.append([env.nb_wins, env.nb_losses, env.nb_draws])
        S = []
        A = []
        R = []
        while not env.is_game_over():
            s = env.state_id()
            S.append(s)
            available_actions = env.available_action_ids()
            if s not in b:
                b[s] = {}
                q[s] = {}
                for a in available_actions:
                    b[s][a] = 1.0 / len(available_actions)
                    q[s][a] = -500
                    pi[s] = {}
            if (play_first and env.player_turn == 'o') or (not play_first and env.player_turn == 'x'):
                chosen_action = random.choice(available_actions)
            else:
                nb_random = random.uniform(0, 1)
                if nb_random < epsilon:
                    chosen_action = np.random.choice(
                        list(b[s].keys()),
                        1,
                        False,
                    )[0]
                else:
                    chosen_action = np.random.choice(
                        list(b[s].keys()),
                        1,
                        False,
                        p=list(b[s].values())
                    )[0]

            A.append(chosen_action)
            old_score = env.get_score()
            env.act_with_action_id(chosen_action)
            r = env.get_score() - old_score
            R.append(r)
        G = 0
        W = 1
        for t in reversed(range(len(S))):
            G = gamma * G + R[t]
            s_t = S[t]
            a_t = A[t]
            if s_t not in c:
                c[s_t] = {}
                c[s_t][a_t] = W
            elif a_t not in c[s_t]:
                c[s_t][a_t] = W
            else:
                c[s_t][a_t] += W

            q[s_t][a_t] += (W / c[s_t][a_t]) * (G - q[s_t][a_t])
            ind_a = list(q[s_t].keys())[np.argmax(list(q[s_t].values()))]
            '''
            if a_t != ind_a and q[s_t][a_t] == q[s_t][ind_a]:
                ind_a = a_t
            '''
            pi[s_t] = {ind_a: 1.0}
            # print(f'{a_t}, {list(pi[s_t].keys())[0]}')
            if a_t != list(pi[s_t].keys())[0]:  # si l'action choisie est différente de l'action optimale
                break
            W = W * (1 / b[s_t][a_t])
    # plot_evolution_tic_tac_toe(stats)
    print(pi)
    print(len(pi))
    return PolicyAndActionValueFunction(pi, q)




def monte_carlo_es_on_secret_env2(
        nb_iter: int,
        gamma: float) -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    env = Env2()
    pi = {}
    q = {}
    returns = {}
    stats = []
    for _ in tqdm(range(nb_iter)):
        env.reset()
        S = []
        A = []
        R = []
        choose_random_action = False  # policy agent play first
        # Generate an episode
        while not env.is_game_over():
            s = env.state_id()
            available_actions = env.available_actions_ids()
            S.append(s)
            if s not in pi:
                pi[s] = {}
                q[s] = {}
                returns[s] = {}
                for a in available_actions:
                    pi[s][a] = 1.0 / len(available_actions)
                    q[s][a] = 0.0
                    returns[s][a] = []
            if choose_random_action:
                chosen_action = random.choice(available_actions)
                choose_random_action = False
            else:
                chosen_action = np.random.choice(
                    list(pi[s].keys()),
                    1,
                    False,
                    p=list(pi[s].values())
                )[0]
                choose_random_action = True

            A.append(chosen_action)
            old_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - old_score
            R.append(r)
        if len(R) == 0:
            stats.append(0)
        else:
            stats.append(R[-1])
        G = 0
        for t in reversed(range(len(S))):
            G = gamma * G + R[t]
            s_t = S[t]
            a_t = A[t]
            found = False
            for p_s, p_a in zip(S[:t], A[:t]):
                if s_t == p_s and a_t == p_a:
                    found = True
                    break
            if found:
                continue
            returns[s_t][a_t].append(G)
            q[s_t][a_t] = np.mean(returns[s_t][a_t])
            ind_best_a = list(q[s_t].keys())[np.argmax(list(q[s_t].values()))]
            pi[s_t] = {ind_best_a: 1.0}
    plot_evolution_secret_env(stats)
    return PolicyAndActionValueFunction(pi, q)


def on_policy_first_visit_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env2()
    # TODO
    pass


def off_policy_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env2()
    # TODO
    pass


def demo():
    # print(monte_carlo_es_on_tic_tac_toe_solo(nb_iter=20000, gamma=0.999999, play_first=True,
    #                                          p_random=0.1, decay_random=1))

    # returns = on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo(nb_iter=50000, gamma=0.99999, play_first=True,
    #                                                                         epsilon=0.1)
    # test_policy_on_env(returns.pi, TicTacToe(True), 1000)
    # exit()

    returns = off_policy_monte_carlo_control_on_tic_tac_toe_solo(nb_iter=100000, gamma=0.999999,
                                                                 play_first=True, epsilon=0.3)
    test_policy_on_env(returns.pi, TicTacToe(True), 1000)
    exit()

    print(monte_carlo_es_on_secret_env2(nb_iter=30000, gamma=0.99999))
    exit()
    print(on_policy_first_visit_monte_carlo_control_on_secret_env2())
    print(off_policy_monte_carlo_control_on_secret_env2())
