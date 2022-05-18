import matplotlib.pyplot as plt
import numpy as np
import random

num_to_string_explanation = {
    0: 'en haut à gauche',
    1: 'en haut au milieu',
    2: 'en haut à droite',
    3: 'au milieu à gauche',
    4: 'au centre',
    5: 'au milieu à droite',
    6: 'en bas à gauche',
    7: 'en bas au milieu',
    8: 'en bas à droite'
}


def plot_evolution_tic_tac_toe(tab):
    wins = []
    losses = []
    draws = []
    cumul_win = 0
    cumul_losses = 0
    cumul_draws = 0
    for i in range(len(tab)):
        if i == 0:
            wins.append(tab[i][0])
            losses.append(tab[i][1])
            draws.append(tab[i][2])
        else:
            wins.append(tab[i][0] - cumul_win)
            losses.append(tab[i][1] - cumul_losses)
            draws.append(tab[i][2] - cumul_draws)
        cumul_win += wins[i]
        cumul_losses += losses[i]
        cumul_draws += draws[i]
    plt.plot(wins, label='wins')
    plt.plot(losses, label='losses')
    plt.plot(draws, label='draws')
    plt.legend()
    plt.show()
    print(f'wins: {wins}')
    print(f'% de win sur les 1000 derniers epochs: {np.mean(wins[-10:])}')
    print(f'losses: {losses}')
    print(f'% de losses sur les 1000 derniers epochs: {np.mean(losses[-10:])}')
    print(f'draws: {draws}')
    print(f'% de draws sur les 1000 derniers epochs: {np.mean(draws[-10:])}')


def plot_evolution_secret_env(tab, pas=100):
    score = []
    for i in range(pas, len(tab), pas):
        score.append(sum(tab[i-pas:i]))
    print(score)
    plt.plot(score)
    plt.show()


def test_policy_on_env(
        pi,
        env,
        nb_iter,
        nb_games_to_watch=100
):
    stats = []
    for it in range(1, nb_iter+1):
        env.reset()
        choose_random_action = False
        if it % 100 == 0 and it != 1:
            stats.append([env.nb_wins, env.nb_losses, env.nb_draws])
        while not env.is_game_over():
            s = env.state_id()
            available_actions = env.available_action_ids()
            if s not in pi:
                pi[s] = {}
                for a in available_actions:
                    pi[s][a] = 1.0 / len(available_actions)
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
                # chosen_action = pi[s].keys()
                choose_random_action = True
            env.act_with_action_id(chosen_action)
            if nb_games_to_watch > 0:
                nb_games_to_watch = watch_game(nb_games_to_watch, env.board, env.player_turn, env.is_game_over(), chosen_action)
    plot_evolution_tic_tac_toe(stats)


def watch_game(nb_games, board, symbol_player, game_over, num_action):
    if symbol_player == 'o':
        symbol_player = 'x'
        num_player = 1
    else:
        symbol_player = 'o'
        num_player = 2

    print(f"Le joueur numéro {num_player} a joué '{symbol_player}' à la case {num_to_string_explanation[num_action]}")
    print(np.reshape(board, (3, 3)))
    if game_over:
        if '*' in board:
            print(f'GAIN DU JOUEUR {num_player} \n'
                  '##################################################### \n'
                  '#####################################################')
        else:
            print('PARTIE NULLE')
        return nb_games - 1
    print('----------------------------------------------------')
    return nb_games
