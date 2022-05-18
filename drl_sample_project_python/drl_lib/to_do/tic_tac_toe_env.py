from ..to_do.single_agent_env import SingleAgentEnv

import numpy as np
import random


class TicTacToe(SingleAgentEnv):
    def __init__(self, play_first):
        self.player_turn = 'x'
        self.play_first = play_first
        self.board = ['*'] * 9
        self.nb_wins = 0
        self.nb_losses = 0
        self.nb_draws = 0
        self.game_over = False

    def state_id(self) -> str:
        return ''.join(self.board)

    def is_game_over(self):
        return self.game_over

    def check_game(self) -> None:
        if self.play_first:
            win_p1 = ['x'] * 3
        else:
            win_p1 = ['o'] * 3
        if self.play_first:
            win_p2 = ['o'] * 3
        else:
            win_p2 = ['x'] * 3

        self.game_over = True

        # check win player 1
        # check lines
        if np.all(self.board[0:3] == win_p1) or np.all(self.board[3:6] == win_p1) or np.all(self.board[6:9] == win_p1):
            self.nb_wins += 1
            return
        # check columns
        if np.all(self.board[0:10:3] == win_p1) or np.all(self.board[1:10:3] == win_p1) or np.all(self.board[2:10:3] == win_p1):
            self.nb_wins += 1
            return
        # check diagonals
        if np.all(self.board[0:10:4] == win_p1) or np.all(self.board[2:7:2] == win_p1):
            self.nb_wins += 1
            return

        # check win player 2
        # check lines
        if np.all(self.board[0:3] == win_p2) or np.all(self.board[3:6] == win_p2) or np.all(
                self.board[6:9] == win_p2):
            self.nb_losses += 1
            return
        # check columns
        if np.all(self.board[0:10:3] == win_p2) or np.all(self.board[1:10:3] == win_p2) or np.all(
                self.board[2:10:3] == win_p2):
            self.nb_losses += 1
            return
        # check diagonals
        if np.all(self.board[0:10:4] == win_p2) or np.all(self.board[2:7:2] == win_p2):
            self.nb_losses += 1
            return
        if '*' not in self.board[0:9]:
            self.nb_draws += 1
            return
        self.game_over = False

    def act_with_action_id(self, case_id: int):
        self.board[case_id] = self.player_turn
        self.check_game()
        self.change_turn()

    def get_score(self) -> float:
        return self.nb_wins - 100 * self.nb_losses

    def print_score(self):
        print(f'nb wins : {self.nb_wins}')
        print(f'nb looses: {self.nb_losses}')
        print(f'nb draws: {self.nb_draws}')

    def available_action_ids(self) -> list[int]:
        return [i for i, val in enumerate(self.board) if val == '*']

    def change_turn(self):
        if self.player_turn == 'x':
            self.player_turn = 'o'
        else:
            self.player_turn = 'x'

    def reset(self):
        self.player_turn = 'x'
        self.board = ['*'] * 9
        self.game_over = False

    def reset_random(self, p_random):
        self.player_turn = 'x'
        self.board = ['*'] * 9
        self.game_over = False
        if random.uniform(0, 1) < p_random:
            nb_random = random.randint(1, 8)
            while nb_random > 0:
                nb_random -= 1
                action_chosen = random.choice(self.available_action_ids())
                self.act_with_action_id(action_chosen)
                if self.is_game_over():
                    return True
        return False
