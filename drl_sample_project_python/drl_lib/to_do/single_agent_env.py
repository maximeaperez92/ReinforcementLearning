class SingleAgentEnv:
    def state_id(self) -> str:
        pass

    def is_game_over(self) -> bool:
        pass

    def act_with_action_id(self, case_id: int):
        pass

    def get_score(self) -> float:
        pass

    def available_action_ids(self) -> list[int]:
        pass

    def reset(self):
        pass
