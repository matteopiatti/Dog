from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def select_switch_card(self, state, hand) -> tuple:
        pass

    @abstractmethod
    def select_action(self, state, actions, player):
        pass

    @abstractmethod
    def select_split_action(self, action, allowed_steps: dict) -> tuple:
        pass

    @abstractmethod
    def no_actions(self):
        pass
