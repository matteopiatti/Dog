from agent.agent import Agent
from dog.cli import render, select_switch_card as cli_select_switch_card
from dog.cli import select_action as cli_select_action
from dog.cli import select_split_action as cli_select_split_action
from dog.cli import print_no_actions as cli_print_no_actions


class HumanAgent(Agent):
    def select_switch_card(self, state, hand) -> tuple:
        return cli_select_switch_card(state)

    def select_action(self, state, actions, player):
        return cli_select_action(state)

    def select_split_action(self, action, allowed_steps: dict, state) -> tuple:
        render(state)
        return cli_select_split_action(action, allowed_steps)

    def no_actions(self):
        return cli_print_no_actions()
