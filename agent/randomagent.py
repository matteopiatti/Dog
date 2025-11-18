import random


class RandomAgent:
    def select_switch_card(self, state, hand):
        return random.choice(hand)

    def select_action(self, state, actions, player):
        return random.choice(actions)

    def select_split_action(self, action, allowed_steps: dict, state):
        marble = random.choice(list(allowed_steps.keys()))
        steps = random.choice(allowed_steps[marble])
        return marble, steps

    def no_actions(self):
        return None
