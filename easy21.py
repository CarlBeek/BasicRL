import random
ACTION_HIT = 1
ACTION_STICK = 0


class Easy21(object):
    def __init__(self):
        self.p_red = 1/3
        self.actions = [ACTION_HIT, ACTION_STICK]
        self.terminate = True
        self.reward = 0

    def reset(self):
        self._configure_dealer()
        self._configure_player()
        self.terminate = False
        self.reward = 0
        return (self.dealers_first_card, self.player_sum), self.reward, self.terminate

    def _configure_dealer(self):
        self.dealers_first_card = random.randint(1, 10)
        self.dealers_sum = self.dealers_first_card

    def _configure_player(self):
        self.player_sum = random.randint(1, 10)

    def _bust(self, card_sum):
        return card_sum > 21 or card_sum < 1

    def _draw_card(self):
        if random.random() < self.p_red:
            return - random.randint(1, 10)
        else:
            return random.randint(1, 10)

    def _terminate_game(self):
        self.terminate = True
        if self._bust(self.player_sum):
            self.reward = -1
        elif self._bust(self.dealers_sum):
            self.reward = 1
        elif self.player_sum > self.dealers_sum:
            self.reward = 1
        elif self.player_sum < self.dealers_sum:
            self.reward = -1
        else:
            self.reward = 0

    def _dealer_step(self):
        if self.dealers_sum <= 17:
            self.dealers_sum += self._draw_card()

    def action_space(self):
        return self.actions

    def step(self, action):
        if action == ACTION_HIT:
            self.player_sum += self._draw_card()
        else:
            while self.dealers_sum <= 17 and not self._bust(self.dealers_sum):
                self._dealer_step()
        if self._bust(self.player_sum) or self._bust(self.dealers_sum) or action != ACTION_HIT:
            self._terminate_game()
        return (self.dealers_first_card, self.player_sum), self.reward, self.terminate, None
