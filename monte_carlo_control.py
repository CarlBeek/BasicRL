import random


class MCControl(object):
    def __init__(self, env):
        self.Ns = dict()
        self.Nsa = dict()
        self.Q = dict()
        self.N0 = 100
        self.actions = env.get_actions()
        self.m = len(self.actions)
        self.env = env

    def _eps(self, s):
        return self.N0/(self.N0 + self._get_ns(s))

    def _alpha(self, s, a):
        return 1/self.Nsa[(s, a)]

    def _sample_pi(self, s):
        if random.random() > self._eps(s):
            return max(self.actions, key=lambda x: self._get_q(s, x))
        else:
            a = random.randint(0, self.m-1)
            return a

    def _get_q(self, s, a):
        try:
            return self.Q[(s, a)]
        except KeyError:
            return 0

    def _get_ns(self, s):
        try:
            return self.Ns[s]
        except KeyError:
            return 0

    def _update_ns(self, state):
        try:
            self.Ns[state] += 1
        except KeyError:
            self.Ns[state] = 1

    def _update_nsa(self, state, action):
        try:
            self.Nsa[(state, action)] += 1
        except KeyError:
            self.Nsa[(state, action)] = 1

    def _update_q(self, s, a, G):
        self.Q[(s, a)] = self._get_q(s, a) + self._alpha(s, a)*(G - self._get_q(s, a))

    def _run_episode(self):
        self.env.configure_game()
        state_actions = []
        while not self.env.is_terminated():
            state = self.env.get_players_observation()
            action = self._sample_pi(state)
            state_actions.append((state, action))
            self.env.step(action)

        reward = self.env.get_reward()

        for state, action in state_actions:
            self._update_ns(state)
            self._update_nsa(state, action)
            self._update_q(state, action, reward)

        return reward

    def train(self, episodes=10000, print_freq=100):
        r = 0
        for e in range(episodes):
            r += self._run_episode()
            if e % print_freq == 0:
                print('Itter: ' + str(e) + ' Score: ' + str(r/print_freq))
                r = 0
        return self.Q

