from collections import defaultdict
import random

class Sarsa:
    def __init__(self, env, gamma=1, eps_decay=0.999):
        self.env = env
        self.actions = env.action_space()
        self.m = len(self.actions)
        self.Q = defaultdict(lambda: 0)
        self.eps = 0.8
        self.eps_decay = eps_decay
        self.gamma = gamma
        self.alpha = 0
        self.alpha_decay = 0

    def _eps(self):
        self.eps *= self.eps_decay
        return self.eps

    def _alpha(self):
        self.alpha = self.alpha*self.alpha_decay
        return self.alpha

    def _sample_pi(self, s):
        if random.random() > self._eps():
            return max(self.actions, key=lambda x: self.Q[(s, x)])
        else:
            a = random.randint(0, self.m - 1)
            return a

    def _run_episode(self):
        s, reward_tot, done = self.env.reset()
        a = self._sample_pi(s)

        while not done:
            s_new, reward, done, _ = self.env.step(a)
            a_new = self._sample_pi(s_new)
            self.Q[(s, a)] = self.Q[(s, a)] \
                + self._alpha()*(reward + self.gamma*self.Q[(s_new, a_new)] - self.Q[(s, a)])
            s = s_new
            a = a_new
            reward_tot += reward
        return reward_tot

    def train(self, alpha=0.01, alpha_dec=0.99, episodes=10000, print_freq=100):
        self.alpha = alpha
        self.alpha_decay = alpha_dec
        r = 0
        for e in range(episodes):
            r += self._run_episode()
            if e % print_freq == 0:
                print('Itter: ' + str(e) + ' Score: ' + str(r/print_freq))
                r = 0
        return self.Q
