import numpy as np
import matplotlib.pyplot as plt
import easy21 as e21
import monte_carlo_control

if __name__ == '__main__':
    env = e21.Easy21()
    mcc = monte_carlo_control.MCControl(env)
    Q = mcc.train(episodes=1000000, print_freq=10000)
    grid = np.zeros([11, 22])
    for k,v in Q.items():
        (s, a) = k
        a_star = max(env.action_space(), key=lambda x: Q[s, x])
        if a == a_star:
            grid[s[0], s[1]] = v
    plt.imshow(grid)
    plt.colorbar()
    plt.show()