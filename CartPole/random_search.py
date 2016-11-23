import gym
import numpy as np
import pickle
env = gym.make('CartPole-v0')

# There are four variable, use a linear model W*x + b so actually five params

# Given some param, run a full episode and compute reward
def run_episode(W, max_step=1000, render=False):
    observation = env.reset()
    total_reward = 0
    N_step = 0
    done = False
    while not done and N_step <= max_step:
        if render:
            env.render()
        b = np.append(observation, 1.0)
        action = 1 if np.dot(W, b) > 0 else 0
        observation, reward, done, info = env.step(action)
        total_reward += reward
        N_step += 1

    return total_reward, N_step


train_episode = 10000
save_model = 10
def train():
    best_W = []
    best_reward = 0
    for i in range(train_episode):
        # generate random param
        W = np.random.rand(5) * 2 - 1
        reward, t = run_episode(W)
        print("Episode {} finished after {} timesteps".format(i, t+1))
        if reward > best_reward:
            best_reward = reward
            best_W = W
        if i % save_model == 0:
            pickle.dump(best_W, open('random_model.p', 'wb'))

    return best_W, best_reward


load_model = True
def run():
    if load_model:
        W = pickle.load(open('random_model.p', 'rb'))
    else:
        W, best_reward = train()
        print ("Best model reward = {}".format(best_reward))

    for i in range(20):
        _, t = run_episode(W, max_step=1000, render=True)
        print("Episode {} finished after {} timesteps".format(i, t+1))


run()
