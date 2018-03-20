from dqn import DQN
from stock_env import StockEnv
import torch
from torch.autograd import Variable
from torch import optim
import numpy as np
import random
import itertools

from collections import namedtuple, deque


def one_hot(state, grad=True):
    tensor = torch.zeros([1, env.observation_space.n])
    tensor[0, state] = 1
    return Variable(tensor, requires_grad=grad)


def updateDQN(minibatch, predDQN, targetDQN):
    # print(minibatch)
    optim.zero_grad()
    states = torch.stack(
        [torch.FloatTensor(example.state) for example in minibatch])
    Qpred = predDQN(states)
    Qs = Qpred.clone()
    for i, replay in enumerate(minibatch):
        if replay.done:
            Qs[i, replay.action] = replay.reward
        else:
            target_dqn = targetDQN(replay.new_state)
            # target_dqn = predDQN(replay.new_state)
            # print(f"target:{target_dqn}")
            Qs[i, replay.action] = replay.reward + gamma * torch.max(
                target_dqn)

    loss = torch.mean((Qpred - Qs) ** 2)
    # print(f"loss:{loss.data[0]}")
    loss.backward()
    optim.step()

'''
035720 카카오
005930 삼성전자
000660 SK하이닉스
000120 CJ대한통운
285130 SK케미칼
008970 동양철관
'''
# env = StockEnv("000660")
env = StockEnv("000660")
Replay = namedtuple("Replay",
    ["state", "action", "new_state", "reward", "done"])
predDQN = DQN(env.num_state(), env.num_action(), 40)
targetDQN = DQN(env.num_state(), env.num_action(), 40)
# optim = torch.optim.Adam(predDQN.parameters(), lr=0.1)
optim = torch.optim.SGD(predDQN.parameters(), lr=0.1)

replay_buffer = deque()
BUFFER_SIZE = 5000

num_episods = 30000
gamma = 0.9

reward_history = []
duration_history = []
prev_score = avg_return = None
decaying = 0.99
e = 1.0
for episode in range(num_episods):
    # e = 1.0 / (np.sqrt(episode) * 1 + 1)
    e = 1.0 / (episode / 10 + 1)
    # e = ramdom_action_prob
    # e = 0.2 / (episode / 5000 + 1) * 0.5 * (1 + np.cos(2 * np.pi * episode/5000))
    if episode > 0.9 * num_episods:
        e = 0.0
    state = env.reset()

    reward_sum = 0.0
    for step in range(5000):
        # if episode % 100 == 0:
        #     env.render()
        if np.random.rand(1) < e:
            a = env.random_action()
        else:
            Qs = predDQN(state)
            _, i = torch.max(Qs.data, 0)
            a = i[0]
        new_state, reward, done, info = env.step(a)

        replay_buffer.append(Replay(state, a, new_state, reward, done))
        if len(replay_buffer) > BUFFER_SIZE:
            replay_buffer.popleft()

        state = new_state
        reward_sum += reward
        if done:
            # print(f"Episode: {episode}, Step: {step}, Reward: {reward_sum}")
            reward_history.append(reward_sum)
            duration_history.append(info.duration)
            break

    if episode % 20 == 0:
        print(f"Episode: {episode}, Return: {reward_history[-1]:.5}, Duration: {duration_history[-1]}, e: {e}")
    if episode % 100 == 0 and episode > 100:
        reward_history = reward_history[-100:]
        avg_return = np.mean(reward_history)
        var_return = np.var(reward_history)
        duration_history = duration_history[-100:]
        avg_duration = np.mean(duration_history)
        var_duration = np.var(duration_history)
        print(
            f"Episode: {episode}, Average return: {avg_return:.5}({var_return:.5}), Average duration: {avg_duration:.5}({var_duration:.5}), e: {e}")
        # for k, v in itertools.groupby(sorted(duration_history), lambda t: t // 10):
        #     print(k, len(v))


    if episode % 10 == 3:
        targetDQN.sync(predDQN)
        for i in range(50):
            minibatch = random.sample(replay_buffer, 32)
            updateDQN(minibatch, predDQN, targetDQN)
    #print(f"{step}, action:{a}, trade price:{s[0]:.5}, buy price:{s[1].buy_price:.5}, reward:{reward:.5}, done:{done}")

