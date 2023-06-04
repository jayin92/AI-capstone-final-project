import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import random
from collections import deque
from torch.autograd import Variable
from torch import Tensor
import os
from tqdm import tqdm
total_rewards = []


class replay_buffer():
    '''
    A deque storing trajectories
    '''
    def __init__(self, capacity):
        self.capacity = capacity  # the size of the replay buffer
        self.memory = deque(maxlen=capacity)  # replay buffer itself

    def insert(self, state, action, reward, next_state, done):

        self.memory.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, done = zip(*batch)
        return observations, actions, rewards, next_observations, done

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):
    def __init__(self,  num_actions, hidden_layer_size=50):
        super(Net, self).__init__()
        self.input_state = 4
        self.num_actions = num_actions
        self.fc1 = nn.Linear(self.input_state, 32)
        self.fc2 = nn.Linear(32, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, num_actions)

    def forward(self, states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class Agent():
    def __init__(self, env, epsilon=0.05, learning_rate=0.0002, GAMMA=0.97, batch_size=32, capacity=10000):
        self.env = env
        self.n_actions = 2  # the number of actions
        self.count = 0

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = GAMMA
        self.batch_size = batch_size
        self.capacity = capacity

        self.buffer = replay_buffer(self.capacity)
        self.evaluate_net = Net(self.n_actions)  # the evaluate network
        self.target_net = Net(self.n_actions)  # the target network

        self.optimizer = torch.optim.Adam(self.evaluate_net.parameters(), lr=self.learning_rate)  # Adam is a method using to optimize the neural network

    def learn(self):
        if self.count % 100 == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())

        batch =  self.buffer.sample(self.batch_size)
        #batch = (observations, actions, rewards, next_observations, done)
        state_batch = Variable(Tensor(np.array(batch[0])))
        action_batch = Variable(Tensor(batch[1]))
        reward_batch = Variable(Tensor(batch[2]))
        mask_batch = Variable(Tensor(batch[4]))
        next_state_batch = Variable(Tensor(np.array(batch[3])))

        Q_evaluate = torch.gather(self.evaluate_net(state_batch), 1, action_batch.view(-1, 1).long()) 
        Q_next = self.target_net(next_state_batch).detach()
        Q_target = reward_batch.view(-1, 1) + self.gamma * Q_next.max(1)[0].view(self.batch_size, 1) * (1-mask_batch.view(self.batch_size, 1))
        loss = F.mse_loss(Q_evaluate, Q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        torch.save(self.target_net.state_dict(), "./Tables/DQN.pt")
        return loss

    def choose_action(self, state):
        with torch.no_grad():
            state = torch.Tensor(state)
            if random.uniform(0,1) > self.epsilon:
                Q_val = self.evaluate_net(state)
                action = torch.argmax(Q_val).item()
            else:
                action = np.array(random.randrange(self.n_actions))
        return action

    def check_max_Q(self):
        return max(self.target_net(Tensor(self.env.reset())))


def train(env):
    agent = Agent(env)
    episode = 1000
    rewards = []
    for _ in tqdm(range(episode)):
        state = env.reset()
        count = 0
        while True:
            count += 1
            agent.count += 1
            # env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.buffer.insert(state, int(action), reward, next_state, int(done))

            if len(agent.buffer) >= 1000:
                agent.learn()
            if done:
                rewards.append(count)
                break
            state = next_state
    total_rewards.append(rewards)


def test(env):
    rewards = []
    testing_agent = Agent(env)
    testing_agent.target_net.load_state_dict(torch.load("./Tables/DQN.pt"))
    for _ in range(100):
        state = env.reset()
        count = 0
        while True:
            count += 1
            env.render()
            Q = testing_agent.target_net(torch.FloatTensor(state)).squeeze(0).detach()
            action = int(torch.argmax(Q).numpy())
            next_state, _, done, _ = env.step(action)
            if done:
                rewards.append(count)
                break
            state = next_state

    print(f"reward: {np.mean(rewards)}")
    print(f"max Q:{testing_agent.check_max_Q()}")


if __name__ == "__main__":
    env = gym.make('CartPole-v0')        
    os.makedirs("./Tables", exist_ok=True)

    train(env)
        
    test(env)
    env.close()

    os.makedirs("./Rewards", exist_ok=True)
    np.save("./Rewards/DQN_rewards.npy", np.array(total_rewards))