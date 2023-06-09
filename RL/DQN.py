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
#from RLenv import TSP_Env
from env import TSP_Env
from torch.utils.tensorboard import SummaryWriter
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
    def __init__(self,  input_dim, num_actions, hidden_layer_size=256):
        super(Net, self).__init__()
        self.input_state = input_dim
        self.num_actions = num_actions
        self.fc1 = nn.Linear(self.input_state, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc3 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fc4 = nn.Linear(hidden_layer_size, num_actions)

    def forward(self, states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_values = self.fc4(x)
        return q_values


class Agent():
    def __init__(self, env, epsilon=0.05, learning_rate=0.01, lr_decay_rate=0.999, GAMMA=0.97, batch_size=32, capacity=10000, hidden_size= 600):
        self.env = env
        self.n_actions = env.n  # the number of actions
        self.state_dim = env.observation_space
        self.count = 0

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = GAMMA
        self.batch_size = batch_size
        self.capacity = capacity

        self.buffer = replay_buffer(self.capacity)
        self.evaluate_net = Net(self.state_dim, self.n_actions, hidden_size)  # the evaluate network
        self.target_net = Net(self.state_dim, self.n_actions, hidden_size)  # the target network

        self.optimizer = torch.optim.Adam(self.evaluate_net.parameters(), lr=self.learning_rate)  # Adam is a method using to optimize the neural network
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=lr_decay_rate)

    def learn(self):
        if self.count % 100 == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())

        batch =  self.buffer.sample(self.batch_size)
        #batch = (observations, actions, rewards, next_observations, done)
        state_batch = Variable(Tensor(np.array(batch[0])))
        action_batch = Variable(Tensor(batch[1]))
        reward_batch = Variable(Tensor(batch[2]))
        next_state_batch = Variable(Tensor(np.array(batch[3])))
        mask_batch = Variable(Tensor(batch[4]))

        Q_evaluate = torch.gather(self.evaluate_net(state_batch), 1, action_batch.view(-1, 1).long()) 
        Q_next = self.target_net(next_state_batch).detach()
        Q_next = Q_next * (1-mask_batch)
        Q_target = reward_batch.view(self.batch_size, 1) + self.gamma * Q_next.max(1)[0].view(self.batch_size, 1)
        #print(Q_target)
        loss = F.mse_loss(Q_evaluate, Q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        torch.save(self.target_net.state_dict(), "./RL/Tables/DQN.pt")
        return loss.detach().numpy()

    def choose_action(self, state, mask):
        with torch.no_grad():
            state = torch.Tensor(state)
            if random.uniform(0,1) > self.epsilon:
                Q_val = self.evaluate_net(state).numpy()
                #print(Q_val)
                bst_val = -999999999
                for i in range(self.n_actions):
                    if mask[i] == 0 and Q_val[i] > bst_val:
                        bst_val = Q_val[i]
                        action = i
                #print(action)
                #action = torch.argmax(Q_val).item()
            else:
                action = np.array(random.randrange(self.n_actions))
                while mask[action]==1:
                    action = np.array(random.randrange(self.n_actions))
        return action

def train(env, episode, epsilon=0.03, learning_rate=0.01, lr_decay_rate=0.999, GAMMA=0.97, batch_size=32, capacity=10000, hidden_size= 500):
    agent = Agent(env, epsilon, learning_rate, lr_decay_rate, GAMMA, batch_size, capacity, hidden_size)
    writer = SummaryWriter('./RL/tb_record/DQN/')
    ewma_rewards = 0
    for e in range(episode):
        state, mask = env.reset()
        count = 0
        tot_rewards = 0
        losses = []
        while True:
            count += 1
            agent.count += 1
            # env.render()
            action = agent.choose_action(state, mask)
            next_state, reward, mask, done, info = env.step(int(action))
            
            agent.buffer.insert(state, int(action), reward, next_state, mask)
            tot_rewards += reward
            if len(agent.buffer) >= agent.batch_size:
                losses.append(agent.learn())
            if done:
                break
            state = next_state
        loss = np.mean(np.array(losses)) if len(losses) > 0 else 0
        ewma_rewards = tot_rewards * 0.05 + 0.95 * ewma_rewards
        print('Episode: {}/ Total reward: {:.3f}/ EWMA rewards: {:.3f}/ Total distance: {:.3f}/ Loss: {:.3f}'.format(e, tot_rewards, ewma_rewards, info['Distance'], loss))
        writer.add_scalar('Rewards', tot_rewards, e)
        writer.add_scalar('EWMA', ewma_rewards, e)
        writer.add_scalar('Distance', info['Distance'], e)
        writer.add_scalar('Loss', loss, e)
    

def test(env, episode, epsilon=-1, hidden_size= 600):
    rewards = []
    distance = []
    testing_agent = Agent(env, epsilon, hidden_size)
    testing_agent.evaluate_net.load_state_dict(torch.load("./RL/Tables/DQN.pt"))
    for e in range(episode):
        state, mask = env.reset()
        R = 0
        while True:
            #env.render()
            action = testing_agent.choose_action(state, mask)
            next_state, reward, mask, done, info = env.step(int(action))
            R+=reward
            if done:
                rewards.append(R)
                distance.append(info['Distance'])
                break
            state = next_state

    print(f"reward: {np.mean(rewards)}")
    print(f"distance: {np.mean(distance)}")

if __name__ == "__main__":
    seed = 10
    env = TSP_Env(name='berlin52', seed=seed)
    torch.manual_seed(seed=seed)
    train(env,episode=125,epsilon=0.03, learning_rate=5e-3, lr_decay_rate=1. - 2e-5,GAMMA=0.997, batch_size=32, capacity=10000, hidden_size= 600)
    test(env, episode=100, epsilon=0,  hidden_size= 600)