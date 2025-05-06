import os
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import gymnasium as gym

# Configuration de la reproductibilité
warnings.simplefilter('ignore')
os.environ['PYTHONHASHSEED'] = '0'

# Vérifier si CUDA est disponible avant de configurer cudnn
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    print("CUDA n'est pas disponible, les paramètres cudnn ne sont pas appliqués")

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hu=24):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hu)
        self.fc2 = nn.Linear(hu, hu)
        self.fc3 = nn.Linear(hu, action_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


lr = 0.005
random.seed(100)
np.random.seed(100)
torch.manual_seed(100)

class DQLAgent2:
    def __init__(self):
        self.epsilon = 1.0
        self.epsilon_decay = 0.9975
        self.epsilon_min = 0.1
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        self.gamma = 0.9
        self.trewards = []
        self.max_treward = 0
        self.env = gym.make('CartPole-v1')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._create_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def _create_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model


    def _reshape(self, state):
        state = state.flatten()
        return np.reshape(state, [1, len(state)])

    def act(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        states = torch.FloatTensor(states).to(self.device).squeeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device).squeeze(1)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        q_values = self.model(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.model(next_states).max(1)[0]
        expected_q_value = rewards + self.gamma * next_q_values * (1 - dones)
        loss = self.criterion(q_value, expected_q_value.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def learn(self, episodes):
        for e in range(1, episodes + 1):
            state, _ = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            for f in range(1, 5000):
                action = self.act(state)
                next_state, reward, done, trunc, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                self.memory.append([state, action, next_state, reward, done])
                state = next_state
                if done or trunc:
                    self.trewards.append(f)
                    self.max_treward = max(self.max_treward, f)
                    templ = f'episode={e:4d} | treward={f:4d}'
                    templ += f' | max={self.max_treward:4d}'
                    print(templ, end='\r')
                    break
            if len(self.memory) > self.batch_size:
                self.replay()
        print()

    def test(self, episodes, min_accuracy=0.0, min_performance=0.0, verbose=True, full=True):
        def test(self, episodes):
            for e in range(1, episodes + 1):
                state, _ = self.env.reset()
                state = np.reshape(state, [1, self.state_size])
                for f in range(1, 5001):
                    state_tensor = torch.FloatTensor(state).to(self.device)
                    with torch.no_grad():
                        q_values = self.model(state_tensor)
                    action = torch.argmax(q_values).item()
                    state, reward, done, trunc, _ = self.env.step(action)
                    state = np.reshape(state, [1, self.state_size])
                    if done or trunc:
                        print(f, end=' ')
                        break