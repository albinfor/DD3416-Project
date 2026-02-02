import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class Dynamics6D:
    """6D Double Integrator: [x, y, z, vx, vy, vz]"""
    def __init__(self, dt=0.1):
        self.dt = dt
        self.state_dim = 6
        self.action_space = [
            np.array([1,0,0]), np.array([-1,0,0]), 
            np.array([0,1,0]), np.array([0,-1,0]), 
            np.array([0,0,1]), np.array([0,0,-1]), 
            np.array([0,0,0])                      
        ]
        self.action_dim = len(self.action_space)

    def step(self, state, action_idx):
        accel = self.action_space[action_idx]
        pos = state[:3]
        vel = state[3:]
        new_vel = (vel + accel * self.dt) * 0.95 
        new_pos = pos + new_vel * self.dt
        return np.concatenate([new_pos, new_vel])

class SafetyDQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.9, lr=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_net(state_t).argmax().item()

    def compute_loss(self, backup_fn, l_margin_fn):
        if len(self.memory) < self.batch_size:
            return None
        batch = random.sample(self.memory, self.batch_size)
        s, a, s_next = zip(*batch)
        s = torch.FloatTensor(np.array(s)).to(self.device)
        a = torch.LongTensor(a).unsqueeze(1).to(self.device)
        s_next = torch.FloatTensor(np.array(s_next)).to(self.device)
        curr_q = self.q_net(s).gather(1, a).squeeze()
        with torch.no_grad():
            targets = backup_fn(self.target_net, s, s_next, l_margin_fn, self.gamma)
        loss = nn.MSELoss()(curr_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())