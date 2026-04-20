import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


NUM_TOPICS       = 5
CONTENT_TYPES    = ["video", "example", "quiz", "summary", "tutorial", "challenge"]
DIFFICULTY_LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]

NUM_CONTENT_TYPES = len(CONTENT_TYPES)
NUM_DIFFICULTIES  = len(DIFFICULTY_LEVELS)

STATE_SIZE  = NUM_TOPICS * 3
ACTION_SIZE = NUM_TOPICS * NUM_CONTENT_TYPES * NUM_DIFFICULTIES  # 150

MASTERY_THRESHOLD  = 0.85
STRUGGLE_THRESHOLD = 0.25
ZPD_TOLERANCE      = 0.25 # Murray & Arroyo (2002)

PROFILES = {
    "fast":      {"lr": 0.15, "forget": 0.005, "init_low": 0.20, "init_high": 0.50},
    "average":   {"lr": 0.10, "forget": 0.008, "init_low": 0.10, "init_high": 0.40},
    "slow":      {"lr": 0.06, "forget": 0.012, "init_low": 0.10, "init_high": 0.35},
    "struggling":{"lr": 0.04, "forget": 0.015, "init_low": 0.05, "init_high": 0.20},
}

CONTENT_MULTIPLIERS = {
    "video":     lambda k: 0.90,
    "example":   lambda k: 1.00,
    "quiz":      lambda k: 1.10 if k >= 0.40 else 0.65,
    "summary":   lambda k: 0.80,
    "tutorial":  lambda k: 1.20 if k <= 0.40 else 0.70,
    "challenge": lambda k: 1.30 if k >= 0.60 else 0.40,
}

class StudentSimulator:
    def __init__(self, profile="average"):
        cfg = PROFILES[profile]
        self.learning_rate  = cfg["lr"]
        self.forgetting_rate = cfg["forget"]
        self.knowledge = np.random.uniform(cfg["init_low"], cfg["init_high"], size=NUM_TOPICS)
        self.preference = random.choice(CONTENT_TYPES)  # hidden student preference

    def step(self, topic, content_type, difficulty):
        k = self.knowledge[topic]

        # ZPD factor Murray & Arroyo 2002
        zpd_factor = max(0.0, 1.0 - abs(difficulty - k) / ZPD_TOLERANCE)
        c_mult = CONTENT_MULTIPLIERS[content_type](k)

        # Small preference boost if content matches student's hidden preference
        pref_boost = 1.15 if content_type == self.preference else 1.0
        # drops off sharply when content is far outside ZPD
        engagement = 1.0 if abs(difficulty - k) <= ZPD_TOLERANCE else 0.30

        # ΔK = LR*ZPD_factor*engagement*preference_boost*content_multiplier
        delta_k = self.learning_rate * zpd_factor * engagement * pref_boost * c_mult

        if k < STRUGGLE_THRESHOLD and content_type in ["tutorial", "example", "summary"]:
            delta_k *= 1.25
        elif k < STRUGGLE_THRESHOLD and content_type in ["challenge", "quiz"]:
            delta_k *= 0.55

        prev_knowledge = self.knowledge.copy()
        self.knowledge[topic] = float(np.clip(k + delta_k, 0.0, 1.0))

        # Ebbinghaus forgetting on all topics each step
        decay = self.forgetting_rate * np.exp(-self.knowledge * 2.0)
        self.knowledge = np.clip(self.knowledge - decay, 0.0, 1.0)

        difficulty_gap = abs(difficulty - k)
        return self.knowledge.copy(), delta_k, engagement, difficulty_gap, prev_knowledge


class LearningEnv:
    def __init__(self, profile="average"):
        self.profile = profile
        self.student  = StudentSimulator(profile)
        self.prev_action_idx = None
        self.prev_knowledge  = self.student.knowledge.copy()

    def reset(self):
        self.student = StudentSimulator(self.profile)
        self.prev_action_idx = None
        self.prev_knowledge  = self.student.knowledge.copy()
        return self._state()

    def _state(self):
        k = self.student.knowledge
        struggle = (k < STRUGGLE_THRESHOLD).astype(float)
        mastered = (k >= MASTERY_THRESHOLD).astype(float)
        return np.concatenate([k, struggle, mastered])

    def decode_action(self, idx):
        topic    = idx // (NUM_CONTENT_TYPES * NUM_DIFFICULTIES)
        rem      = idx %  (NUM_CONTENT_TYPES * NUM_DIFFICULTIES)
        c_idx    = rem // NUM_DIFFICULTIES
        d_idx    = rem %  NUM_DIFFICULTIES
        return topic, CONTENT_TYPES[c_idx], DIFFICULTY_LEVELS[d_idx]

    def step(self, action_idx):
        topic, ctype, diff = self.decode_action(action_idx)

        next_k, delta_k, engagement, diff_gap, prev_k = \
            self.student.step(topic, ctype, diff)

        # Reward function (from paper):
        # R = 3.0*ΔK + 1.5*E + 2.0*ΔK_total + B_mastery - 0.8*I_redundant - λ * max(0, |d - k_τ| - δ_ZPD)

        delta_k_total = float(np.sum(next_k - prev_k))

        mastered_now  = np.sum(next_k  >= MASTERY_THRESHOLD)
        mastered_prev = np.sum(prev_k  >= MASTERY_THRESHOLD)
        mastery_bonus = 0.1 * max(0, mastered_now - mastered_prev)

        redundancy = 1.0 if self.prev_action_idx == action_idx else 0.0

        zpd_penalty = 2.0 * max(0.0, diff_gap - ZPD_TOLERANCE)

        reward = (
            3.0 * delta_k
            + 1.5 * engagement
            + 2.0 * delta_k_total
            + mastery_bonus
            - 0.8 * redundancy
            - zpd_penalty
        )

        #scaffolds a struggling student
        if self.student.knowledge[topic] < STRUGGLE_THRESHOLD and \
           ctype in ["tutorial", "example", "summary"]:
            reward += 0.5

        self.prev_action_idx = action_idx
        self.prev_knowledge  = next_k.copy()

        done = bool(np.all(next_k >= MASTERY_THRESHOLD))
        return self._state(), reward, done

class DuelingDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(STATE_SIZE, 128), nn.ReLU(),
            nn.Linear(128, 128),        nn.ReLU(),
        )
        self.value_head     = nn.Linear(128, 1)
        self.advantage_head = nn.Linear(128, ACTION_SIZE)

    def forward(self, x):
        h = self.shared(x)
        v = self.value_head(h)
        a = self.advantage_head(h)
        return v + (a - a.mean(dim=1, keepdim=True))

class Agent:
    def __init__(self):
        self.online = DuelingDQN()
        self.target = DuelingDQN()
        self.target.load_state_dict(self.online.state_dict())

        self.optimizer = optim.Adam(self.online.parameters(), lr=5e-4)
        self.memory    = deque(maxlen=10000)

        self.gamma         = 0.99
        self.epsilon       = 1.0
        self.epsilon_decay = 0.9997 
        self.epsilon_min   = 0.05
        self.batch_size    = 64

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_SIZE - 1)
        with torch.no_grad():
            q = self.online(torch.FloatTensor(state).unsqueeze(0))
        return int(q.argmax().item())

    def store(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch               = random.sample(self.memory, self.batch_size)
        s, a, r, s2, done   = zip(*batch)

        s    = torch.FloatTensor(np.array(s))
        s2   = torch.FloatTensor(np.array(s2))
        a    = torch.LongTensor(a)
        r    = torch.FloatTensor(r)
        done = torch.FloatTensor(done)

        current_q = self.online(s).gather(1, a.unsqueeze(1)).squeeze()

        with torch.no_grad():
            best_a   = self.online(s2).argmax(dim=1)          # Double DQN
            target_q = r + self.gamma * (1 - done) * \
                       self.target(s2).gather(1, best_a.unsqueeze(1)).squeeze()

        loss = nn.HuberLoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())


def train(episodes=1000, steps=60, sync_every=10):
    agent    = Agent()
    profiles = list(PROFILES.keys())
    ep_rewards = []

    for ep in range(episodes):
        profile = random.choice(profiles)
        env     = LearningEnv(profile=profile)
        state   = env.reset()
        total_r = 0.0

        for _ in range(steps):
            action              = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.store(state, action, reward, next_state, float(done))
            agent.train_step()
            state   = next_state
            total_r += reward
            if done:
                break

        if ep % sync_every == 0:
            agent.sync_target()

        ep_rewards.append(total_r)

        if (ep + 1) % 100 == 0:
            last100 = np.mean(ep_rewards[-100:])
            avg_k   = np.mean(env.student.knowledge)
            print(f"Ep {ep+1:>4} | Profile: {profile:<10} | "
                  f"Reward: {total_r:>7.2f} | "
                  f"Avg(last 100): {last100:>6.2f} | "
                  f"Avg Know: {avg_k:.3f} | "
                  f"ε: {agent.epsilon:.3f}")

    return agent, ep_rewards

if __name__ == "__main__":
    agent, rewards = train()
