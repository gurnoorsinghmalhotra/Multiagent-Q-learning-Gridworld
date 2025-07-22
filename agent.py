from replay_buffer import ReplayBuffer
import numpy as np
import random

class DQNAgent:
    def __init__(self, agent_id, agent_type, shared_dqn, shared_replay_buffer, action_size=4, gamma=0.95, 
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.99, batch_size=64, target_update_freq=200):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.shared_dqn = shared_dqn
        self.shared_replay_buffer = shared_replay_buffer
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_counter = 0

    def remember(self, state, action, reward, next_state, done):
        """Store the experience in the shared replay buffer."""
        # print(f"Storing state with shape: {state.shape}")
        self.shared_replay_buffer.append((state, int(action), reward, next_state, done))

    def act(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() <= self.epsilon:
            return int(np.random.randint(self.action_size))
        else:
            qvals = self.shared_dqn.get_qvals(state)
            return int(np.argmax(qvals[0].detach().numpy()))

    def replay(self):
        """Train the DQN with experiences sampled from the shared replay buffer."""
        if len(self.shared_replay_buffer) < self.batch_size:
            return

        # Sample a minibatch of experiences
        minibatch = random.sample(self.shared_replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Ensure each state has the correct shape before passing to train_one_step
        # This converts each state to a 2D tensor with a single row (if it's not already)
        states = [np.array(state).reshape(-1) for state in states]
        next_states = [np.array(state).reshape(-1) for state in next_states]

        # Calculate the targets for training
        targets = [
            reward + self.gamma * (1 - done) * self.shared_dqn.get_maxQ(next_state)
            for reward, next_state, done in zip(rewards, next_states, dones)
        ]

        # Train the model with the prepared states, actions, and calculated targets
        loss = self.shared_dqn.train_one_step(states, actions, targets)
        return loss 


    def update_epsilon(self):
        """Decrease the exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        """Update the target network."""
        self.shared_dqn.update_target()

    def increment_step(self):
        """Increment the step counter and update target network if needed."""
        self.step_counter += 1
        if self.step_counter % self.target_update_freq == 0:
            self.update_target_network()
