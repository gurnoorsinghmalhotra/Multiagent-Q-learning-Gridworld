import numpy as np


class GridWorld:
    def __init__(self, size=5, n_agents=4, agent_types=None, reward=None):
        self.size = size
        self.n_agents = n_agents
        self.reward = reward
        self.agent_types = agent_types
        self.reset()

    def assign_agent_types(self):
       """Assign agent types (0 or 1) to the agents."""
       if self.agent_types is None:
           # Default assignment if no agent types are provided
           self.agent_types = [i % 2 for i in range(self.n_agents)]
       else:
           # Use the provided agent types
           self.agent_types = self.agent_types

    def reset(self):
        """Initialize agent positions, paths, and the target location (B)."""
        self.agent_positions = []
        self.paths = [[] for _ in range(self.n_agents)]
        self.has_full_secret = [False] * self.n_agents

        # Assign types to agents
        self.assign_agent_types()

        # Randomly set the location of B
        self.b_location = self.random_position()

        # Ensure valid positions for all agents
        for _ in range(self.n_agents):
            while True:
                pos = self.random_position()
                if pos != self.b_location:
                    self.agent_positions.append(pos)
                    break

        # Return the states for all agents as a list of individual states
        all_states = [self.get_state(agent_index) for agent_index in range(self.n_agents)]
        return all_states


    def random_position(self):
        """Generate a random (x, y) position in the grid."""
        return np.random.randint(0, self.size), np.random.randint(0, self.size)
    
    def get_closest_opposite_agent_location(self, agent_index):
        """
        Find the location (x, y) of the closest agent of the opposite type.
        :param agent_index: Index of the current agent.
        :return: (x, y) coordinates of the closest opposite type agent.
        """
        current_position = self.agent_positions[agent_index]
        current_type = self.agent_types[agent_index]

        closest_position = None
        min_distance = float('inf')

        # Iterate through other agents to find the closest of the opposite type
        for i, pos in enumerate(self.agent_positions):
            if i != agent_index and self.agent_types[i] != current_type:
                distance = self.manhattan_distance(current_position, pos)
                if distance < min_distance:
                    min_distance = distance
                    closest_position = pos

        # If no opposite agent is found, return a default value (e.g., the agent's own position)
        return closest_position if closest_position else current_position

    def manhattan_distance(self, pos1, pos2):
        """Calculate the Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_state(self, agent_index):
        """
        Return the normalized state for a specific agent as a vector with its position, the location of B,
        its secret status, and the location of the closest opposite agent.
        """
        state = []

        # Add the agent's own position (x, y)
        state.extend(self.agent_positions[agent_index])

        # Add B's location (x, y)
        state.extend(self.b_location)

        # Add the agent's secret status (1 if it has the full secret, 0 otherwise)
        state.append(int(self.has_full_secret[agent_index]))

        # Add the location of the closest opposite agent
        closest_pos = self.get_closest_opposite_agent_location(agent_index)
        state.extend(closest_pos)

        # Convert state to a NumPy array of type float for normalization
        state = np.array(state, dtype=float)

        # Normalize position features by dividing by (grid_size - 1)
        grid_max = self.size - 1
        state[0] /= grid_max  # Agent's x position
        state[1] /= grid_max  # Agent's y position
        state[2] /= grid_max  # B's x position
        state[3] /= grid_max  # B's y position
        # state[4] is the secret status (0 or 1), no need to normalize
        state[5] /= grid_max  # Closest opposite agent's x position
        state[6] /= grid_max  # Closest opposite agent's y position

        return state



    def agent_step(self, agent_index, action):
        """Perform a step for a single agent and return its transition."""
        moved = False
        x, y = self.agent_positions[agent_index]

        if action == 0 and y > 0:  # North
            y -= 1
            moved = True
        elif action == 1 and y < self.size - 1:  # South
            y += 1
            moved = True
        elif action == 2 and x > 0:  # West
            x -= 1
            moved = True
        elif action == 3 and x < self.size - 1:  # East
            x += 1
            moved = True

        if moved:
            self.agent_positions[agent_index] = (x, y)
            self.paths[agent_index].append(self.agent_positions[agent_index])
            exchange_occurred = self.check_secret_exchange(agent_index)
            reward = self.calculate_reward(agent_index, exchange_occurred)
        else:
            reward = self.reward['invalid_move']

        done = self.check_terminal_state()
        new_state = self.get_state(agent_index)

        return new_state, reward, done

    def calculate_reward(self, agent_index, exchange_occurred=False):
        """Calculate reward for an agent based on its state and position."""
        reward = 0
        pos = self.agent_positions[agent_index]
        has_full_secret = self.has_full_secret[agent_index]

        reward += self.reward['move'] # Movement penalty

        # Check if the agent has reached B
        if pos == self.b_location:
            if has_full_secret:
                reward += self.reward['reach_b_with_full_secret']
            else:
                reward += self.reward['reach_b_without_secret']

        # Reward for secret exchange
        if exchange_occurred:
            reward += self.reward['secret_exchange']

        return reward

    
    def check_secret_exchange(self, moved_agent_index):
        """
        Check if the moved agent can exchange secrets with any opposite type agent.
        Agents with the full secret can share it with others who don't have it.
        The method returns True if the moved agent gains the full secret during the exchange.
        """
        moved_agent_position = self.agent_positions[moved_agent_index]
        moved_agent_type = self.agent_types[moved_agent_index]
        moved_agent_has_full_secret = self.has_full_secret[moved_agent_index]
        
        if moved_agent_position == self.b_location:
            # Agent cannot exchange secrets at B
            return False  # No exchange occurred

        for i in range(self.n_agents):
            if i != moved_agent_index:
                other_agent_position = self.agent_positions[i]
                other_agent_type = self.agent_types[i]
                other_agent_has_full_secret = self.has_full_secret[i]

                if (other_agent_position == moved_agent_position and
                    other_agent_type != moved_agent_type):

                    # Secret exchange can happen
                    if not moved_agent_has_full_secret:
                        # Moved agent gains the full secret
                        self.has_full_secret[moved_agent_index] = True
                        if not other_agent_has_full_secret:
                            # Both agents did not have the full secret; other agent also gains it
                            self.has_full_secret[i] = True
                        return True  # Moved agent gained the secret
                    elif not other_agent_has_full_secret:
                        # Moved agent has full secret, so other agent gains it aswell
                        self.has_full_secret[i] = True
                    # Moved agent already had full secret, so no reward
                    return False

        return False  # No exchange occurred


    def check_terminal_state(self):
        """Check if the task is completed (i.e., any agent reaches B with the full secret)."""
        for i in range(self.n_agents):
            if self.agent_positions[i] == self.b_location and self.has_full_secret[i]:
                return True
        return False

    def get_paths(self):
        """Return the paths taken by each agent."""
        return self.paths
    
    
    def get_map_array(self):
        """Generate a map array for visualization."""
        # Initialize a blank grid with zeros
        map_array = np.zeros((self.size, self.size))

        # Mark the agents on the map with their unique IDs
        for i, pos in enumerate(self.agent_positions):
            map_array[pos[1], pos[0]] = i + 1  # Use agent index + 1 as marker

        # Mark the B location with a special value (-1)
        map_array[self.b_location[1], self.b_location[0]] = -1

        return map_array
    