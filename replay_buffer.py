# replay_buffer.py
"""
Simple experience replay buffer.

This is deliberately just a thin wrapper around collections.deque
so it behaves 100 % like the notebook version you used, while still
giving you a named class you can import elsewhere.

Usage
-----
    from replay_buffer import ReplayBuffer

    buffer = ReplayBuffer(maxlen=50_000)
    buffer.append((state, action, reward, next_state, done))

    # Sample a random minibatch:
    import random
    minibatch = random.sample(buffer, k=32)
"""

from collections import deque

class ReplayBuffer(deque):
    """
    Inherits everything from deque:
        append, extend, popleft, etc.

    Args
    ----
    maxlen (int):
        Maximum number of experiences to keep (older ones drop off
        automatically).  Same semantics as deque(maxlen=...).
    """
    def __init__(self, maxlen: int = 50_000):
        super().__init__(maxlen=maxlen)

    # You can add convenience helpers later, e.g.:
    # def sample(self, k):
    #     import random
    #     return random.sample(self, k)
