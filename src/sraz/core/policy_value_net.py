from typing import Any, Iterable
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class PolicyValueNet(ABC):
    """Abstract base class for policy-value networks used in reinforcement learning."""

    @abstractmethod
    def __init__(self):
        """Initialize the policy-value network."""
        pass

    @abstractmethod
    def train(self, examples: Iterable[tuple[Any, tuple[Any, Any]]]):
        """
        Train the network on examples.

        Args:
            examples: Training examples as tuples of (state, (policy, value)),
                where state is in the format returned by the game.
        """
        pass

    @abstractmethod
    def predict(self, state) -> tuple[Any, Any]:
        """
        Predict policy and value for a given state.

        Args:
            state: State observation in the format returned by the game.

        Returns:
            Tuple of (policy, value) predictions.
        """
        pass

    @abstractmethod
    def save_checkpoint(self, save_dir):
        """
        Save the model checkpoint to the given directory.

        Args:
            save_dir: Directory path where the checkpoint will be saved.
        """
        pass

    @abstractmethod
    def load_checkpoint(self, save_dir):
        """
        Load the model checkpoint from the given directory.

        Args:
            save_dir: Directory path from which to load the checkpoint.
        """
        pass

    @abstractmethod
    def push_multiprocessing(self):
        """
        Prepare the instance for Python multiprocessing.

        Moves tensors from GPU to CPU and returns any state that should be restored
        after multiprocessing via pop_multiprocessing().
        """
        pass

    @abstractmethod
    def pop_multiprocessing(self, *args):
        """
        Restore the instance after multiprocessing.

        Args:
            *args: State returned from push_multiprocessing() to be restored.
        """
        pass

class UniformPolicyValueNet(PolicyValueNet):
    """A net that never learns: uniform prior, constant value.

    Substituting this for a learned net turns AlphaZero into pure UCT. The
    prior contributes a constant factor to every UCB term, so selection is
    driven entirely by the Q values that search backs up, and `value` is what
    a nonterminal leaf is worth before any rollout.

    `train` is a no-op that reports zero losses: the caller's training loop
    still runs, but nothing about the returned policy or value changes. An
    "iteration" therefore batches self-play games without learning from them,
    which is exactly the null baseline a learned run must beat.

    `predict` returns a **fresh** array each call. MCTS masks the prior in
    place (`query_net_masked`), so a shared buffer would be silently corrupted
    after the first masked state.

    Pairs with rollout leaf evaluation (`rollout_n > 0`, `rollout_blend = 0.0`)
    to give net-free classic MCTS: uninformative prior, leaves valued purely by
    random rollouts.

    Has no `.model` attribute, so callers that persist weights must guard on
    `hasattr(net, "model")`.
    """

    def __init__(self, n_actions: int, value: float = 0.0):
        super().__init__()
        if n_actions <= 0:
            raise ValueError(f"n_actions must be positive, got {n_actions}")
        self.n_actions = n_actions
        self.value = float(value)

    def train(self, examples):
        """No-op. Returns the 5-tuple shape `Trainer._train_network` unpacks."""
        return None, [], [0.0], [0.0], [0.0]

    def predict(self, state):
        policy = np.full(self.n_actions, 1.0 / self.n_actions, dtype=np.float64)
        return policy, np.float64(self.value)

    def save_checkpoint(self, save_dir):
        """Nothing to save: the net is fully determined by its constructor."""
        pass

    def load_checkpoint(self, save_dir):
        pass

    def push_multiprocessing(self):
        pass

    def pop_multiprocessing(self, *args):
        pass


class TorchPolicyValueNet(PolicyValueNet):
    """
    Abstract base class for PyTorch-based policy-value networks.

    Contains a field `model: nn.Module` populated in the constructor; implements
    `save_checkpoint` and `load_checkpoint` methods that save and load this `model`.
    Users still need to implement `train` and `predict` methods to define their own
    training logic and any transformations that may need to happen before or after
    calling `model.forward()`. Functional defaults for `push_multiprocessing` and
    `pop_multiprocessing` are provided, but override these if needed for additional
    state (e.g., an optimizer).
    """

    model: nn.Module
    save_file_name: str = "model_checkpoint.pt"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, model: nn.Module):
        """
        Initialize the torch policy-value network.

        Args:
            model: The neural network model to use for predictions.
        """
        super().__init__()
        self.model = model.to(self.DEVICE)

    def save_checkpoint(self, save_dir):
        """
        Save the model checkpoint to the given directory.

        Args:
            save_dir: Directory path where the checkpoint will be saved.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), save_dir / self.save_file_name)

    def load_checkpoint(self, save_dir):
        """
        Load the model checkpoint from the given directory.

        Args:
            save_dir: Directory path from which to load the checkpoint.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        save_dir = Path(save_dir)
        save_file = save_dir / self.save_file_name
        if not save_file.exists():
            raise FileNotFoundError(f"Checkpoint file {save_file} does not exist.")
        self.model.load_state_dict(torch.load(save_file))

    def push_multiprocessing(self):
        """Prepare the model for multiprocessing by moving it to CPU."""
        self.model.to("cpu")

    def pop_multiprocessing(self, *args):
        """Restore the model after multiprocessing."""
        self.model.to(self.DEVICE)

class PolicyValueNetModel(nn.Module):
    """
    Neural network with separate policy and value heads for reinforcement learning.

    This network uses a shared body with separate heads for policy and value predictions.
    """

    def __init__(self, input_size: int, output_size: int, n_hidden_layers: int = 2,
                 hidden_size: int = 128):
        """
        Initialize the policy-value network model.

        Args:
            input_size: Dimension of the input state.
            output_size: Dimension of the output policy.
            n_hidden_layers: Number of hidden layers in the body (default: 2).
            hidden_size: Number of neurons in each hidden layer (default: 128).
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.body = nn.Sequential(
            nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU()),
            *[nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU())
              for _ in range(n_hidden_layers)],
        )
        self.policy_head = nn.Linear(hidden_size, output_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input state tensor.

        Returns:
            Tuple of (policy_logits, value) predictions.
        """
        x = self.body(x)
        policy = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return policy, value
