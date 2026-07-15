import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sraz.core.policy_value_net import TorchPolicyValueNet, PolicyValueNetModel
from sraz.utils import get_device


class SymRegPolicyValueNet(TorchPolicyValueNet):
    """MLP policy/value net over a one-hot encoding of the token buffer.

    Token IDs carry no ordinal structure, so each of the `state_len` buffer
    slots is one-hot encoded over the `n_tokens` vocabulary (symbols + pad).
    """

    save_file_name = "symreg_checkpoint.pt"
    default_training_params = {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "policy_weight": 1.0,
    }

    def __init__(self, state_len=15, n_tokens=12, n_actions=105, random_seed=None,
                 n_hidden_layers=2, hidden_size=128, training_params={}, device=None):
        if random_seed is not None:
            torch.manual_seed(random_seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        self.state_len = state_len
        self.n_tokens = n_tokens
        self.n_actions = n_actions
        model = PolicyValueNetModel(input_size=state_len * n_tokens, output_size=n_actions,
                                    n_hidden_layers=n_hidden_layers, hidden_size=hidden_size)
        super().__init__(model)
        self.training_params = self.default_training_params | training_params

        self.DEVICE = get_device() if device is None else device
        logging.info(f"Neural network training will occur on device '{self.DEVICE}'")

    def _encode(self, obs) -> np.ndarray:
        "One-hot encode a single token-buffer observation to a flat float32 vector."
        return np.eye(self.n_tokens, dtype=np.float32)[np.asarray(obs)].reshape(-1)

    def train(self, examples, needs_reshape=True, print_all_epochs=False):
        model = self.model
        model.to(self.DEVICE)
        tp = self.training_params
        policy_weight = tp["policy_weight"]

        criterion_value = nn.MSELoss()
        criterion_policy = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=tp["learning_rate"],
                                     weight_decay=tp["weight_decay"])

        if needs_reshape:
            states = torch.from_numpy(np.stack([self._encode(state) for state, _, _ in examples]))
            policies = torch.from_numpy(np.array([policy for _, policy, _ in examples], dtype=np.float32))
            values = torch.from_numpy(np.array([value for _, _, value in examples], dtype=np.float32))
            dataset = torch.utils.data.TensorDataset(states, policies, values)
        else:
            dataset = examples

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=tp["batch_size"], shuffle=True)

        train_batch_losses = []
        train_losses = []
        policy_losses = []
        value_losses = []

        for epoch in range(tp["epochs"]):
            model.train()
            train_loss = 0.0
            policy_loss = 0.0
            value_loss = 0.0
            for inputs, targets_policy, targets_value in train_loader:
                inputs = inputs.to(self.DEVICE)
                targets_value = targets_value.to(self.DEVICE)
                targets_policy = targets_policy.to(self.DEVICE)
                optimizer.zero_grad()
                assert inputs.shape[1] == self.model.input_size
                assert targets_policy.shape[1] == self.n_actions
                outputs_policy, outputs_value = model(inputs)
                loss_value = criterion_value(outputs_value, targets_value)
                loss_policy = criterion_policy(outputs_policy, targets_policy)
                loss = loss_value + policy_weight * loss_policy

                loss.backward()
                optimizer.step()
                loss = loss.item()
                train_batch_losses.append(loss)
                train_loss += loss
                policy_loss += loss_policy.item()
                value_loss += loss_value.item()

            train_losses.append(train_loss / len(train_loader))
            policy_losses.append(policy_loss / len(train_loader))
            value_losses.append(value_loss / len(train_loader))
            if print_all_epochs or epoch == 0 or epoch == tp["epochs"] - 1:
                logging.info(
                    f"Epoch {epoch+1}/{tp['epochs']}, Train Loss: {train_losses[-1]:.4f} "
                    f"(value: {value_losses[-1]:.4f}, policy: {policy_losses[-1]:.4f})")

        return model, train_batch_losses, train_losses, policy_losses, value_losses

    def predict(self, state):
        self.model.cpu()
        nn_input = torch.from_numpy(self._encode(state)).reshape(1, -1)
        with torch.no_grad():
            policy, value = self.model(nn_input)
            policy_prob = F.softmax(policy, dim=-1)

        policy_prob = policy_prob.numpy().squeeze(0)
        assert policy_prob.shape == (self.n_actions,)

        value = value.numpy().squeeze(0)
        assert value.shape == ()

        return policy_prob, value
