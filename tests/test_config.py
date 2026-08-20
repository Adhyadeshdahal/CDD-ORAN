import torch

from cdd_oran.config import load_config
from cdd_oran.experiments.train import ReplayBuffer


def test_auto_device_uses_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert load_config("env_i_cdl.yaml").device == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert load_config("env_i_cdl.yaml").device == "cpu"


def test_replay_buffer_samples_on_its_device():
    buffer = ReplayBuffer(capacity=2, state_dim=2, action_dim=3, device="cpu")
    buffer.add(torch.tensor([1.0, 2.0]), (0, 1, 2), torch.tensor([3.0, 4.0]))

    state, action, next_state = buffer.sample(3)

    assert state.device.type == action.device.type == next_state.device.type == "cpu"
    assert state.shape == (3, 2) and action.shape == (3, 3) and next_state.shape == (3, 2)
    assert buffer.data.device.type == "cpu"


def test_replay_buffer_length_is_bounded():
    buffer = ReplayBuffer(capacity=2, state_dim=1, action_dim=1, device="cpu")
    for value in range(3):
        buffer.add(torch.tensor([float(value)]), (0,), torch.tensor([float(value + 1)]))

    assert len(buffer) == 2
