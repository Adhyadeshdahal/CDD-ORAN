import torch

from cdd_oran.models.cdl import CDL, masked_max_from_top_two


def _make_model():
    return CDL(
        state_dim=3,
        action_dim=3,
        kpi_start=1,
        feature_fc_dims=(8, 8),
        generative_fc_dims=(8, 8),
        lr=1e-3,
        cmi_threshold=0.2,
        eval_tau=0.99,
        grad_clip=10.0,
        device="cpu",
        node_names=["p0", "k0", "k1"],
        eval_steps=2,
    )


def _assert_optimizer_states_equal(left, right):
    left_states = list(left.opt.state.values())
    right_states = list(right.opt.state.values())
    assert len(left_states) == len(right_states)
    for left_state, right_state in zip(left_states, right_states, strict=True):
        assert left_state.keys() == right_state.keys()
        for key in left_state:
            if isinstance(left_state[key], torch.Tensor):
                torch.testing.assert_close(left_state[key], right_state[key])
            else:
                assert left_state[key] == right_state[key]


def test_cdl_batched_training_and_mask_update():
    model = _make_model()
    state_pairs = torch.rand(8, 2, 3)
    actions = torch.tensor([[0.0, 1.0, 2.0]] * 8)

    loss = model.train_step(state_pairs, actions)
    model.update_mask(state_pairs, actions)

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert model.get_causal_graph().shape == (3, 4)


def test_cdl_targets_align_with_predictor_axis(monkeypatch):
    model = _make_model()
    original_nll = model._nll

    def checked_nll(mu, std, target):
        if mu.ndim == 3:
            assert target.shape == mu.shape
        else:
            assert target.shape == (mu.shape[0], 1, mu.shape[2], 1)
        return original_nll(mu, std, target)

    monkeypatch.setattr(model, "_nll", checked_nll)
    state_pairs = torch.rand(8, 2, 3)
    actions = torch.tensor([[0.0, 1.0, 2.0]] * 8)

    model.train_step(state_pairs, actions)
    model.update_mask(state_pairs, actions)


def test_top_two_pool_matches_explicit_single_source_ablation():
    torch.manual_seed(7)
    features = torch.rand(3, 4, 5, 6)
    features[:, :, 0, 0] = features[:, :, 1, 0]
    features[0, 0, 2, 1] = float("nan")
    features[1, 1, 3, 2] = float("inf")
    drop_one = torch.eye(features.shape[2], dtype=torch.bool).view(
        features.shape[2], 1, 1, features.shape[2], 1
    )
    expected = features.unsqueeze(0).masked_fill(drop_one, float("-inf")).amax(dim=3)

    actual = masked_max_from_top_two(features)

    torch.testing.assert_close(actual, expected, equal_nan=True)


def test_cdl_checkpoint_round_trip_preserves_training_and_cmi_state(tmp_path):
    torch.manual_seed(11)
    state_pairs = torch.rand(8, 2, 3)
    actions = torch.tensor([[0.0, 1.0, 2.0]] * 8)
    model = _make_model()

    torch.manual_seed(12)
    model.train_step(state_pairs, actions)
    model.update_mask(state_pairs, actions)
    assert model._eval_step_count == 1
    model.update_mask(state_pairs, actions)
    assert model._eval_step_count == 0

    checkpoint = tmp_path / "checkpoint.pt"
    model.save_model(checkpoint)
    restored = _make_model()
    restored.load_model(checkpoint)

    assert list(model.models.state_dict()) == list(restored.models.state_dict())
    for key, value in model.models.state_dict().items():
        torch.testing.assert_close(value, restored.models.state_dict()[key])
    torch.testing.assert_close(model.mask_CMI, restored.mask_CMI)
    torch.testing.assert_close(model._eval_cmi_acc, restored._eval_cmi_acc)
    assert model._eval_step_count == restored._eval_step_count
    _assert_optimizer_states_equal(model, restored)

    torch.manual_seed(13)
    left_loss = model.train_step(state_pairs, actions)
    torch.manual_seed(13)
    right_loss = restored.train_step(state_pairs, actions)
    torch.testing.assert_close(left_loss, right_loss)
    for key, value in model.models.state_dict().items():
        torch.testing.assert_close(value, restored.models.state_dict()[key])
    _assert_optimizer_states_equal(model, restored)
