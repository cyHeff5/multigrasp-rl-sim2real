from src.policies.wrappers import clip_action_delta


def test_action_clipping():
    out = clip_action_delta([0.2, -0.2, 0.01], 0.05)
    assert out == [0.05, -0.05, 0.01]
