from src.envs.reward import compute_reward


def test_reward_terms_placeholder():
    assert compute_reward({"reward": 1.0}) == 1.0
