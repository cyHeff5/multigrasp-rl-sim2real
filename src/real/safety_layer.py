from src.common.math_utils import clip_vector


class SafetyLayer:
    def __init__(self, lower, upper, max_delta):
        self.lower = lower
        self.upper = upper
        self.max_delta = max_delta

    def sanitize_delta(self, delta_q):
        return [max(-self.max_delta, min(v, self.max_delta)) for v in delta_q]

    def enforce_limits(self, q_target):
        return clip_vector(q_target, self.lower, self.upper)
