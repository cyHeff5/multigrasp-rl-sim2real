from collections import deque


class FrameStack:
    def __init__(self, n: int):
        self.n = n
        self.buf = deque(maxlen=n)

    def reset(self, obs):
        self.buf.clear()
        for _ in range(self.n):
            self.buf.append(list(obs))
        return self.get()

    def push(self, obs):
        self.buf.append(list(obs))
        return self.get()

    def get(self):
        out = []
        for frame in self.buf:
            out.extend(frame)
        return out


def clip_action_delta(delta_q, max_delta):
    return [max(-max_delta, min(v, max_delta)) for v in delta_q]
