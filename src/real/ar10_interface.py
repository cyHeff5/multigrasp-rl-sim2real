class AR10Interface:
    """Adapter for real hand IO: read q_measured, send q_target."""

    def read_q_measured(self):
        return []

    def send_q_target(self, q_target):
        return None
