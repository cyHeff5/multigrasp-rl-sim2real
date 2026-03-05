class Curriculum:
    def next_stage(self, score: float) -> int:
        return 1 if score > 0.8 else 0
