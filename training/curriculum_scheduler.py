class CurriculumScheduler:
    def __init__(self, total_steps: int, initial_keep_ratio: float = 1.0):
        self.total_steps = total_steps
        self.current_step = 0
        self.initial_keep_ratio = initial_keep_ratio

    def get_keep_ratio(self):
        decay_factor = (1 - self.current_step/self.total_steps)**2
        return self.initial_keep_ratio * decay_factor

    def step(self):
        self.current_step = min(self.current_step + 1, self.total_steps)