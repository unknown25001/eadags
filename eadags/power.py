from dataclasses import dataclass

from .dag import Schedule


@dataclass
class PowerModel:
    alpha: float
    beta: float
    gamma: float

    def p_dynamic(self, freq):
        return self.alpha * (freq**self.gamma)

    def p_static(self):
        return self.beta

    def energy(self, sched: "Schedule"):
        energy_dynamic = 0.0
        for subtask in sched.schedule:
            duration = subtask.finish_time - subtask.begin_time
            cost = subtask.cost

            freq = cost / duration
            P = self.p_dynamic(freq)

            energy_dynamic += P * duration

        energy_static = self.p_static * sched.m * sched.deadline
        return energy_static + energy_dynamic
