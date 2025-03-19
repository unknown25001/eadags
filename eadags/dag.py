import enum
import json
import math
import subprocess
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from typing import Dict, Iterable, Optional, List, Set
from dataclasses import dataclass, field


beta = 0.5
alpha = 1.7
gamma = 2.0


def critical_freq():
    return math.sqrt(alpha / beta)


@dataclass
class TimeSlice:
    begin_time: float
    finish_time: float
    length: float = field(init=False)

    def __post_init__(self):
        self.length = self.finish_time - self.begin_time

    def copy(self):
        return TimeSlice(self.begin_time, self.finish_time)


@dataclass
class DAGTask:
    cost: Dict[int, float]

    succ: Optional[Dict[int, Set[int]]] = field(default=None)
    prec: Optional[Dict[int, Set[int]]] = field(default=None)

    deadline: float = field(default=12)
    core: int = field(default=1)

    @property
    def nodes(self):
        return self.cost.keys()

    @property
    def cmap(self):
        return plt.get_cmap("tab20")(np.linspace(0.15, 0.85, len(self.nodes)))
        # return plt.get_cmap("tab20")(np.linspace(0.15, 0.85, 512))

    def __post_init__(self):
        assert self.succ or self.prec

        if self.succ:
            self.prec = {
                node: {k for k, v in self.succ.items() if node in v}
                for node in self.nodes
            }
        elif self.prec:
            self.succ = {
                node: {k for k, v in self.prec.items() if node in v}
                for node in self.nodes
            }

        for node in self.nodes:
            if node not in self.succ:
                self.succ[node] = set()
            if node not in self.prec:
                self.prec[node] = set()

        self.succ = {k: set(v) for k, v in self.succ.items()}
        self.prec = {k: set(v) for k, v in self.prec.items()}

    def visualize_to(self, ax: plt.Axes, *, title=""):

        graph = nx.DiGraph([(u, v) for u, vs in self.succ.items() for v in vs])

        for layer, nodes in enumerate(nx.topological_generations(graph)):
            for node in nodes:
                graph.nodes[node]["layer"] = layer

        pos = nx.multipartite_layout(graph, subset_key="layer")

        nx.draw_networkx(
            graph,
            pos=pos,
            ax=ax,
            nodelist=sorted(self.nodes),
            node_color=self.cmap,
            node_size=1500,
            labels={node: f"N{node}\n{self.cost[node]:.2f}" for node in self.nodes},
        )
        ax.set_title(title)

    def show(self):
        self.visualize_to(plt.gca())
        plt.show()


@dataclass
class Subtask:
    node: int
    cost: float
    cpu: int
    begin_time: float
    finish_time: float
    subtask_name: str = field(default="")
    in_slice: int = field(default=-1)

    def __post_init__(self):
        if self.subtask_name == "":
            self.subtask_name = f"N{self.node}"

    def power(self) -> float:
        if self.finish_time - self.begin_time < 1e-4:
            return 999999

        freq = self.cost / (self.finish_time - self.begin_time)
        return alpha * (freq**gamma) + beta

    def copy(self):
        return Subtask(
            node=self.node,
            cost=self.cost,
            cpu=self.cpu,
            begin_time=self.begin_time,
            finish_time=self.finish_time,
            subtask_name=self.subtask_name,
            in_slice=self.in_slice,
        )


class Schedule:
    task: DAGTask
    schedule: List[Subtask]

    def __init__(self, task: DAGTask = None):
        self.task = task
        self.schedule = []

    def copy(self):
        sched = Schedule(self.task)
        sched.schedule = [subtask.copy() for subtask in self.schedule]
        return sched

    @property
    def cmap(self):
        return np.array(
            [
                self.task.cmap[idx - 1]
                for idx in [subtask.node for subtask in self.schedule]
            ]
        )

    @property
    def m(self):
        return len(set(subtask.cpu for subtask in self.schedule))

    def add(self, subtask: Subtask):
        self.schedule.append(subtask)

    def makespan(self):
        return max(subtask.finish_time for subtask in self.schedule)

    def visualize_to(self, ax: plt.Axes, *, label_style="", cmap=None, title=""):

        if cmap is None:
            cmap = self.cmap

        df = pd.DataFrame(self.schedule)

        xlim = 0, max([sc.finish_time for sc in self.schedule])
        ax.set_xlim(xlim)

        # bars
        ax.barh(
            y=df.cpu,
            width=df.finish_time - df.begin_time,
            left=df.begin_time,
            tick_label=df.node,
            color=cmap,
            height=0.5,
        )

        # text in bars
        for _, row in df.iterrows():
            if row.finish_time - row.begin_time < 1e-1:
                continue

            label = {
                "": "",
                "cost": f"C={row.cost:.2f}",
                "time": f"T={(row.finish_time - row.begin_time):.2f}",
            }[label_style]
            ax.text(
                x=row.begin_time + 0.5 * (row.finish_time - row.begin_time),
                y=row.cpu,
                s=(f"{row.subtask_name}\n{label}"),
                color="black",
                ha="center",
                va="center",
            )

        # title
        ax.set_title(title)

    def show(self):
        self.visualize_to(plt.gca())
        plt.show()

    def power(self) -> float:

        power_dyn = 0.0
        for subtask in self.schedule:
            power_dyn += subtask.power()

        makespan = max([subtask.finish_time for subtask in self.schedule])
        cpu_num = len(set(subtask.cpu for subtask in self.schedule))
        power_sta = cpu_num * beta * makespan

        return power_dyn + power_sta


class SlicedSchedule(Schedule):

    slice_points: List[float]
    time_slices: List[TimeSlice]
    # subtask_to_node: Dict[int, int]

    def __init__(self, task=None):
        super().__init__(task)
        self.slice_points = []
        self.time_slices = []

    def copy(self):
        sched = SlicedSchedule(self.task)
        sched.schedule = [subtask.copy() for subtask in self.schedule]
        sched.slice_points = [_ for _ in self.slice_points]
        sched.time_slices = [_ for _ in self.time_slices]

        return sched

    def power(self):
        merged = merge_slices(self)
        return merged.power()
        # power_dyn = 0
        # for node in self.task.nodes:
        #     subtasks_of_node = [st for st in self.schedule if st.node == self.node]
        #     times_of_subtasks = [st.finish_time - st.begin_time for st in subtasks_of_node]
        #     time_of_node = sum(times_of_subtasks)
        #     freq = self.task.cost[node] / time_of_node

        # makespan = max([subtask.finish_time for subtask in self.schedule])
        # cpu_num = len(set(subtask.cpu for subtask in self.schedule))
        # power_sta = cpu_num * beta * makespan

    def subtasks_idx_of_node(self, node: int) -> List[int]:
        ret = []
        return ret

    def visualize_to(self, ax: plt.Axes, *, title="", cmap=None, label_style=""):

        cmap = np.array(
            [
                self.task.cmap[idx - 1]
                for idx in [subtask.node for subtask in self.schedule]
            ]
        )

        super().visualize_to(ax, label_style=label_style, cmap=cmap, title=title)

        # slices
        for slice_point in self.slice_points:
            ax.axvline(slice_point, color="black", linestyle="--", linewidth=0.5)

        # text between the vertical slice lines, indicating the gap between lines
        # for sl in self.time_slices:
        #     ax.text(
        #         x=(sl.begin_time + sl.finish_time) / 2,
        #         y=self.m + 0.5,
        #         s=f"{sl.length:.2f}",
        #         color="black",
        #         ha="center",
        #         va="center",
        #     )


@dataclass
class CPUs:
    jobs: List[List[Subtask]] = field(default_factory=list)

    def num(self):
        return len(self.jobs)

    def alloc(self, start_time):
        for i, cpu in enumerate(self.jobs):
            if cpu[-1].finish_time <= start_time:
                return i
        self.jobs.append([])
        return len(self.jobs) - 1

    def assign(self, subtask: Subtask):
        self.jobs[subtask.cpu].append(subtask)


# class FreqProfile:

#     sched: Schedule

#     def __init__(self, sched: Schedule):
#         self.sched = sched

#     def visualize_to(self, ax: plt.Axes, *, show_cost=False):

#         df = pd.DataFrame(self.sched.schedule)

#         xlim = 0, max([self.task.deadline, *df.finish_time])
#         ax.set_xlim(xlim)

#         # bars
#         # cmap =

#         # TODO:

#         ax.barh(
#             y=df.cpu,
#             width=df.finish_time - df.begin_time,
#             left=df.begin_time,
#             tick_label=df.node,
#             color=cmap,
#             height=0.5,
#         )

#         # text in bars
#         for _, row in df.iterrows():
#             ax.text(
#                 x=row.begin_time + 0.5 * (row.finish_time - row.begin_time),
#                 y=row.cpu,
#                 s=(
#                     f"{row.subtask_name}\n{row.cost:.2f}"
#                     if show_cost
#                     else f"N{int(row.subtask_name)}"
#                 ),
#                 color="black",
#                 ha="center",
#                 va="center",
#             )

#     def show(self):
#         self.visualize_to(plt.gca())
#         plt.show()


def dag_from_process(proc: subprocess.Popen):

    while proc.poll() is None:

        raw_succ = proc.stdout.readline()
        raw_cost = proc.stdout.readline()
        raw_priority = proc.stdout.readline()
        raw_core = proc.stdout.readline()

        # print(raw_cost)
        # print(raw_succ)

        succ = eval(raw_succ)
        cost = eval(raw_cost)
        priority = eval(raw_priority)
        core = eval(raw_core)

        # for k in cost.keys():
        #     cost[k] = float(int(cost[k]) // 10)

        dag = DAGTask(
            cost=cost,
            succ=succ,
            deadline=-1,
            core=core,
        )
        yield dag


def merge_slices(sched: SlicedSchedule) -> Schedule:

    merged_sched = Schedule(task=sched.task)

    for node in sched.task.nodes:

        subtasks_of_node = [st for st in sched.schedule if st.node == node]

        merged_subtask = Subtask(
            node=node,
            cost=sum([st.cost for st in subtasks_of_node]),
            cpu=subtasks_of_node[0].cpu,
            begin_time=min([st.begin_time for st in subtasks_of_node]),
            finish_time=max([st.finish_time for st in subtasks_of_node]),
            subtask_name=f"N{node}",
        )
        merged_sched.schedule.append(merged_subtask)

    return merged_sched

