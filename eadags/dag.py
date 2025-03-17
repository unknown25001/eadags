import json
import subprocess
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from typing import Dict, Iterable, Optional, List, Set
from dataclasses import dataclass, field


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

    def visualize_to(self, ax: plt.Axes):

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
        # ax.set_title(f"Deadline: {self.deadline}")

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


class Schedule:
    task: DAGTask
    schedule: List[Subtask]

    def __init__(self, task: DAGTask = None):
        self.task = task
        self.schedule = []

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

    def visualize_to(self, ax: plt.Axes):
        df = pd.DataFrame(self.schedule)

        xlim = 0, max([self.task.deadline, *df.finish_time])
        ax.set_xlim(xlim)

        ax.barh(
            y=df.cpu,
            width=df.finish_time - df.begin_time,
            left=df.begin_time,
            tick_label=df.node,
            # color=plt.get_cmap("RdYlGn")(np.linspace(0.15, 0.85, len(df))),
            color=self.cmap,
            height=0.5,
        )
        for _, row in df.iterrows():
            ax.text(
                x=row.begin_time + 0.5 * (row.finish_time - row.begin_time),
                y=row.cpu,
                s=f"N{int(row.node)}\n{row.cost:.2f}",
                color="black",
                ha="center",
                va="center",
            )

    def show(self):
        self.visualize_to(plt.gca())
        plt.show()


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


def dag_from_process(proc: subprocess.Popen):

    while proc.poll() is None:

        raw_succ = proc.stdout.readline()
        raw_cost = proc.stdout.readline()
        raw_priority = proc.stdout.readline()
        raw_core = proc.stdout.readline()

        succ = eval(raw_succ)
        cost = eval(raw_cost)
        priority = eval(raw_priority)
        core = eval(raw_core)

        dag = DAGTask(
            cost=cost,
            succ=succ,
            deadline=-1,
            core=core,
        )
        yield dag