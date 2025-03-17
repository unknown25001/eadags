"""
Energy-Efficient Real-Time Scheduling of DAG Tasks
by Bhuiyan, Guo et al. (2018)
https://dl.acm.org/doi/10.1145/3241049
"""

from typing import List
from queue import PriorityQueue

from ..dag import DAGTask, Schedule, Subtask, CPUs


def init_schedule(task: DAGTask) -> Schedule:
    """
    First stage of the method the paper introduced.
    It assumes infinite cpu,
    first assign cpu to all source node,
    then assign other nodes in lexicographic order
    according to: begin_time = latest(finish_time of preccessors)
    """

    sched = Schedule(task)

    cpus = CPUs()

    indegree = {node: len(prec) for node, prec in task.prec.items()}
    assigned = {}

    pq = PriorityQueue()
    for node, deg in indegree.items():
        if deg == 0:
            pq.put(node)

    while not pq.empty():

        node = pq.get()

        # 1. Schedule current node

        prec_finish_time = [assigned[p].finish_time for p in task.prec[node]]
        begin_time = max(prec_finish_time) if prec_finish_time else 0
        finish_time = begin_time + task.cost[node]
        cpu = cpus.alloc(begin_time)

        subtask = Subtask(
            node=node,
            cpu=cpu,
            begin_time=begin_time,
            finish_time=finish_time,
            cost=task.cost[node],
        )
        sched.add(subtask)
        cpus.assign(subtask)

        assigned[node] = subtask

        # 2. Push unassigned successors to queue

        for child in task.succ[node]:
            indegree[child] -= 1
            if indegree[child] == 0 and child not in assigned:
                pq.put(child)

    return sched


def strech_makespan(sched: Schedule) -> Schedule:
    """
    Second step:
    Strech all subtask's begin and finish time on a fixed ratio,
    making sure the task finishes exactly at deadline
    """

    makespan = max([subtask.finish_time for subtask in sched.schedule])
    deadline = sched.task.deadline

    ratio = deadline / makespan

    for subtask in sched.schedule:
        subtask.begin_time *= ratio
        subtask.finish_time *= ratio

    return sched

def slice_tasks(sched: Schedule):
    
    slice_points = []

    for subtask in 

def schedule_federated(task: DAGTask) -> Schedule:
    decomp = init_schedule(task)
    # decomp.show()
    streched = strech_makespan(decomp)
    # streched.show()
    return streched
