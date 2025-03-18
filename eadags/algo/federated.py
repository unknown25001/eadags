"""
Energy-Efficient Real-Time Scheduling of DAG Tasks
by Bhuiyan, Guo et al. (2018)
https://dl.acm.org/doi/10.1145/3241049
"""

import numpy as np

from typing import List
from queue import PriorityQueue

from ..dag import DAGTask, Schedule, Subtask, CPUs, SlicedSchedule, TimeSlice

from scipy.optimize import minimize


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


# ! deprecated
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


def slice_points_to_time_slices(points: List[float]) -> List[TimeSlice]:

    time_slices = []

    for i in range(len(points) - 1):
        time_slices.append(TimeSlice(points[i], points[i + 1]))

    return time_slices


def slice_tasks(sched: Schedule):
    
    slice_points = []
    for subtask in sched.schedule:
        slice_points.append(subtask.begin_time)
        slice_points.append(subtask.finish_time)
    slice_points = sorted(list(set(slice_points)))

    slice_point_to_slice_index = {point: i for i, point in enumerate(slice_points)}

    time_slices = slice_points_to_time_slices(slice_points)


    sliced_schedule: List[Subtask] = []
    subtask_cnt = 1

    for subtask in sched.schedule:

        local_slice_points = \
            [point for point in slice_points if point > subtask.begin_time and point < subtask.finish_time]
        slice_start = subtask.begin_time

        for slice_end in (local_slice_points + [subtask.finish_time]):
            cost_ratio = (slice_end - slice_start) / subtask.cost
            sliced_subtask = Subtask(
                node=subtask.node,
                cost=(subtask.cost * cost_ratio),
                cpu=subtask.cpu,
                begin_time=slice_start,
                finish_time=slice_end,
                in_slice=slice_point_to_slice_index[slice_start],
                subtask_name=f"N{subtask.node}.{slice_point_to_slice_index[slice_start]}"
            )
            
            sliced_schedule.append(sliced_subtask)

            subtask_cnt += 1
            slice_start = slice_end


    new_sched = SlicedSchedule(sched.task)
    new_sched.schedule = sliced_schedule
    new_sched.task = sched.task
    new_sched.slice_points = slice_points
    new_sched.time_slices = time_slices
    
    return new_sched
    

def laten_subtask_finish(sched: Schedule) -> float:

    global_finish_time = max(
        [subtask.finish_time for subtask in sched.schedule]
    )
    
    for i, subtask in enumerate(sched.schedule):
        if sched.task.succ[subtask.node]:
            succ_nodes = sched.task.succ[subtask.node]
            succ_subtasks = [subtask for subtask in sched.schedule if subtask.node in succ_nodes]
            sched.schedule[i].finish_time = min([subtask.begin_time for subtask in succ_subtasks])
        else:
            sched.schedule[i].finish_time = global_finish_time
    
    return sched


def adjust_sliced_schedule(sched: SlicedSchedule, t_arr: np.array) -> SlicedSchedule:
    
    for i, subtask in enumerate(sched.schedule):

        in_slice = subtask.in_slice

        sched.schedule[i].begin_time = sum(t_arr[:in_slice])
        sched.schedule[i].finish_time = sum(t_arr[:in_slice+1])

    for i, subtask in enumerate(sched.schedule):
        subtasks_of_node = [st for st in sched.schedule if st.node == subtask.node]
        window_of_subtasks = [st.finish_time - st.begin_time for st in subtasks_of_node]
        node_total_window = sum(window_of_subtasks)
        cost_ratio = (subtask.finish_time - subtask.begin_time) / node_total_window

        # print(f"window of N{subtask.node}: {window_of_subtasks}")
        # print(f"")

        sched.schedule[i].cost = sched.task.cost[subtask.node] * cost_ratio

    sched.slice_points = [sum(t_arr[:i+1]) for i in range(len(t_arr))]

    return sched


def optimize_energy(sched: SlicedSchedule) -> SlicedSchedule:

    time_slices = sched.time_slices

    seg_lengths = [ts.length for ts in time_slices]
    seg_lengths = np.array(seg_lengths, dtype=np.float32)

    def optm_obj(t_arr: np.array) -> float:
        tmp_sched = adjust_sliced_schedule(sched, t_arr)
        return tmp_sched.power()
    
    res = minimize(optm_obj, seg_lengths, bounds=[(0, None) for _ in seg_lengths])

    if not res.success:
        print(res)
        while True:
            pass

    t_arr = res.x

    adjusted_sched = adjust_sliced_schedule(sched, t_arr)

    return adjusted_sched


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


def schedule_federated(task: DAGTask) -> Schedule:
    
    sched = init_schedule(task)
    # decomp.show()

    sched = laten_subtask_finish(sched)
    # latened.show()

    sliced_sched = slice_tasks(sched)

    sliced_sched = optimize_energy(sliced_sched)

    sched = merge_slices(sliced_sched)

    return sched

