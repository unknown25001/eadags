from ast import Sub
import enum
import sched
from matplotlib import stackplot
import numpy as np

from typing import List
from queue import PriorityQueue

from ..dag import (
    DAGTask,
    Schedule,
    Subtask,
    CPUs,
    SlicedSchedule,
    TimeSlice,
    merge_slices,
    alpha,
    beta,
    gamma,
    critical_freq,
)

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
        finish_time = begin_time + task.cost[node] / critical_freq()
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


def time_slices_to_slice_points(time_slices: List[TimeSlice]) -> List[float]:
    
    points = [ts.begin_time for ts in time_slices] + \
        [ts.finish_time for ts in time_slices]
    return list(set(points))


def slice_points_to_time_slices(points: List[float]) -> List[TimeSlice]:

    time_slices = []

    points_start_with_0 = points
    for i in range(len(points) - 1):
        time_slices.append(
            TimeSlice(points_start_with_0[i], points_start_with_0[i + 1])
        )

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

        local_slice_points = [
            point
            for point in slice_points
            if point > subtask.begin_time and point < subtask.finish_time
        ]
        slice_start = subtask.begin_time

        for slice_end in local_slice_points + [subtask.finish_time]:
            cost_ratio = (slice_end - slice_start) / subtask.cost
            sliced_subtask = Subtask(
                node=subtask.node,
                cost=(subtask.cost * cost_ratio),
                cpu=subtask.cpu,
                begin_time=slice_start,
                finish_time=slice_end,
                in_slice=slice_point_to_slice_index[slice_start],
                subtask_name=f"N{subtask.node}.{slice_point_to_slice_index[slice_start]}",
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


def laten_subtask_finish(sched: Schedule) -> Schedule:

    new_sched = sched.copy()

    global_finish_time = max([subtask.finish_time for subtask in sched.schedule])

    for i, subtask in enumerate(sched.schedule):
        if sched.task.succ[subtask.node]:
            succ_nodes = sched.task.succ[subtask.node]
            succ_subtasks = [
                subtask for subtask in sched.schedule if subtask.node in succ_nodes
            ]
            new_sched.schedule[i].finish_time = min(
                [subtask.begin_time for subtask in succ_subtasks]
            )
        else:
            new_sched.schedule[i].finish_time = global_finish_time
    return new_sched


def adjust_sliced_schedule(sched: SlicedSchedule, t_arr: np.array) -> SlicedSchedule:

    new_sched = sched.copy()

    for i, subtask in enumerate(sched.schedule):

        in_slice = subtask.in_slice

        new_sched.schedule[i].begin_time = sum(t_arr[:in_slice])
        new_sched.schedule[i].finish_time = sum(t_arr[: in_slice + 1])

    for i, subtask in enumerate(sched.schedule):
        subtasks_of_node = [st for st in sched.schedule if st.node == subtask.node]
        window_of_subtasks = [st.finish_time - st.begin_time for st in subtasks_of_node]
        node_total_window = sum(window_of_subtasks)
        cost_ratio = (subtask.finish_time - subtask.begin_time) / node_total_window

        # print(f"window of N{subtask.node}: {window_of_subtasks}")
        # print(f"")

        new_sched.schedule[i].cost = sched.task.cost[subtask.node] * cost_ratio

    new_sched.slice_points = [0.0] + [sum(t_arr[: i + 1]) for i in range(len(t_arr))]
    new_sched.time_slices = slice_points_to_time_slices(new_sched.slice_points)

    return new_sched


def optimize_energy(sched: SlicedSchedule) -> SlicedSchedule:

    time_slices = sched.time_slices

    seg_lengths = [ts.length for ts in time_slices]
    seg_lengths = np.array(seg_lengths, dtype=np.float32)

    def optm_obj(t_arr: np.array) -> float:
        tmp_sched = adjust_sliced_schedule(sched, t_arr)
        return tmp_sched.power()

    # res = minimize(optm_obj, seg_lengths, bounds=[(0, None) for _ in seg_lengths], options={"maxiter": 1000})
    res = minimize(
        optm_obj, seg_lengths, bounds=[(0, None) for _ in seg_lengths], options={}
    )

    if not res.success:
        print(res)
        while True:
            pass

    t_arr = res.x

    adjusted_sched = adjust_sliced_schedule(sched, t_arr)

    # verify total cost identicality
    # print(sum([st.cost for st in adjusted_sched.schedule]), sum(sched.task.cost.values()))

    return adjusted_sched


def schedule_federated_sliced(task: DAGTask) -> SlicedSchedule:

    sched = init_schedule(task)

    sched = laten_subtask_finish(sched)

    sliced_sched = slice_tasks(sched)

    sliced_sched = optimize_energy(sliced_sched)

    return sliced_sched


def schedule_federated(task: DAGTask) -> Schedule:

    sliced_sched = schedule_federated_sliced(task)

    sched = merge_slices(sliced_sched)

    return sched


def energy_merged(sched: SlicedSchedule) -> float:

    power = sched.power()
    merged_m = sched.m

    costs_in_slices = [
        [st.cost for st in sched.schedule if st.in_slice == i]
        for i, tslice in enumerate(sched.time_slices)
    ]

    sta_cost = beta * sched.makespan()
    # print(f"sta: {sta_cost}, makespan: {sched.makespan()}")

    while True:

        if merged_m < 2:
            break

        dyn_overhead = 0

        for i, tslice in enumerate(sched.time_slices):

            costs = costs_in_slices[i]

            if len(costs) < merged_m or len(costs) < 2:
                continue
            if tslice.length < 1e-6:
                continue

            costs.sort()
            cost_a, cost_b = costs[:2]

            freq_a = cost_a / tslice.length
            power_a = alpha * (freq_a**gamma)
            freq_b = cost_b / tslice.length
            power_b = alpha * (freq_b**gamma)

            freq_ab = (cost_a + cost_b) / tslice.length
            power_ab = alpha * (freq_ab**gamma)

            dyn_overhead += (power_ab - (power_a + power_b)) * tslice.length

            # print(f"costs from {costs} to {costs[2:] + [cost_a + cost_b]}")

            costs = costs[2:] + [cost_a + cost_b]
            costs_in_slices[i] = costs

        # print(dyn_overhead, sta_cost)

        if merged_m > sched.task.core or dyn_overhead < sta_cost:
            # print(f"m {merged_m + 1} -> {merged_m} ,power: {power} -> {new_power}")
            merged_m -= 1
            new_power = power + dyn_overhead - sta_cost
            power = new_power
            continue
        break

    print(f"m {sched.m} -> {merged_m}")
    return power


def merge_processor(sched: SlicedSchedule, target_num: int = None):
    pass
    # new_sched = sched.copy()

    # if target_num is None:
    #     target_num = sched.task.core

    # erased_subtasks = []

    # while True:

    #     for i, tslice in enumerate(sched.time_slices):

    #         subtasks = [st for st in new_sched.schedule if st.in_slice == i]
    #         subtasks.sort(key=lambda st: st.cost)

    #         merged_subtask = Subtask(
    #             node=None,
    #             cost=sum([st.cost for st in subtasks]),
    #             cpu=None,
    #             begin_time=tslice.begin_time,
    #             finish_time=tslice.end_time,
    #             in_slice=i,
    #         )

    #     if sched.m


def power_lbound(task: DAGTask) -> float:

    total_cost = sum([cost for cost in task.cost.values()])
    freq = critical_freq()
    makespan = total_cost / freq

    power_sta = beta * makespan
    power_dyn = alpha * (freq**gamma) * makespan

    return power_sta + power_dyn


def split_slice(sched: SlicedSchedule) -> SlicedSchedule:

    new_sched = sched.copy()

    st_in_slice = [
        [st for st in sched.schedule if st.in_slice == i]
        for i, slice in enumerate(sched.time_slices)
    ]

    n_in_slice = [len(sts) for sts in st_in_slice]

    i_slice_with_max_st = np.argmax(n_in_slice)

    # spliting
    i_tgt_slice = i_slice_with_max_st
    len_tgt_slice = new_sched.time_slices[i_tgt_slice].length

    st_to_split = [st.copy() for st in new_sched.schedule if st.in_slice == i_tgt_slice]

    st_stay = st_to_split[: len(st_to_split) // 2]
    st_move = st_to_split[len(st_to_split) // 2 :]

    # print(len(st_move), len(st_stay))
    assert len(st_move) <= len(st_stay), "不满足 move less than stay"

    new_subtasks = [
        st.copy() for st in new_sched.schedule if st.in_slice != i_tgt_slice
    ]
    new_subtasks += st_stay
    for i, st in enumerate(new_subtasks):
        if st.in_slice > i_tgt_slice:
            new_subtasks[i].in_slice += 1
            new_subtasks[i].begin_time += len_tgt_slice
            new_subtasks[i].finish_time += len_tgt_slice

    for i, st in enumerate(st_move):
        st.in_slice = i_tgt_slice + 1
        st.begin_time += len_tgt_slice
        st.finish_time += len_tgt_slice
        st.cpu = st_stay[i].cpu

        new_subtasks += [st.copy()]


    new_sched.schedule = new_subtasks
    new_sched.time_slices = \
        sched.time_slices[: i_tgt_slice] + \
        [sched.time_slices[i_tgt_slice]] + \
        sched.time_slices[i_tgt_slice :]
    new_sched.slice_points = time_slices_to_slice_points(new_sched.time_slices)

    return new_sched