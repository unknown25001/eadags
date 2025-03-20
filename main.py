
import pandas as pd
import pathlib
import subprocess
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from eadags.algo import federated
from eadags.dag import DAGTask, dag_from_process
from eadags.algo.federated import *

proc = subprocess.Popen(["bash", pathlib.Path("./tmp/run.sh")], stdout=subprocess.PIPE)


def expr_nodenum_to_approx():

    node_num, approx = [], []
    iter = 10

    gen = dag_from_process(proc)

    fig, ax = plt.subplots()

    # for task in dag_from_process(proc):
    def fn_animation(frame):

        # iter -= 1
        # if iter < 0:
        #     break

        task = next(gen)
        
        sliced_sched = schedule_federated_sliced(task)

        power_init = init_schedule(task).power()
        power_optm = sliced_sched.power()
        power_mrge = energy_merged(sliced_sched)

        power_lbnd = power_lbound(task)
        
        optm_cost = [sum([st.cost for st in sliced_sched.schedule])]
        init_cost = sum(task.cost.values())

        node_num.append(len(task.cost))
        approx.append(power_optm / power_lbnd)
        
        print("Power optm:", power_optm)
        print("Power lbnd:", power_lbnd)
        print("Power lbnd approx:", (power_optm / power_lbnd))
        # print("Power saved ratio:", (power_optm - power_mrge) / power_optm)

        return ax.scatter(node_num, approx, color="blue", marker="x")

    ani = animation.FuncAnimation(fig, fn_animation, frames=iter, fargs=(), interval=100)
    plt.show()


def vis_pipeline(task: DAGTask):

    init = init_schedule(task)
    latened = laten_subtask_finish(init)
    sliced = slice_tasks(latened)
    optm = optimize_energy(sliced)
    merged = merge_slices(optm)

    print("E slice:", sliced.power())
    print("E optm:", optm.power())
    print("E saved ratio:", (sliced.power() - optm.power()) / sliced.power())

    process = (sliced, optm)
    names = ["sliced", "optm"]
    
    fig, axs = plt.subplots(ncols=len(process))

    for i, subtask in enumerate(process):
        subtask.visualize_to(axs[i], title=names[i])

    plt.show()


TASK_SINGLE = DAGTask(
    cost={1: 1, 2: 1},
    succ={1: {2}},
)


def main():

    task = DAGTask(
        cost={1: 4, 2: 3, 3: 3, 4: 2, 5: 2, 6: 4},
        succ={
            1: {4, 5},
            2: {3, 4},
            4: {6},
            5: {6},
        },
        deadline=12,
        core=3
    )

    # task = DAGTask(
    #     cost={1: 1, 2: 2, 3: 2},
    #     succ={
    #         1: {2, 3},
    #     },
    # )

    cfreq = critical_freq()

    for task in dag_from_process(proc):

        # task = TASK_SINGLE
        
        sliced_sched = schedule_federated_sliced(task)

        power_init = init_schedule(task).power()
        power_optm = sliced_sched.power()
        power_mrge = energy_merged(sliced_sched)

        power_lbnd = power_lbound(task)
        
        optm_cost = [sum([st.cost for st in sliced_sched.schedule])]
        init_cost = sum(task.cost.values())

        # print(f"cost: {optm_cost} from {init_cost}")

        # print("Sched init:", init_schedule(task).schedule)
        # print("Power init:", power_init)
        
        # print("Sched optm:", sliced_sched.schedule)
        # print("Power optm:", power_optm)
        print("Power optm:", power_optm)
        print("Power mrge:", power_mrge)
        print("Power lbnd:", power_lbnd)
        # print("Power saved ratio:", (power_optm - power_mrge) / power_optm)
        print("Power lbnd approx:", (power_mrge / power_lbnd))
        # print()
        
        # break
        


    # sliced_sched = schedule_federated_sliced(task)
    # power_optm = sliced_sched.power()
    # power_mrge = energy_merged(sliced_sched)

    # print("Power optm:", power_optm)
    # print("Power mrge:", power_mrge)
    # print("Power saved ratio:", (power_optm - power_mrge) / power_optm)

    # sched = schedule_federated(task)

    # fig, (ax1, ax2) = plt.subplots(ncols=2)
    # task.visualize_to(ax1)
    # sched.visualize_to(ax2)

    # fig, (ax1) = plt.subplots(ncols=1)
    # task.visualize_to(ax1)
    # sched.visualize_to(ax1, label_style="time", title=f"Power: {sched.power():.2f}, Makespan: {sched.makespan():.2f}")


if __name__ == "__main__":

    main()

    # expr_nodenum_to_approx()
