import pathlib
import subprocess
import matplotlib.pyplot as plt

from eadags.algo import federated
from eadags.dag import DAGTask, dag_from_process
from eadags.algo.federated import *

proc = subprocess.Popen(["bash", pathlib.Path("./tmp/run.sh")], stdout=subprocess.PIPE)


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

    for task in dag_from_process(proc):
        
        sliced_sched = schedule_federated_sliced(task)
        power_optm = sliced_sched.power()
        power_mrge = energy_merged(sliced_sched)

        # print("Power optm:", power_optm)
        # print("Power mrge:", power_mrge)
        print("Power saved ratio:", (power_optm - power_mrge) / power_optm)
        # print()

        
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
