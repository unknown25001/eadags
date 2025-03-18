import pathlib
import subprocess
import matplotlib.pyplot as plt

from eadags.dag import DAGTask, dag_from_process
from eadags.algo import schedule_federated

# proc = subprocess.Popen(["bash", pathlib.Path("./tmp/run.sh")], stdout=subprocess.PIPE)


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
    )

    # task = DAGTask(
    #     cost={1: 1, 2: 2, 3: 2},
    #     succ={
    #         1: {2, 3},
    #     },
    # )

    # for dag in dag_from_process(proc):
    #     dag.show()

    sched = schedule_federated(task)

    # fig, (ax1, ax2) = plt.subplots(ncols=2)
    # task.visualize_to(ax1)
    # sched.visualize_to(ax2)

    fig, (ax1) = plt.subplots(ncols=1)
    # task.visualize_to(ax1)
    sched.visualize_to(ax1, label_style="time", title=f"Power: {sched.power():.2f}, Makespan: {sched.makespan():.2f}")

    plt.show()


if __name__ == "__main__":
    main()
