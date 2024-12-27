import matplotlib.pyplot as plt

from eadags.dag import DAGTask
from eadags.algo import schedule_federated


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
    print(task.prec)

    sched = schedule_federated(task)

    fig, (ax1, ax2) = plt.subplots(ncols=2)

    task.visualize_to(ax1)
    sched.visualize_to(ax2)

    plt.show()


if __name__ == "__main__":
    main()
