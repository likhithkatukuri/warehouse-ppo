import matplotlib.pyplot as plt


def show_grid(
    grid,
    start=None,
    goal=None,
    path=None
):

    plt.figure(figsize=(10,10))

    plt.imshow(
        grid,
        cmap="gray_r",
        origin="lower"
    )

    if path is not None:

        x = [p[1] for p in path]
        y = [p[0] for p in path]

        plt.plot(
            x,
            y,
            color="gold",
            linewidth=3,
            label="Path"
        )

    if start is not None:

        plt.scatter(
            start[1],
            start[0],
            c="green",
            s=120,
            marker="o",
            label="Start"
        )

    if goal is not None:

        plt.scatter(
            goal[1],
            goal[0],
            c="red",
            s=120,
            marker="X",
            label="Goal"
        )

    plt.legend()

    plt.axis("equal")

    plt.show()