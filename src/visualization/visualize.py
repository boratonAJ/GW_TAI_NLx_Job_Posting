from pathlib import Path
import matplotlib.pyplot as plt


def save_basic_plot(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    ax.set_title("Sample Plot")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
