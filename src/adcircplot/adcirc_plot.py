import argparse

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

from .plotter import AdcircPlotter


def adcirc_plot() -> None:
    parser = argparse.ArgumentParser(description="Plot ADCIRC data")
    parser.add_argument("filename", type=str, help="The name of the yaml options file")
    parser.add_argument(
        "--screen",
        action="store_true",
        help="Display the plot on screen after plotting",
    )
    parser.add_argument("--animate", action="store_true", help="Animate the plot")
    parser.add_argument(
        "--slider", action="store_true", help="Add an animation slider to the plot"
    )
    args = parser.parse_args()

    plotter = AdcircPlotter(args.filename)
    plotter.plot()

    def update_plot(t):
        plotter.update_array(plotter.options()["contour"]["variable"], int(t))

    if args.slider:
        slider_axes = plt.axes([0.1, 0.05, 0.8, 0.05])
        slider = Slider(
            slider_axes, "Time", 0, plotter.n_time_steps() - 1, valinit=0, valstep=1
        )
        slider.on_changed(update_plot)
        plotter.show()
    elif args.animate:
        anim = FuncAnimation(  # noqa: F841
            plotter.figure(),
            update_plot,
            frames=plotter.n_time_steps(),
            interval=100,
        )
        plotter.show()
    else:
        if plotter.options()["output"]["filename"] is not None:
            plotter.save(plotter.options()["output"]["filename"])

        if args.screen:
            plotter.show()


if __name__ == "__main__":
    adcirc_plot()
