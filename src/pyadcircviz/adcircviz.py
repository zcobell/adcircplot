import argparse

from .adcirc_plot import AdcircPlot


def adcirc_viz() -> None:
    parser = argparse.ArgumentParser(description="Plot ADCIRC data")
    parser.add_argument("filename", type=str, help="The name of the yaml options file")
    args = parser.parse_args()
    plotter = AdcircPlot(args.filename)
    plotter.plot()


if __name__ == "__main__":
    adcirc_viz()
