from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .adcirc_file import AdcircFile
from .plot_options import read_input_file


class AdcircPlot:
    def __init__(self, configuration_file: str):
        """
        Constructor for the AdcircPlot class

        Args:
            configuration_file (str): The name of the configuration file
        """
        self.__configuration_file = configuration_file
        self.__options = read_input_file(configuration_file)
        extent = self.__options["geometry"]["extent"]
        self.__adcirc = AdcircFile(
            self.__options["contour"]["filename"],
            self.__options["geometry"]["projection"],
            extent,
        )

    def options(self) -> dict:
        """
        Return the options dictionary

        This allows modification of the options and re-plotting without
        re-reading the configuration file or re-reading the ADCIRC file

        Returns:
            dict: The options dictionary
        """
        return self.__options

    def plot(self) -> None:
        """
        Plot the mesh z elevation using matplotlib and cartopy as filled contours
        """
        fig, ax = self.__generate_plot_handles()
        self.__configure_plot_extents(ax)
        self.__add_features(ax)
        self.__plot_contours(ax)
        self.__plot_mesh(ax)
        self.__add_map_labels(ax)
        self.__plot_output()

    def __plot_output(self) -> None:
        """
        Display the plot on the screen or save it to a file

        Returns:
            None
        """
        if self.__options["output"]["screen"]:
            plt.show()
        if self.__options["output"]["filename"] is not None:
            plt.savefig(self.__options["output"]["filename"])

    def __plot_contours(self, ax: plt.Axes) -> None:
        """
        Plot the filled contours of the mesh and return the map object

        Args:
            ax: The matplotlib axes object

        Returns:
            None
        """
        variable = self.__adcirc.array(
            self.__options["contour"]["variable"],
            self.__options["contour"]["time_index"],
        )
        levels, ticks = self.__configure_colorbar(variable)
        triangulation = self.__adcirc.masked_triangulation(variable)

        contour_map = ax.tricontourf(
            triangulation,
            variable * self.__options["contour"]["scale"],
            cmap=self.__options["colorbar"]["colormap"],
            extend="both",
            levels=levels,
            alpha=self.__options["contour"]["transparency"],
        )

        self.__add_colorbar(ax, contour_map, ticks)

    def __plot_mesh(self, ax: plt.Axes) -> None:
        """
        Plot the mesh triangles on the map

        Args:
            ax: The matplotlib axes object

        Returns:
            None
        """
        if self.__options["features"]["triangles"]:
            ax.triplot(self.__adcirc.triangulation(), color="black", alpha=1.0)

    def __add_colorbar(self, ax: plt.Axes, contour_map, ticks: List[float]) -> None:
        """
        Add the colorbar to the plot

        Args:
            ax: The matplotlib axes object
            contour_map: The contour map object
            ticks: The colorbar ticks

        Returns:
            None
        """
        cbar = plt.colorbar(
            contour_map,
            ax=ax,
            orientation=self.__options["colorbar"]["orientation"],
            ticks=ticks,
            pad=0.1,
        )
        if self.__options["colorbar"]["label"] is not None:
            cbar.set_label(self.__options["colorbar"]["label"])

    def __add_map_labels(self, ax: plt.Axes) -> None:
        """
        Add the map labels to the plot

        Args:
            ax: The matplotlib axes object

        Returns:
            None
        """
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        if self.__options["features"]["title"] is not None:
            ax.set_title(self.__options["features"]["title"])
        ax.gridlines(draw_labels=True)

    def __configure_plot_extents(self, ax: plt.Axes) -> None:
        """
        Configure the plot extents based on the mesh extents

        Args:
            ax: The matplotlib axes object

        Returns:
            None
        """
        if len(self.__options["geometry"]["extent"]) == 0:
            self.__options["geometry"]["extent"] = [
                float(self.__adcirc.mesh()["x"].min()),
                float(self.__adcirc.mesh()["x"].max()),
                float(self.__adcirc.mesh()["y"].min()),
                float(self.__adcirc.mesh()["y"].max()),
            ]

        ax.set_extent(self.__options["geometry"]["extent"])

    def __generate_plot_handles(self) -> Tuple[plt.Figure, plt.Axes]:
        """
        Generate the figure and axes handles for the plot

        Returns:
            Tuple[plt.Figure, plt.Axes]: The figure and axes handles
        """
        fig = plt.figure(figsize=self.__options["geometry"]["size"])
        ax = fig.add_subplot(1, 1, 1, projection=self.__adcirc.projection())
        return fig, ax

    def __configure_colorbar(self, variable: xr.DataArray) -> Tuple[list, list]:
        """
        Configure the colorbar based on the mesh depths

        Args:
            variable: The variable to plot

        Returns:
            Tuple[list, list]: The colorbar levels and ticks
        """
        if self.__options["colorbar"]["minimum"] is None:
            cbar_min = variable.min()
        else:
            cbar_min = self.__options["colorbar"]["minimum"]
        if self.__options["colorbar"]["maximum"] is None:
            cbar_max = variable.max()
        else:
            cbar_max = self.__options["colorbar"]["maximum"]
        if self.__options["colorbar"]["ticks"] is None:
            ticks = np.linspace(
                cbar_min, cbar_max, self.__options["colorbar"]["tick_count"]
            )
            levels = np.linspace(
                cbar_min, cbar_max, self.__options["colorbar"]["contour_count"]
            )
        else:
            ticks = self.__options["colorbar"]["ticks"]
            levels = self.__options["colorbar"]["ticks"]
        return levels, ticks

    def __add_features(self, ax: plt.Axes) -> None:
        """
        Add the map features to the plot (land, ocean, coastline, borders, lakes, etc.)

        Args:
            ax: The matplotlib axes object

        Returns:
            None
        """
        import cartopy.feature as cfeature

        if self.__options["features"]["wms"] is not None:
            self.__add_basemap(ax)
        else:
            if self.__options["features"]["land"]:
                ax.add_feature(cfeature.LAND)
            if self.__options["features"]["ocean"]:
                ax.add_feature(cfeature.OCEAN)
            if self.__options["features"]["coastline"]:
                ax.add_feature(cfeature.COASTLINE)
            if self.__options["features"]["borders"]:
                ax.add_feature(cfeature.BORDERS, linestyle=":")
            if self.__options["features"]["lakes"]:
                ax.add_feature(cfeature.LAKES, alpha=0.5)

    def __add_basemap(self, ax: plt.Axes) -> None:
        """
        Add a basemap to the plot using contextily

        Args:
            ax: The matplotlib axes object

        Returns:
            None
        """
        import contextily as ctx

        if self.__options["features"]["wms"] == "streets":
            ctx.add_basemap(
                ax,
                zoom="auto",
                crs=self.__adcirc.projection(),
                source=ctx.providers.Esri.WorldTopoMap,
                attribution="Basemap: Esri",
            )
        elif self.__options["features"]["wms"] == "satellite":
            ctx.add_basemap(
                ax,
                zoom="auto",
                crs=self.__adcirc.projection(),
                source=ctx.providers.Esri.WorldImagery,
                attribution="Basemap: Esri",
            )
