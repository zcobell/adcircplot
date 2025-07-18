from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.feature import GSHHSFeature

from .adcirc_file import AdcircFile
from .plot_options import read_input_file
from .storm_track import StormTrack


class AdcircPlotter:
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
            extent=extent,
            projection_center=self.__options["geometry"]["projection_center"],
        )
        self.__figure = None
        self.__ax = None
        self.__contour_map = None
        self.__contour_levels = None

    def figure(self) -> plt.Figure:
        """
        Return the figure handle

        Returns:
            plt.Figure: The figure handle
        """
        return self.__figure

    def axes(self) -> plt.Axes:
        """
        Return the axes handle

        Returns:
            plt.Axes: The axes handle
        """
        return self.__ax

    def n_time_steps(self) -> int:
        """
        Return the number of time steps in the ADCIRC file

        Returns:
            int: The number of time steps
        """
        return self.__adcirc.n_time_steps(self.__options["contour"]["variable"])

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
        self.__figure, self.__ax = self.__generate_plot_handles()
        self.__configure_plot_extents()
        self.__add_features()
        self.__plot_contours()
        self.__plot_mesh()
        self.__add_map_labels()
        plt.tight_layout()

    @staticmethod
    def show():
        """
        Display the plot on the screen

        Returns:
            None
        """
        plt.show()

    def save(self, filename: str):
        """
        Save the plot to a file

        Args:
            filename (str): The name of the file to save the plot to

        Returns:
            None
        """
        plt.savefig(
            filename,
            dpi=self.__options["output"]["dpi"],
            bbox_inches="tight",
            pad_inches=0.1,
        )

    def __plot_contours(self) -> None:
        """
        Plot the filled contours of the mesh and return the map object

        Returns:
            None
        """
        variable = self.__adcirc.array(
            self.__options["contour"]["variable"],
            self.__options["contour"]["time_index"],
        )
        self.__contour_levels, ticks = self.__configure_colorbar(variable)
        self.update_array(variable, self.__options["contour"]["time_index"])
        self.__add_colorbar(ticks)

    def __plot_mesh(self) -> None:
        """
        Plot the mesh triangles on the map

        Returns:
            None
        """
        if self.__options["features"]["triangles"]:
            self.__ax.triplot(self.__adcirc.triangulation(), color="black", alpha=1.0)

    def __add_colorbar(self, ticks: List[float]) -> None:
        """
        Add the colorbar to the plot

        Args:
            ticks: The colorbar ticks

        Returns:
            None
        """
        cbar = plt.colorbar(
            self.__contour_map,
            ax=self.__ax,
            orientation=self.__options["colorbar"]["orientation"],
            ticks=ticks,
        )
        if self.__options["colorbar"]["label"] is not None:
            cbar.set_label(self.__options["colorbar"]["label"])

    def __add_map_labels(self) -> None:
        """
        Add the map labels to the plot

        Returns:
            None
        """
        self.__ax.set_xlabel("Longitude")
        self.__ax.set_ylabel("Latitude")
        if self.__options["features"]["title"] is not None:
            self.__ax.set_title(self.__options["features"]["title"])

        gl = self.__ax.gridlines(draw_labels=True)
        if not self.__options["features"]["grid"]:
            gl.xlines = False
            gl.ylines = False

    def __configure_plot_extents(self) -> None:
        """
        Configure the plot extents based on the mesh extents

        Returns:
            None
        """
        if self.__options["geometry"]["projection"].lower() != "orthographic":
            if len(self.__options["geometry"]["extent"]) == 0:
                self.__options["geometry"]["extent"] = [
                    float(self.__adcirc.mesh()["x"].min()),
                    float(self.__adcirc.mesh()["x"].max()),
                    float(self.__adcirc.mesh()["y"].min()),
                    float(self.__adcirc.mesh()["y"].max()),
                ]
            self.__ax.set_extent(self.__options["geometry"]["extent"])

            # Check if the user has specified a height and width
            if (
                self.__options["output"]["height"] is None
                or self.__options["output"]["width"] is None
            ):
                self.__set_automatic_plot_size()

        elif self.__options["geometry"]["projection"].lower() == "orthographic":
            if self.__options["geometry"]["global"]:
                self.__ax.set_global()

    def __set_automatic_plot_size(self) -> None:
        """
        Set the plot size automatically based on the aspect ratio

        Returns:
            None
        """
        width = (
            self.__options["geometry"]["extent"][1]
            - self.__options["geometry"]["extent"][0]
        )
        height = (
            self.__options["geometry"]["extent"][3]
            - self.__options["geometry"]["extent"][2]
        )
        aspect_ratio = width / height

        # Assume the default width is 12 inches
        self.__options["output"]["width"] = 12

        # Calculate the height based on the aspect ratio
        self.__options["output"]["height"] = (
            self.__options["output"]["width"] / aspect_ratio
        )

        self.__figure.set_size_inches(
            self.__options["output"]["width"],
            self.__options["output"]["height"],
        )

    def __generate_plot_handles(self) -> Tuple[plt.Figure, plt.Axes]:
        """
        Generate the figure and axes handles for the plot

        Returns:
            Tuple[plt.Figure, plt.Axes]: The figure and axes handles
        """

        if (
            self.__options["output"]["height"] is None
            or self.__options["output"]["width"] is None
        ):
            # Initialize with default values and we'll fix later
            height = 8
            width = 12
        else:
            height = self.__options["output"]["height"]
            width = self.__options["output"]["width"]

        fig = plt.figure(figsize=(width, height))
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

    def __add_features(self) -> None:
        """
        Add the map features to the plot (land, ocean, coastline, borders, lakes, etc.)

        Returns:
            None
        """
        import cartopy.feature as cfeature

        resolution = self.__options["features"]["resolution"]

        if self.__options["features"]["wms"] is not None:
            self.__add_basemap()
        else:
            if self.__options["features"]["land"]:
                self.__ax.add_feature(cfeature.LAND)
            if self.__options["features"]["ocean"]:
                self.__ax.add_feature(cfeature.OCEAN)
            if self.__options["features"]["coastline"]:
                self.__ax.add_feature(
                    cfeature.GSHHSFeature(levels=[1], scale=resolution)
                )
            if self.__options["features"]["borders"]:
                self.__ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.75)
                self.__ax.add_feature(
                    cfeature.NaturalEarthFeature(
                        category="cultural",
                        name="admin_1_states_provinces_lines",
                        facecolor="none",
                        scale="10m",
                        edgecolor="black",
                        linewidth=0.5,
                        linestyle=":",
                    )
                )
            if self.__options["features"]["lakes"]:
                self.__ax.add_feature(
                    cfeature.GSHHSFeature(levels=[2, 3, 4, 5, 6], scale=resolution),
                    zorder=0,
                    facecolor=np.array((152, 183, 226)) / 256.0,
                    edgecolor="black",
                    linewidth=0.5,
                )

        if self.__options["features"]["storm_track"] is not None:
            track_plotter = StormTrack(
                adcirc_data=self.__adcirc,
                ax=self.__ax,
                options=self.__options,
            )
            track_plotter.plot()

    def __add_basemap(self) -> None:
        """
        Add a basemap to the plot using contextily

        Returns:
            None
        """
        import contextily as ctx

        if self.__options["features"]["wms"] == "streets":
            ctx.add_basemap(
                self.__ax,
                zoom="auto",
                crs=self.__adcirc.projection(),
                source=ctx.providers.Esri.WorldTopoMap,
                attribution="Basemap: Esri",
            )
        elif self.__options["features"]["wms"] == "satellite":
            ctx.add_basemap(
                self.__ax,
                zoom="auto",
                crs=self.__adcirc.projection(),
                source=ctx.providers.Esri.WorldImagery,
                attribution="Basemap: Esri",
            )

    def update_array(self, var: Union[str, xr.DataArray], time_index: int) -> None:
        """
        Update the array to plot

        Args:
            var (Union[str, xr.DataArray]): The variable to plot
            time_index (int): The time index to plot

        Returns:
            None
        """

        if isinstance(var, xr.DataArray):
            variable = var
        elif isinstance(var, str):
            variable = self.__adcirc.array(var, time_index)
        else:
            msg = "Variable must be a string or xarray.DataArray"
            raise ValueError(msg)

        self.__contour_map = self.__ax.tricontourf(
            self.__adcirc.masked_triangulation(variable),
            variable * self.__options["contour"]["scale"],
            cmap=self.__options["colorbar"]["colormap"],
            extend="both",
            levels=self.__contour_levels,
            alpha=self.__options["contour"]["transparency"],
        )
