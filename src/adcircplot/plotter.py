from typing import List, Tuple, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .adcirc_file import AdcircFile
from .plot_options import read_input_file


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

    def show(self):
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
        plt.savefig(filename, tight_layout=True, dpi=300)

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
            pad=0.1,
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
        self.__ax.gridlines(draw_labels=True)

    def __configure_plot_extents(self) -> None:
        """
        Configure the plot extents based on the mesh extents

        Returns:
            None
        """
        if not self.__options["geometry"]["projection"].lower() == "orthographic":
            if len(self.__options["geometry"]["extent"]) == 0:
                self.__options["geometry"]["extent"] = [
                    float(self.__adcirc.mesh()["x"].min()),
                    float(self.__adcirc.mesh()["x"].max()),
                    float(self.__adcirc.mesh()["y"].min()),
                    float(self.__adcirc.mesh()["y"].max()),
                ]
            self.__ax.set_extent(self.__options["geometry"]["extent"])
        elif self.__options["geometry"]["projection"].lower() == "orthographic":
            if self.__options["geometry"]["global"]:
                self.__ax.set_global()

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

    def __add_features(self) -> None:
        """
        Add the map features to the plot (land, ocean, coastline, borders, lakes, etc.)

        Returns:
            None
        """
        import cartopy.feature as cfeature

        if self.__options["features"]["wms"] is not None:
            self.__add_basemap()
        else:
            if self.__options["features"]["land"]:
                self.__ax.add_feature(cfeature.LAND)
            if self.__options["features"]["ocean"]:
                self.__ax.add_feature(cfeature.OCEAN)
            if self.__options["features"]["coastline"]:
                self.__ax.add_feature(cfeature.COASTLINE)
            if self.__options["features"]["borders"]:
                self.__ax.add_feature(cfeature.BORDERS, linestyle=":")
            if self.__options["features"]["lakes"]:
                self.__ax.add_feature(cfeature.LAKES, alpha=0.5)

        if self.__options["features"]["storm_track"] is not None:
            self.__add_storm_track()

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

    def __add_storm_track(self) -> None:
        """
        Add storm track data to the plot

        Returns:
            None
        """
        if self.__options["features"]["storm_track"]["source"]["geojson"] is not None:
            for file in self.__options["features"]["storm_track"]["source"]["geojson"]:
                filename = file["file"]
                line_color = file["color"]
                line_width = file["width"]
                plot_markers = file["markers"]
                alpha = file["alpha"]
                self.__process_storm_track(
                    filename, line_color, line_width, plot_markers, alpha
                )

        if self.__options["features"]["storm_track"]["source"]["metget"] is not None:
            for track_dict in self.__options["features"]["storm_track"]["source"][
                "metget"
            ]:
                self.__process_metget_track(track_dict)

    def __process_metget_track(self, track_dict: dict) -> None:
        """
        Make a request to the MetGet A-Deck API to get the storm track data
        as geojson

        Note: You must have a MetGet API key and endpoint set in your environment
        variables as :
            - METGET_API_KEY
            - METGET_ENDPOINT

        Args:
            track_dict: The dictionary with the storm track information

        Returns:
            None
        """
        import os
        import requests

        if "METGET_API_KEY" not in os.environ:
            raise ValueError("METGET_API_KEY environment variable not set")
        else:
            api_key = os.environ["METGET_API_KEY"]

        if "METGET_ENDPOINT" not in os.environ:
            raise ValueError("METGET_ENDPOINT environment variable not set")
        else:
            endpoint = os.environ["METGET_ENDPOINT"]

        cycle = track_dict["cycle"].isoformat()
        year = track_dict["cycle"].year
        basin = track_dict["basin"]
        storm = track_dict["storm"]
        model = track_dict["model"]
        line_color = track_dict["color"]
        line_width = track_dict["width"]
        plot_markers = track_dict["markers"]
        alpha = track_dict["alpha"]

        request_url = f"{endpoint}/adeck/{year}/{basin}/{model}/{storm}/{cycle}"
        headers = {"x-api-key": api_key}
        response = requests.get(request_url, headers=headers)

        if response.status_code == 200:
            if "storm_track" in response.json()["body"]:
                storm_track = response.json()["body"]["storm_track"]
                self.__plot_track(
                    storm_track, line_color, line_width, plot_markers, alpha
                )
            elif "storm_tracks" in response.json()["body"]:
                storm_track_dict = response.json()["body"]["storm_tracks"]
                for track_id in response.json()["body"]["storm_tracks"]:
                    self.__plot_track(
                        storm_track_dict[track_id],
                        line_color,
                        line_width,
                        plot_markers,
                        alpha,
                    )

    def __process_storm_track(
        self,
        filename: str,
        line_color: str,
        line_width: float,
        plot_markers: bool,
        alpha: float,
    ) -> None:
        """
        Add a storm track to the plot using a geojson file

        Args:
            filename (str): The name of the geojson file
            line_color (str): The color of the storm track line
            line_width (float): The width of the storm track line
            plot_markers (bool): Plot the storm classification markers
            alpha (float): The transparency of the storm track line

        The minimum specification for the file is the following spec:

        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat]
                    },
                    "properties": {
                        "max_wind_speed_mph": 100
                    }
                }
            ]
        }

        Where the max_wind_speed_mph is optional

        Returns:
            None
        """
        import json

        with open(filename, "r") as f:
            storm = json.load(f)
        self.__plot_track(storm, line_color, line_width, plot_markers, alpha)

    def __plot_track(
        self,
        storm_track: dict,
        line_color: str,
        line_width: float,
        plot_markers: bool,
        alpha: float,
    ) -> None:
        """
        Plot the storm track on the map

        Args:
            storm_track: The storm track geojson
            line_color: The color of the storm track line
            line_width: Width of the storm track line
            plot_markers: Plot the storm classification markers
            alpha: The transparency of the storm track line

        Returns:
            None
        """
        from cartopy.feature import ShapelyFeature
        from shapely.geometry import LineString
        from cartopy import crs as ccrs

        has_wind_speed = (
            "max_wind_speed_mph" in storm_track["features"][0]["properties"]
        )

        line = []
        points = []
        for feature in storm_track["features"]:

            pt_x = feature["geometry"]["coordinates"][0]
            pt_y = feature["geometry"]["coordinates"][1]

            pt_xpj, pt_ypj = self.__adcirc.projection().transform_point(
                pt_x, pt_y, ccrs.PlateCarree()
            )

            line.append((pt_xpj, pt_ypj))

            wind_speed = (
                feature["properties"]["max_wind_speed_mph"]
                if "max_wind_speed_mph" in feature["properties"]
                else None
            )

            points.append(
                {
                    "coordinates": (pt_xpj, pt_ypj),
                    "storm_class": self.__get_storm_classification(wind_speed),
                }
            )

        storm_line = LineString(line)
        storm_track_feature = ShapelyFeature(
            [storm_line],
            self.__adcirc.projection(),
        )

        self.__ax.add_feature(
            storm_track_feature,
            edgecolor=line_color,
            linewidth=line_width,
            facecolor="none",
            alpha=alpha,
        )

        if has_wind_speed and plot_markers:
            for point in points:
                color = AdcircPlotter.__storm_class_to_color(point["storm_class"])
                if color:
                    self.__ax.plot(
                        point["coordinates"][0],
                        point["coordinates"][1],
                        marker="o",
                        markersize=20,
                        color=color,
                        markeredgecolor="black",
                        markeredgewidth=1.5,
                        alpha=alpha,
                    )
                    txt = self.__ax.text(
                        point["coordinates"][0],
                        point["coordinates"][1],
                        point["storm_class"],
                        color="black",
                        fontsize=12,
                        ha="center",
                        va="center",
                        clip_on=True,
                        alpha=alpha,
                    )
                    txt.clipbox = self.__ax.bbox

    @staticmethod
    def __storm_class_to_color(storm_class: str) -> Optional[str]:
        """
        Convert the storm class to a color

        TD: gray
        TS: light blue
        H1: yellow
        H2: orange
        H3: red
        H4: pink
        H5: magenta

        Args:
            storm_class (str): The storm class

        Returns:
            str: The color and symbol representation
        """
        if storm_class == "D":
            return "gray"
        elif storm_class == "S":
            return "lightblue"
        elif storm_class == "1":
            return "yellow"
        elif storm_class == "2":
            return "orange"
        elif storm_class == "3":
            return "red"
        elif storm_class == "4":
            return "pink"
        elif storm_class == "5":
            return "magenta"
        else:
            return None

    @staticmethod
    def __get_storm_classification(wind_speed: Optional[float]) -> Optional[str]:
        """
        Get the storm classification based on the wind speed

        Args:
            wind_speed (float): The wind speed in knots

        Returns:
            str: The storm classification
        """

        if wind_speed is None:
            return None

        mph_to_knots = 0.868976

        wind_speed = wind_speed * mph_to_knots

        if wind_speed < 34:
            return "D"
        elif wind_speed < 64:
            return "S"
        elif wind_speed < 83:
            return "1"
        elif wind_speed < 96:
            return "2"
        elif wind_speed < 113:
            return "3"
        elif wind_speed < 137:
            return "4"
        else:
            return "5"

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
            raise ValueError("Variable must be a string or xarray.DataArray")

        self.__contour_map = self.__ax.tricontourf(
            self.__adcirc.masked_triangulation(variable),
            variable * self.__options["contour"]["scale"],
            cmap=self.__options["colorbar"]["colormap"],
            extend="both",
            levels=self.__contour_levels,
            alpha=self.__options["contour"]["transparency"],
        )
