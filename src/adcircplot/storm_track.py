from typing import Optional

import matplotlib.pyplot as plt

from .adcirc_file import AdcircFile


class StormTrack:
    def __init__(self, adcirc_data: AdcircFile, options: dict, ax: plt.Axes):
        """
        Constructor for the StormTrack class

        Args:
            adcirc_data: The AdcircFile object
            options: Plotting options
            ax: The matplotlib axes object to plot the storm track(s) on
        """
        self.__adcirc = adcirc_data
        self.__options = options
        self.__ax = ax

    def plot(self) -> None:
        """
        Plot the storm track data on the map

        Returns:
            None
        """
        self.__add_storm_track()

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

        import warnings
        import requests

        if "METGET_API_KEY" not in os.environ:
            msg = "METGET_API_KEY environment variable not set"
            raise ValueError(msg)
        else:
            api_key = os.environ["METGET_API_KEY"]

        if "METGET_ENDPOINT" not in os.environ:
            msg = "METGET_ENDPOINT environment variable not set"
            raise ValueError(msg)
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
        else:
            msg = f"MetGet Storm Track Error: {response.status_code}: {response.text}"
            warnings.warn(msg)

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

        with open(filename) as f:
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
        from cartopy import crs as ccrs
        from cartopy.feature import ShapelyFeature
        from shapely.geometry import LineString

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
                color = StormTrack.__storm_class_to_color(point["storm_class"])
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
        storm_class_dict = {
            "D": "gray",
            "S": "lightblue",
            "1": "yellow",
            "2": "orange",
            "3": "red",
            "4": "pink",
            "5": "magenta",
        }
        if storm_class in storm_class_dict:
            return storm_class_dict[storm_class]
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
