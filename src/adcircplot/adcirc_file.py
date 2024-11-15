from typing import List

import cartopy.crs as ccrs
import numpy as np
import xarray as xr
from matplotlib.tri import Triangulation


class AdcircFile:
    """
    A class to read and manipulate ADCIRC files
    """

    def __init__(
        self,
        filename: str,
        proj_str: str,
        extent: List[float],
        projection_center: List[float],
    ):
        """
        Constructor for the AdcircFile class

        Args:
            filename: The name of the ADCIRC file
            proj_str: The projection string to use (default is PlateCarree)
            extent: The extent of the plot (xmin, xmax, ymin, ymax) used to mask the data
        """
        self.__filename = filename
        self.__projection = self.__projection_from_string(proj_str, projection_center)
        self.__adcirc_file = xr.open_dataset(filename)
        self.__mesh = self.__read_mesh()
        self.__triangulation = self.__generate_triangulation(extent)

    @staticmethod
    def __projection_from_string(proj_str: str, center: List[float]) -> ccrs.Projection:
        """
        Convert a string to a cartopy projection object

        Args:
            proj_str: The projection string

        Returns:
            ccrs.Projection: The cartopy projection object
        """
        proj_str = proj_str.lower()
        if proj_str is None or proj_str == "platecarree":
            return ccrs.PlateCarree()
        elif proj_str == "robinson":
            return ccrs.Robinson()
        elif proj_str == "orthographic":
            if center is None:
                return ccrs.Orthographic()
            else:
                return ccrs.Orthographic(
                    central_longitude=center[0], central_latitude=center[1]
                )
        else:
            msg = f"Projection {proj_str} not recognized"
            raise ValueError(msg)

    def projection(self) -> ccrs.Projection:
        """
        Return the cartopy projection object

        Returns:
            ccrs.Projection: The cartopy projection object
        """
        return self.__projection

    def n_time_steps(self, variable_name: str) -> int:
        """
        Return the number of time steps in the ADCIRC file

        Args:
            variable_name: The name of the variable

        Returns:
            int: The number of time steps
        """
        return self.__adcirc_file[variable_name].shape[0]

    def __read_mesh(self) -> xr.Dataset:
        """
        Read the ADCIRC file and place the resulting data into a xarray Dataset

        Returns:
            xr.Dataset: The mesh data
        """
        return xr.Dataset(
            {
                "x": (["node"], self.__adcirc_file["x"].to_numpy()),
                "y": (["node"], self.__adcirc_file["y"].to_numpy()),
                "depth": (["node"], self.__adcirc_file["depth"].to_numpy()),
                "element": (["element", 3], self.__adcirc_file["element"].to_numpy()),
            }
        )

    def __generate_triangulation(
        self,
        extent: List[float],
    ) -> Triangulation:
        """
        Generate a matplotlib Triangulation object from the mesh data.

        This method will mask the data so matplotlib can correctly plot the
        triangulation.

        Args:
            extent: The extent of the plot (xmin, xmax, ymin, ymax) used to
                mask the data for plotting in PlateCarree projection

        Returns:
            Triangulation: A matplotlib Triangulation object
        """
        if isinstance(self.__projection, ccrs.PlateCarree):
            return self.__generate_triangulation_plate(extent)
        elif isinstance(self.__projection, ccrs.Orthographic):
            return self.__generate_triangulation_orthographic()
        elif isinstance(self.__projection, ccrs.Robinson):
            return self.__generate_triangulation_robinson()
        else:
            msg = "Projection not recognized"
            raise ValueError(msg)

    def __generate_triangulation_plate(self, extent: List[float]):
        if len(extent) == 4:
            buffer_x = 0.1 * (extent[1] - extent[0])
            buffer_y = 0.1 * (extent[3] - extent[2])
            node_mask = (
                (self.__mesh["x"] >= extent[0] - buffer_x)
                & (self.__mesh["x"] <= extent[1] + buffer_x)
                & (self.__mesh["y"] >= extent[2] - buffer_y)
                & (self.__mesh["y"] <= extent[3] + buffer_y)
            )
            mask = ~np.all(node_mask[self.__mesh["element"] - 1], axis=1)
            return Triangulation(
                self.__mesh["x"],
                self.__mesh["y"],
                self.__mesh["element"] - 1,
                mask=mask,
            )
        else:
            mask = None

        return Triangulation(
            self.__mesh["x"],
            self.__mesh["y"],
            self.__mesh["element"] - 1,
            mask=mask,
        )

    def __generate_triangulation_orthographic(
        self,
    ) -> Triangulation:
        """
        Generate the triangulation for an orthographic projection

        Returns:
            Triangulation: The triangulation object
        """
        x, y = self.__projection.transform_points(
            ccrs.PlateCarree(), self.__mesh["x"], self.__mesh["y"]
        )[:, :2].T
        self.__mesh["x_pj"] = x
        self.__mesh["y_pj"] = y

        node_mask = np.isnan(x) | np.isnan(y)
        mask = np.any(node_mask[self.__mesh["element"] - 1], axis=1)

        return Triangulation(
            self.__mesh["x_pj"],
            self.__mesh["y_pj"],
            self.__mesh["element"] - 1,
            mask=mask,
        )

    def __generate_triangulation_robinson(self) -> Triangulation:
        """
        Generate a matplotlib Triangulation object from the mesh data.

        This method will mask the data so matplotlib can correctly plot the
        triangulation. Matplotlib will break down when plotting triangles
        which wrap around the globe, so a check on the triangle with nodes
        that cross the wrapping point is performed.

        Returns:
            Triangulation: A matplotlib Triangulation object
        """
        mask = ~np.all(
            np.abs(
                np.diff(
                    self.__mesh["x"][self.__mesh["element"] - 1].to_numpy(),
                    axis=1,
                )
            )
            < 90,
            axis=1,
        )

        x, y = self.__projection.transform_points(
            ccrs.PlateCarree(), self.__mesh["x"], self.__mesh["y"]
        )[:, :2].T

        self.__mesh["x_pj"] = x
        self.__mesh["y_pj"] = y

        return Triangulation(
            self.__mesh["x_pj"],
            self.__mesh["y_pj"],
            self.__mesh["element"] - 1,
            mask=mask,
        )

    def __repr__(self) -> str:
        """
        Return a string representation of the ADCIRC file object

        Returns:
            str: A string representation of the ADCIRC file object
        """
        return f"Mesh({self.__filename}, vertices={self.__triangulation.x.shape[0]}, triangles={self.__triangulation.triangles.shape[0]})"

    def mesh(self) -> xr.Dataset:
        """
        Return the mesh data as a xarray Dataset

        Returns:
            xr.Dataset: The mesh data
        """
        return self.__mesh

    def triangulation(self) -> Triangulation:
        """
        Return the base triangulation object

        Returns:
            Triangulation: The base triangulation object
        """
        return self.__triangulation

    def masked_triangulation(self, scalar: xr.DataArray) -> Triangulation:
        """
        Mask the pre-computed triangulation with a variable

        This method will mask the triangulation based on the scalar values in
        addition to the triangle connectivity.

        Args:
            scalar: The scalar values to use for masking

        Returns:
            Triangulation: The masked triangulation object
        """
        import copy

        mask = self.__triangulation.mask
        t = copy.deepcopy(self.__triangulation)

        if mask is None:
            mask = np.full(t.triangles.shape[0], False)

        # Mask the triangulation based on the existing triangle connectivity mask
        # and the scalar values that are nan or < -9999
        new_mask = mask | ~np.isfinite(scalar.to_numpy()[t.triangles].min(axis=1))

        t.set_mask(new_mask)

        return t

    def array(self, variable_name: str, time_index: int) -> xr.DataArray:
        """
        Return the dataset for a given variable and time index

        Args:
            variable_name: The name of the variable
            time_index: The time index

        Returns:
            xr.DataArray: The data array for the variable at the given time index
        """
        if variable_name not in self.__adcirc_file:
            msg = f"Variable {variable_name} not found in the file"
            raise ValueError(msg)

        if len(self.__adcirc_file[variable_name].shape) == 1:
            return self.__adcirc_file[variable_name][:]
        else:
            return self.__adcirc_file[variable_name][time_index, :]
