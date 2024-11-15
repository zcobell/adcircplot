import os
from datetime import datetime

import yaml
from schema import And, Optional, Or, Schema, Use

"""
Schema for the input file
"""
METGET_TRACK_SCHEMA = Schema(
    [
        {
            "model": And(str, lambda s: s.upper()),
            "storm": int,
            "cycle": Use(datetime.fromisoformat),
            Optional("basin", default="AL"): And(str, lambda s: s.upper()),
            Optional("markers", default=True): bool,
            Optional("color", default="black"): str,
            Optional("width", default=1.0): Use(float),
            Optional("alpha", default=1.0): And(Use(float), lambda a: 0.0 <= a <= 1.0),
        }
    ]
)

GEOJSON_TRACK_SCHEMA = Schema(
    [
        {
            "file": And(str, os.path.exists),
            Optional("markers", default=True): bool,
            Optional("color", default="black"): str,
            Optional("width", default=1.0): Use(float),
            Optional("alpha", default=1.0): And(Use(float), lambda a: 0.0 <= a <= 1.0),
        }
    ]
)


ADCIRC_PLOT_SCHEMA = Schema(
    {
        "contour": {
            "filename": And(str, len),
            Optional("variable", default="zeta"): And(str, len),
            Optional("time_index", default=0): Or(int, list[int]),
            Optional("scale", default=1.0): Use(float),
            Optional("transparency", default=1.0): Use(float),
        },
        "features": {
            Optional("triangles", default=False): bool,
            Optional("colorbar", default=True): bool,
            Optional("wms", default=None): str,
            Optional("land", default=True): bool,
            Optional("ocean", default=True): bool,
            Optional("coastline", default=True): bool,
            Optional("borders", default=True): bool,
            Optional("lakes", default=True): bool,
            Optional("title", default=None): str,
            Optional("grid", default=True): bool,
            Optional("feature_resolution", default="medium"): And(
                str, lambda s: s in ["low", "medium", "high"]
            ),
            Optional("storm_track", default=None): {
                "source": {
                    Optional("geojson", default=None): GEOJSON_TRACK_SCHEMA,
                    Optional("metget", default=None): METGET_TRACK_SCHEMA,
                },
            },
        },
        Optional(
            "geometry",
            default={
                "extent": [],
                "projection": "PlateCarree",
                "projection_center": None,
                "size": [16, 10],
                "global": False,
            },
        ): {
            Optional("extent", default=[]): And(
                list[Use(float)],
                lambda extent: len(extent) == 4,
            ),
            Optional("projection", default="PlateCarree"): And(
                str, lambda s: s.lower() in ["platecarree", "robinson", "orthographic"]
            ),
            Optional("projection_center", default=None): And(
                list[Use(float)], lambda center: len(center) == 2
            ),
            Optional("global", default=False): bool,
        },
        "colorbar": {
            Optional("label", default=None): str,
            Optional("orientation", default="vertical"): And(
                str, lambda s: s in ["vertical", "horizontal"]
            ),
            Optional("tick_count", default=10): And(int, lambda n: n > 0),
            Optional("ticks", default=None): [Use(float)],
            Optional("minimum", default=None): Use(float),
            Optional("maximum", default=None): Use(float),
            Optional("colormap", default="jet"): And(str, len),
            Optional("contour_count", default=20): And(int, lambda n: n > 0),
        },
        Optional("output", default={"filename": None, "width": None, "height": None}): {
            Optional("filename", default=None): And(str, len),
            Optional("dpi", default=300): And(int, lambda n: n > 0),
            Optional("width", default=None): And(int, lambda n: n > 0),
            Optional("height", default=None): And(int, lambda n: n > 0),
        },
    }
)


def read_input_file(filename: str) -> dict:
    """
    Read the input file and validate the schema

    Args:
        filename: The name of the input file

    Returns:
        The validated schema
    """
    with open(filename) as file:
        data = yaml.safe_load(file)
    return ADCIRC_PLOT_SCHEMA.validate(data)
