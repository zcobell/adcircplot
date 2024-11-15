import yaml
from schema import And, Optional, Or, Schema, Use

"""
Schema for the input file
"""
ADCIRC_VIZ_SCHEMA = Schema(
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
        },
        Optional(
            "geometry",
            default={
                "extent": [],
                "projection": None,
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
            Optional("size", default=[10, 10]): And(
                list[Use(float)],
                lambda size: len(size) == 2,
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
        Optional("output", default={"filename": None, "screen": False}): {
            Optional("filename", default=None): And(str, len),
            Optional("screen", default=False): bool,
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
    return ADCIRC_VIZ_SCHEMA.validate(data)
