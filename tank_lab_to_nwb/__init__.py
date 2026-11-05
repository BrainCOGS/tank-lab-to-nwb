__all__ = ["TowersNWBConverter", "VirmenDataInterface"]


def __getattr__(name):
    # The ephys converter needs heavy optional packages; the behavior converter must import without them.
    if name == "TowersNWBConverter":
        from .convert_towers_task.towersnwbconverter import TowersNWBConverter

        return TowersNWBConverter
    if name == "VirmenDataInterface":
        from .virmen.interface import VirmenDataInterface

        return VirmenDataInterface
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
