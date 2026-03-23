try:
    from hue_visualizer._version import __version__  # CI-injected (frozen builds)
except ImportError:
    try:
        from importlib.metadata import version

        __version__ = version("vslzr")
    except Exception:
        __version__ = "0.0.0-dev"
