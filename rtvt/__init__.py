"""Roman Target Visibility Tool package wrapper."""

__all__ = ["compute_visibility"]
__version__ = "0.1.0"


def __getattr__(name):
    if name == "compute_visibility":
        from tgt_vis import compute_visibility

        return compute_visibility
    raise AttributeError(f"module 'rtvt' has no attribute {name!r}")
