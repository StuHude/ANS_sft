try:
    from .video_io import VideoReader
except ModuleNotFoundError as e:
    # Keep `third_parts.mmdet` importable even if optional video deps (e.g. opencv-python) are absent.
    _missing = str(e)

    class VideoReader:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                f"VideoReader requires optional dependencies (e.g. `cv2`). Original error: {_missing}"
            )
