"""Camera discovery: map OpenCV camera indices to device names on macOS.

OpenCV numbers cameras in the order AVFoundation reports them (sorted by
uniqueID), which often puts an iPhone (Continuity Camera) at index 0. These
helpers let the app pick the MacBook's built-in camera by name instead.
"""
import sys


def list_cameras():
    """Return [(opencv_index, name, is_builtin)], or [] if names are unavailable."""
    if sys.platform != "darwin":
        return []
    try:
        import AVFoundation as AV
    except ImportError:
        return []

    devices = (list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo))
               + list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeMuxed)))
    # Same ordering OpenCV's AVFoundation backend uses.
    devices.sort(key=lambda d: str(d.uniqueID()))
    return [(i, str(d.localizedName()), _is_builtin(d)) for i, d in enumerate(devices)]


def _is_builtin(device):
    name = str(device.localizedName()).lower()
    model = str(device.modelID()).lower()
    if any(s in name for s in ("iphone", "continuity", "desk view")) or "iphone" in model:
        return False
    return ("facetime" in name or "built-in" in name
            or str(device.deviceType()) == "AVCaptureDeviceTypeBuiltInWideAngleCamera")


def default_camera_index():
    """Index of the built-in camera if it can be identified, else 0."""
    for index, _, builtin in list_cameras():
        if builtin:
            return index
    return 0
