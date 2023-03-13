from typing import Dict, Hashable, Mapping, Optional

from monai import transforms
from monai.config.type_definitions import KeysCollection, NdarrayOrTensor
from monai.utils.enums import PostFix, TransformBackends

DEFAULT_POST_FIX = PostFix.meta()


class ApplyWindowing(transforms.Transform):
    """
    Apply window presets to DICOM images
    Windowing adapts the greyscale component of a CT image to highlight particular structures
    by reducing the range of Hounsfield units (HU) to be displayed. Windows are usually defined by
    a width (ww), the range of HU to be considered and level (wl, the center of the window). A level of 50
    and width of 100 will thus clip all values to the range of 0 and 100.

    Args:
        window: a string for preset windows. Implemented presets are:
            brain: ww 80, wl 40
            subdural: ww 130, wl = 50
            stroke: ww 8, wl 40
            temporal bone: ww 2800, wl 700
            lungs: ww 150, wl -600
            abdomen: ww 400, wl 50
            liver: ww 150, wl 30
            bone: ww 1800, wl 400
        upper: upper threshold for windowing
        lower: lower threshold for windowing
        width: window width
        level: window level (or windo center)

    Raises:
        Either `window`, `lower`/`upper` or `width`/`level` should be specified.
        Otherwise ValueError is raised
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        window: Optional[str] = None,
        upper: Optional[int] = None,
        lower: Optional[int] = None,
        width: Optional[int] = None,
        level: Optional[int] = None,
    ):

        error_message = "Please specifiy either window or upper/lower or width/level."
        if window:
            if upper or lower:
                raise ValueError(error_message)
            if width or level:
                raise ValueError(error_message)
        elif upper and lower:
            if window:
                raise ValueError(error_message)
            if width or level:
                raise ValueError(error_message)
        elif width and level:
            if upper or lower:
                raise ValueError(error_message)
            if window:
                raise ValueError(error_message)
        else:
            raise ValueError(error_message)

        if window:
            if window == "brain":
                width, level = 80, 40
            elif window == "subdural":
                width, level = 130, 50
            elif window == "stroke":
                width, level = 8, 40
            elif window == "temporal bone":
                width, level = 2800, 700
            elif window == "lungs":
                width, level = 150, -600
            elif window == "abdomen":
                width, level = 400, 50
            elif window == "liver":
                width, level = 150, 30
            elif window == "bone":
                width, level = 1800, 400

        if width and level:
            upper = level + width // 2
            lower = level - width // 2

        self.upper = upper
        self.lower = lower

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        return img.clip(self.lower, self.upper)


class ApplyWindowingd(transforms.MapTransform):
    "Dictionary-based wrapper of :py:class:`ApplyWindowing`."

    def __init__(
        self,
        keys: KeysCollection,
        window: Optional[str] = None,
        upper: Optional[int] = None,
        lower: Optional[int] = None,
        width: Optional[int] = None,
        level: Optional[int] = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.windowing = ApplyWindowing(
            window=window, upper=upper, lower=lower, width=width, level=level
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.windowing(d[key])
        return d

