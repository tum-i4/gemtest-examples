import abc
from typing import Tuple, Optional

import PIL.Image


class LoadImageStrategy(abc.ABC):

    def load_next_image(self) -> Optional[Tuple["PIL.Image", int]]:
        raise NotImplementedError
