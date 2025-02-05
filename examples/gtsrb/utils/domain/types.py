import typing

LabelledExample = typing.Tuple["PIL.Image", int]
DataIterator = typing.Generator[LabelledExample, None, None]
