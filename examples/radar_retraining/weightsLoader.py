import io
import zipfile

import requests


def weightsLoader(file_url: str) -> None:
    """

    Downloads the pretrained weights for the RADDet model

    Parameters
    ----------
    file_url : str
        link to be downloaded and unzipped

    Returns
    -------

    """

    path = r"./model/RADDet/logs"

    r = requests.get(file_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path)


if __name__ == '__main__':
    # Insert the downloadable link
    link = r""
    weightsLoader(link)
