#!/usr/bin/env python

import re
from typing import Dict, List, Union

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import yaml

from patterns import PATTERNS


def load_yaml(filename: str) -> dict:
    """Load to a dictionary from YAML file."""
    with open(filename) as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def parse_config(filename: str = 'config.yml') -> None:
    """Get a srting of custom config for running Tesseract."""
    with open(filename) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    args = []
    for key, value in config.items():
        if value:
            if key == 'lang':
                args.append(f'-l {value}')
            elif key in ('oem', 'psm'):
                args.append(f'--{key} {value}')
            elif key in ('whitelist', 'blacklist'):
                args.append(f'-c tessedict_char_{key}={value}')
    config = ' '.join(args)
    return config


def get_params(filename: str = 'params.yml') -> Dict[str, Union[int, float]]:
    """Get a dicionary of parameters for running Tesseract."""
    params = load_yaml(filename)
    return params


class OCR:
    def __init__(self, config: str,
                 params: Dict[str, Union[int, float]]) -> None:
        self.__config = config
        self.__params = params

    @property
    def config(self) -> str:
        """Get a string of custom config when running Tesseract."""
        return self.__config

    @config.setter
    def config(self, filename: str) -> None:
        """Set a string of custom config when running Tesseract."""
        self.__config = parse_config()

    @property
    def params(self) -> Dict[str, Union[int, float]]:
        """Get a string of custom parameters when running Tesseract."""
        return self.__params

    @params.setter
    def params(self, filename: str) -> None:
        """Set a string of custom parameters when running Tesseract."""
        self.__params = get_params()

    def _get_pos(self, data: dict, i: int) -> tuple:
        """Get the location of the bounding box."""
        l, t, w, h = 'left', 'top', 'width', 'height'
        return data[l][i], data[t][i], data[w][i], data[h][i]

    def get_texts(self, img: np.array,
                  patterns: Dict[str, str]) -> Dict[str, str]:
        """Get a collection of texts extracted from the input image."""
        params = self.params
        conf = params['conf']
        color = params['bbox']['color']
        rgb = (color['r'], color['g'], color['b'])
        data = pytesseract.image_to_data(img, config=self.config,
                                         output_type=Output.DICT)
        n_bboxes = len(data['text'])
        texts = {}
        for i in range(n_bboxes):
            if int(data['conf'][i]) > conf:
                text = data['text'][i]
                for field, pattern in patterns.items():
                    if re.match(pattern, text):
                        x, y, w, h = self._get_pos(data, i)
                        img = cv2.rectangle(img, (x, y), (x + w, y + h), rgb, 2)
                        texts[field] = text
        return img, texts


if __name__ == '__main__':
    CONFIG = parse_config()
    PARAMS = get_params()
    ocr = OCR(config=CONFIG, params=PARAMS)
    img = cv2.imread('test1.jpg')
    img, texts = ocr.get_texts(img, PATTERNS)
    print(texts)
    print(type(img))
