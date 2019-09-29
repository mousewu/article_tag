#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""环境变量类"""

__author__ = ''

import importlib


class ConfigParser(object):

    def __call__(self, module):
        module = module.split('.')
        m = importlib.import_module('Config.' + module[0])
        return getattr(m, module[-1])


config = ConfigParser()

if __name__ == '__main__':
    data = config('stopwords.chinese')
    print(data)
