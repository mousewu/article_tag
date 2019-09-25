#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""环境变量类"""

__author__ = ''

from configparser import ConfigParser
import os


class MyConfigParser(ConfigParser):
    """解决ConfigParser会将key名转成小写的问题"""

    def optionxform(self, str):
        return str


class Env(object):
    """Env主要是为了获取.env中的配置数据"""

    __basepath__ = str(os.path.dirname(os.path.dirname(__file__))) + '/'    # 配置目录

    __config_file__ = '.env'                                                # 配置文件名称

    __base__ = 'environment'                                                # 基本配置模块名

    _configs = {}

    def __init__(self):
        '''
        初始化配置类
        '''
        self._config_parser = MyConfigParser()
        self._config_parser.read(self.__basepath__ + self.__config_file__)

    def __call__(self, name, defaule=None):
        '''
        通过方法获取基本配置
        :param name: 配置名称
        :param defaule: 默认名称
        :return:
        '''
        if self._configs.get(self.__base__, None) is None:
            self._configs[self.__base__] = dict(self._config_parser.items(self.__base__))
        return self._configs[self.__base__].get(name, defaule)

    def __getattr__(self, name):
        '''
        通过属性获取基本配置
        :param name: 配置名称
        :param defaule: 默认名称
        :return:
        '''
        if self._configs.get(self.__base__, None) is None:
            self._configs[self.__base__] = dict(self._config_parser.items(self.__base__))
        return self._configs[self.__base__].get(name)

    def items(self, key):
        '''
        读取块配置
        :param key: 块名称
        :return: dict
        '''
        return dict(self._config_parser.items(key))

    @property
    def base_path(self):
        """项目根目录"""
        return self.__basepath__

    @property
    def config_path(self):
        """项目配置目录"""
        return self.__basepath__ + 'Config/'

    @property
    def django_path(self):
        """项目django目录"""
        return self.__basepath__ + 'Django/'

    @property
    def storage_path(self):
        """项目log日志目录"""
        return self.__basepath__ + 'Storage/'

    @property
    def util_path(self):
        """项目util库目录"""
        return self.__basepath__ + 'Util/'

    @property
    def resource_path(self):
        """项目util库目录"""
        return self.__basepath__ + 'Resource/'


env = Env()

if __name__ == '__main__':
    # print(4123)
    # 获取项目文件夹路径
    # env = Env()
    # print("项目根目录:", env.base_path)
    # print("log根目录:", env.storage_path)
    # #
    # # # 获取一般 .env配置
    # print('获取当前环境:', env('ENVIRONMENT'))
    # print('db host:', env('DB_HOST'))
    # print('db name:', env('DB_NAME'))
    # #
    # # # 带默认值
    # print('db host with default:', env('DB_HOST1', "127.0.0.1"))
    # print('db name with default:', env('DB_NAME1', "jeanku"))
    # #
    # # # 按属性获取
    # print('db name:', env.DB_NAME)
    # print('db host:', env.DB_HOST)
    #
    # # 不存在的key 则返回None
    # print('none key:', env.DB_NONE)
    #
    # print("mobule config", env.items('icoape'))

    # print(env.CYBEXACCOUNT)
    pass
