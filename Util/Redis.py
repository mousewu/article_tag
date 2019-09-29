#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""json数据访问类"""

__author__ = ''

import redis
from Util.Env import env


class Redis(object):
    _instance = None

    # Redis为单例模式
    def __new__(cls, *args, **kw):
        if not cls._instance:
            pool = redis.ConnectionPool(host=env("REDIS_HOST"), port=env("REDIS_PORT"), db=env("REDIS_DBID"))
            cls._instance = redis.Redis(connection_pool=pool)
        return cls._instance


Redis = Redis()

if __name__ == '__main__':
    # 建议用实例Redis处理数据Redis.get("key1")， 而不是Redis().get("key1")

    # string demo
    print(Redis.set("key1", "value1"))
    print(Redis.get("key1"))

    # list demo
    print(Redis.lpush("key2", "value1"))
    print(Redis.lpush("key2", "value2"))
    print(Redis.lpop("key2"))
