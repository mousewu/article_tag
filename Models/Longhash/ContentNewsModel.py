#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""model类"""

__author__ = 'jemes'

from Models.Longhash.BaseModel import BaseModel


class ContentNewsModel(BaseModel):

    __tablename__ = 'lh_content_news'  # 表名

    __create_time__ = 'create_time'  # 插入时间字段 如果该字段为None create_time则不会自动添加

    __update_time__ = 'update_time'  # 更新时间字段 如果该字段为None create_time则不会自动添加

    columns = [
        'id',                                                              # 自增ID
        'type',                                                            # 语言[0:英文 1:中文 2:日文]
        'title',                                                           # 文章标题
        'shorttitle',                                                      # 文章副标题
        'author',                                                          # 作者
        'author_id',                                                       # 作者id
        'summary',                                                         # 文章概述
        'content',                                                         # 文章内容
        'release_time',                                                    # 发布时间
        'image_key',                                                       # 图片key
        'labels',                                                          # 文章标签
        'category',                                                        # 文章所属分类
        'recommended',                                                     # 是否推荐[0:不推荐 1:推荐]
        'praise_num',                                                      # 点赞数
        'share_num',                                                       # 分享数量
        'read_num',                                                        # 文章阅读数量
        'status',                                                          # 状态[0:下线 1:上线]
        'create_time',                                                     # 创建时间
        'update_time',                                                     # 更新时间
    ]