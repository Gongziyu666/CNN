# -*- coding: utf-8 -*-
# @Time   : 2019/3/7 13:51
# @Author : taojiang
# @Email  : taojiang@64365.com
# @Project : CNN
# @FileName: string.py
# @Software: PyCharm

import logging

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

category = ['婚姻家庭', '拆迁安置', '债权债务', '劳动纠纷', '合同纠纷', '交通事故', '人身损害', '刑事辩护', '房产纠纷', '医疗纠纷']


def category_to_id(in_category):
    return category.index(in_category)


def id_to_category(out_id):
    return category[out_id]
