# -*- coding: utf-8 -*-

# 1.公共模块
import logging as logger

logger.basicConfig(level = logger.DEBUG,format = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S -',filename='log.log',filemode='a')

#设置日志的控制台输出
console = logger.StreamHandler()
console.setLevel(logger.INFO)
formatter = logger.Formatter('%(asctime)s  %(name)-6s:  %(levelname)-6s %(message)s')
console.setFormatter(formatter)
logger.getLogger('').addHandler(console)