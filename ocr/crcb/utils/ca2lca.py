#/usr/bin/python3
#-*- encoding=utf-8 -*-
'''
大写金额转为小写金额
'''
from decimal import Decimal


table={
    "整":"+0",
    "正":"+0",
    "零":"+0",
    "壹":"+1",
    "贰":"+2",
    "叁":"+3",
    "肆":"+4",
    "伍":"+5",
    "陆":"+6",
    "柒":"+7",
    "捌":"+8",
    "玖":"+9",
    "分":"*0.01",
    "角":"*0.1",
    "圆":"*1",
    "元":"*1",
    "拾":"*10",
    "十":"*10",
    "百":"*100",
    "佰":"*100",
    "千":"*1000",
    "仟":"*1000",
    "万":"*10000",
    "萬":"*10000",
    "亿":"*100000000"
}

import re
def fn(s):
    s=re.sub(r'(.+?)([亿万])',r'+(\1)\2',s)
    for k in set(s)-set('()+'):
        s=s.replace(k,table[k])
    return eval(s)

def ca2lca(value):
    if value[0] in ['万', '拾', '佰', '仟', '亿']:
        return 0
    tmp = fn(value)
    tmp = Decimal(tmp).quantize(Decimal('0.01'))
    return str(tmp)

# print(ca2lca('壹仟壹拾万元正'))