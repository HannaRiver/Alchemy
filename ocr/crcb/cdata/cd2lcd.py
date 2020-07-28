#/usr/bin/python3
#-*- coding=utf-8 -*-
'''
大写日期转为小写日期
'''

def cd2lcd(value):
    '''
    大写日期转换为数字日期
    '''
    num = ('零', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖')
    date_index = []
    re_value = ''
    for i in ['年', '月', '日']:
        if i not in value:
            return 0
        assert(i in value), "value error: " + value
        date_index.append(value.index(i))
    if date_index[2] - date_index[1] == 5 and value[date_index[1] + 1] == '零':
        return cd2lcd(value[: date_index[1] + 1] + value[date_index[1] + 2 :])
    for i, istr in enumerate(value):
        if istr in ['年', '月', '日']:
            continue
        if i < date_index[0]:
            re_value += str(num.index(istr))
        elif i < date_index[1]:
            if date_index[1] - date_index[0] == 2:
                re_value += '0'
            if istr == '拾':
                re_value += '1'
                continue
            re_value += str(num.index(istr))
        else:
            if date_index[2] - date_index[1] <= 2:
                if istr == '拾':
                    return re_value+'10'
                re_value += '0'
            if date_index[2] - date_index[1] == 4:
                if istr == '零':
                    continue
                if istr == '拾' and (value[i-2] == '零' or value[i+1] == '零'):
                    re_value += '0'
                    continue
            if istr == '拾':
                if date_index[2] - date_index[1] == 4:
                    continue
                else:
                    if date_index[1] == i - 1:
                        re_value += '1'
                        continue
                    else:
                        re_value += '0'
                        continue
            re_value += str(num.index(istr))
    
    # assert(len(re_value) == 8), "code error: " + re_value

    return re_value
            

if __name__ == '__main__':
    value = '贰零壹捌年零捌月壹拾日'
    print(cd2lcd(value))
