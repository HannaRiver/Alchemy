#/usr/bin/python3
#-*- coding=utf-8 -*-
"""
This script rewritten by caffe plot_training_log.py.example
support caffe verison: ctc, ssd
base tools

Author: He Na

Time: 2018.12.29
"""

import inspect
import os
import random
import sys
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.legend as lgd
import matplotlib.markers as mks


def get_log_parsing_script(script_name='parse_ctc_log.sh'):
    dirname = os.path.dirname(os.path.abspath(inspect.getfile(
        inspect.currentframe())))
    return os.path.join(dirname, script_name)

def get_log_file_suffix():
    return '.log'

def get_chart_type_description_separator():
    return '  vs. '

def is_x_axis_field(field):
    x_axis_fields = ['Iters', 'Seconds']
    return field in x_axis_fields

def create_field_index(caffe_type='ctc'):
    train_key = 'Train'
    test_key = 'Test'
    if caffe_type == 'ctc':
        field_index = {train_key:{'Iters':0, 'Seconds':1, train_key + ' loss':2,
                                  train_key + ' learning rate':3,
                                  train_key + ' accuracy':4},
                       test_key:{'Iters':0, 'Seconds':1, test_key + ' accuracy':2,
                                 test_key + ' loss':3}}
    elif caffe_type == 'ssd':
        field_index = {train_key:{'Iters':0, 'Seconds':1, train_key + ' loss':2,
                                  train_key + ' learning rate':3},
                       test_key:{'Iters':0, 'Seconds':1, test_key + ' accuracy':2,
                                  test_key + ' loss':3}}
    fields = set()
    for data_file_type in field_index.keys():
        fields = fields.union(set(field_index[data_file_type].keys()))
    fields = list(fields)
    fields.sort()
    return field_index, fields

def get_supported_chart_types(caffe_type='ssd'):
    _, fields = create_field_index(caffe_type)
    num_fields = len(fields)
    supported_chart_types = []
    for i in range(num_fields):
        if not is_x_axis_field(fields[i]):
            for j in range(num_fields):
                if i != j and is_x_axis_field(fields[j]):
                    supported_chart_types.append('%s%s%s' % (
                        fields[i], get_chart_type_description_separator(),
                        fields[j]))
    return supported_chart_types

def get_chart_type_description(chart_type):
    supported_chart_types = get_supported_chart_types(caffe_type)
    chart_type_description = supported_chart_types[chart_type]
    return chart_type_description

def get_data_file_type(chart_type):
    description = get_chart_type_description(chart_type)
    data_file_type = description.split()[0]
    return data_file_type

def get_data_file(chart_type, path_to_log):
    return (os.path.basename(path_to_log) + '.' +
            get_data_file_type(chart_type).lower())

def get_field_descriptions(chart_type):
    description = get_chart_type_description(chart_type).split(
        get_chart_type_description_separator())
    y_axis_field = description[0]
    x_axis_field = description[1]
    return x_axis_field, y_axis_field

def get_field_indices(x_axis_field, y_axis_field, caffe_type='ctc'):
    data_file_type = get_data_file_type(chart_type)
    fields = create_field_index(caffe_type)[0][data_file_type]
    return fields[x_axis_field], fields[y_axis_field]

def load_data(data_file, field_idx0, field_idx1):
    data = [[], []]
    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line[0] != '#':
                fields = line.split()
                data[0].append(float(fields[field_idx0].strip()))
                data[1].append(float(fields[field_idx1].strip()))
    return data

def random_marker():
    markers = mks.MarkerStyle.markers
    num = len(markers.values())
    idx = random.randint(0, num - 1)
    return markers.values()[idx]

def get_data_label(path_to_log):
    label = path_to_log[path_to_log.rfind('/')+1 : path_to_log.rfind(
        get_log_file_suffix())]
    return label

def get_legend_loc(chart_type):
    x_axis, y_axis = get_field_descriptions(chart_type)
    loc = 'lower right'
    if y_axis.find('accuracy') != -1:
        pass
    if y_axis.find('loss') != -1 or y_axis.find('learning rate') != -1:
        loc = 'upper right'
    return loc

def get_script_name(caffe_type):
    if caffe_type == 'ctc':
        return 'parse_ctc_log.sh'
    elif caffe_type == 'ssd':
        return 'parse_ssd_log.sh'
    else:
        print("暂不支持该caffe, 支持类型为: ctc, ssd")
        sys.exit()

def plot_chart(chart_type, path_to_png, path_to_log_list, caffe_type='ctc'):
    script_name = get_script_name(caffe_type)
    for path_to_log in path_to_log_list:
        os.system('%s %s' % (get_log_parsing_script(script_name), path_to_log))
        data_file = get_data_file(chart_type, path_to_log)
        x_axis_field, y_axis_field = get_field_descriptions(chart_type)
        x, y = get_field_indices(x_axis_field, y_axis_field, caffe_type)
        data = load_data(data_file, x, y)
        ## TODO: more systematic color cycle for lines
        color = [random.random(), random.random(), random.random()]
        label = get_data_label(path_to_log)
        linewidth = 0.75
        ## If there too many datapoints, do not use marker.
        use_marker = False
        # use_marker = True
        if not use_marker:
            plt.plot(data[0], data[1], label = label, color = color,
                     linewidth = linewidth)
        else:
            ok = False
            ## Some markers throw ValueError: Unrecognized marker style
            while not ok:
                try:
                    marker = random_marker()
                    plt.plot(data[0], data[1], label = label, color = color,
                             marker = marker, linewidth = linewidth)
                    ok = True
                except:
                    pass
        log_name = os.path.basename(path_to_log)
        os.remove(log_name + '.test')
        os.remove(log_name + '.train')
    legend_loc = get_legend_loc(chart_type)
    plt.legend(loc = legend_loc, ncol = 1) # ajust ncol to fit the space
    plt.title(get_chart_type_description(chart_type))
    plt.xlabel(x_axis_field)
    plt.ylabel(y_axis_field)
    plt.savefig(path_to_png)

def print_help():
    print("""Usage:
        ./plot lstmctc_trainning_log.py chart_type[0-%s] /where/to/save.png caffe_type /path/to/first.log ...
    Notes:
        1. Supporting multiple logs.
        2. Log file name must end with the lower-cased "%s".
        3. Save image name must end with ".png"
        4. Supported caffe types: ctc, ssd
    Supported ssd chart types:
    """ %(len(get_supported_chart_types()) - 1, get_log_file_suffix()))
    supported_chart_types = get_supported_chart_types()
    num = len(supported_chart_types)
    for i in range(num):
        print('    %d: %s' % (i, supported_chart_types[i]))

    print('\n    Supported ctc chart types: \n')
    supported_chart_types = get_supported_chart_types('ctc')
    num = len(supported_chart_types)
    for i in range(num):
        print('    %d: %s' % (i, supported_chart_types[i]))
    
    sys.exit()

def is_valid_chart_type(chart_type):
    return chart_type >= 0 and chart_type < len(get_supported_chart_types())

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print_help()
    else:
        chart_type = int(sys.argv[1])
        if not is_valid_chart_type(chart_type):
            print('%s is not a valid chart type.' % chart_type)
            print_help()
        path_to_png = sys.argv[2]
        if not path_to_png.endswith('.png'):
            print('Path must ends with png' % path_to_png)
            sys.exit()
        caffe_type = sys.argv[3]
        path_to_logs = sys.argv[4:]
        for path_to_log in path_to_logs:
            if not os.path.exists(path_to_log):
                print('Path does not exist: %s' % path_to_log)
                sys.exit()
            if not path_to_log.endswith(get_log_file_suffix()):
                print('Log file must end in %s.' % get_log_file_suffix())
                print_help()

        ## plot_chart accpets multiple path_to_logs
        plot_chart(chart_type, path_to_png, path_to_logs, caffe_type)