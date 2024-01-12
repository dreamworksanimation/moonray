#!/rel/lang/python/2.7.3-6/opt-debug-icc140_64/bin/python
# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
import sys

gnuplot_file = 'gnuplot_tmp.gp'
data_file    = 'toplot_tmp.dat'
plot_dir     = 'plot_anim'
max_points   = 1024

gnuplot_commands = '''
    unset key
    set terminal png size 650,650 enhanced font \'Helvetica,20\'
    set output \'{in_plot_dir}/output{0:04}.png\'
    set xrange [0:1]
    set yrange [0:1]
    plot \'{in_data_file}\'
'''

def usage():
    print('Usage: {0} [-rotate] <infile>'.format(sys.argv[0]))
    sys.exit(2)

def cant_read(file):
    print('Unable to read "{0}"'.format(file))
    usage()

def write_lines_to_file(lines):
    with open(data_file, 'w') as f:
        for line in lines:
            for n in line:
                f.write(str(n))
                f.write(' ')
            f.write('\n')

def write_gnuplot_config(count):
    with open(gnuplot_file, 'w') as f:
        f.write(gnuplot_commands.format(count,
                                        in_plot_dir=plot_dir,
                                        in_data_file=data_file))

def cranley_patterson_rotation(x, amount):
    x += amount;
    return x if x < 1.0 else x - 1.0

def rotate_line(line):
    n = line.split()
    for x in n:
        x = float(x)
        x = cranley_patterson_rotation(x, 0.5)
        x = string(x)
    return ' '.join(n)

rotate = False
infile = ''

if len(sys.argv) == 3:
    if sys.argv[1] != '-rotate':
        usage()
    rotate = True
    infile = sys.argv[2]
elif len(sys.argv) == 2:
    infile = sys.argv[1]
else:
    usage()

if not os.path.isfile(infile):
    cant_read(infile)

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

lines = []

with open(infile, 'r') as f:
    lines = f.readlines()

num_lines = min(len(lines), max_points)

for counter in range(num_lines):
    print(counter)

    current_lines = lines[ : counter+1]
    floats = []

    for line in current_lines:
        fs = line.split()
        line_floats = []
        for x in fs:
            if rotate:
                line_floats.append(cranley_patterson_rotation(float(x), 0.5))
            else:
                line_floats.append(float(x))
        floats.append(line_floats)

    write_lines_to_file(floats)
    write_gnuplot_config(counter)
    subprocess.call('gnuplot {0}'.format(gnuplot_file), shell=True)

subprocess.call('r2mov {0}/output*.png -combine -o points.mov'.format(plot_dir), shell=True)
shutil.rmtree(plot_dir)
