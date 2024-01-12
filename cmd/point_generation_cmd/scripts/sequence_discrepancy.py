#!/rel/lang/python/2.7.3-6/opt-debug-icc140_64/bin/python
# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import subprocess
import sys

def average_files(directory, count):
    files = glob.glob(os.path.join(directory, '*.dat'))
    if len(files) == 0:
        return 1.0

    num_files = min(len(files), 128)

    sum = 0.0
    # Only take the first 128 files
    for file in files[:num_files]:
        command = '../discrepancy -in {0} -count {1}'.format(file, count)
        sum += float(subprocess.check_output(command, shell=True).decode("utf-8"))

    return sum / num_files

if len(sys.argv) != 2:
    print('Usage: {0} <director>'.format(sys.argv[0]))
    sys.exit(2)

directory = sys.argv[1].rstrip(r'/')

if not os.path.isdir(directory):
    print('{0} is not a directory.'.format(directory))
    sys.exit(3)

outfile = os.path.basename(directory) + '.discrepancy.dat'

with open(outfile, 'w') as f:
    for i in range(1024):
        print('File {0}/1024'.format(i+1))
        a = average_files(directory, i+1)
        f.write(str(a))
        f.write('\n')
