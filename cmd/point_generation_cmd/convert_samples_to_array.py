# Copyright 2023-2024 DreamWorks Animation LLC
# SPDX-License-Identifier: Apache-2.0

import getopt
import struct
import sys

# Python only works in doubles. We want to store these as hex floats.
def floatAsHex(f):
    return f;
    # Convert to double hex
    hexRep = f.hex()
    # Grab the sixth fractional bit
    # Representation: 0x1.123456
    # Index:          0123456789
    b = hexRep[9]

    # Make sure that that last bit is even.
    b = int(b, 16)
    if b % 2 == 1:
        b -= 1

    locationOfP = hexRep.find('p')
    return hexRep[:9] + '{0:x}'.format(b) + hexRep[locationOfP:] + 'f'

def writeHeader(f, dims, lines, arrayname):
    f.write('const std::size_t k{0}Size = {1};\n\n'.format(arrayname, lines))
    f.write('const NPoint<{0}> {1}[k{1}Size] = {{\n'.format(dims, arrayname))

def writeLine(f, data):
    f.write('     {{ {0}'.format(floatAsHex(data[0])))
    for v in data[1:]:
        f.write(', {0}'.format(floatAsHex(v)))
    f.write(' }')

def writeData(f, dims, data):
    a = []
    first = True
    for v in data():
        a.append(v)
        if len(a) == dims:
            if not first:
                f.write(',\n')
            else:
                first = False
            writeLine(f, a)
            a = []

def countValues(data):
    count = 0
    for v in data():
        count += 1
    return count

def writeFooter(f):
    f.write('\n};')

def writeFile(outputfile, dims, arrayname, reader):
    with open(outputfile, 'w') as f:
        lines = countValues(reader) / dims
        writeHeader(f, dims, lines, arrayname)
        writeData(f, dims, reader)
        writeFooter(f)

def readBinaryFile(inputfile):
    with open(inputfile, 'rb') as f:
        bin = f.read(4)
        while bin:
            val = struct.unpack('f', bin)
            yield val[0]
            bin = f.read(4)

def printHelp(scriptName):
    print('{0} -i <inputfile> -o <outputfile> -d <dimensions> -a <arrayname> [-b]'.format(scriptName))

def main(argv):
    dimensions = 2
    isBinary = False
    inputfile = ''
    outputfile = ''
    arrayname = ''
    try:
        opts, args = getopt.getopt(argv[1:], 'hi:o:bd:a:')
    except getopt.GetoptError:
        printHelp(argv[0])
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            printHelp(argv[0])
            sys.exit()
        elif opt == '-i':
            inputfile = arg
        elif opt == '-o':
            outputfile = arg
        elif opt == '-b':
            isBinary = True
        elif opt == '-d':
            dimensions = int(arg)
        elif opt == '-a':
            arrayname = arg

    if inputfile == '' or outputfile == '' or arrayname == '':
        printHelp(argv[0])
        sys.exit(2)

    print('Writing {0} in {1} dimensions.'.format(outputfile, dimensions))
    writeFile(outputfile, dimensions, arrayname, lambda : readBinaryFile(inputfile))

if __name__ == "__main__":
    main(sys.argv)
