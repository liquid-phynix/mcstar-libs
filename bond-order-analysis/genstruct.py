#!/usr/bin/env python2
from numpy import array
from numpy.random import randn
import sys
from StringIO import StringIO

divs = 10
dm1 = divs - 1

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('usage: %s <structure: BCC, FCC, SC> <noise>' % sys.argv[0])
        sys.exit(1)
    s = StringIO()
    struct = sys.argv[1]
    amp = float(sys.argv[2])
    atoms = 0
    if struct == 'BCC':
        for x in range(divs):
            for y in range(divs):
                for z in range(divs):
                    atoms += 1
                    point = array([x, y, z]) + amp * randn(3)
                    s.write('A %f %f %f\n' % tuple(point))
                    if True: #x != dm1 and y != dm1 and z != dm1:
                        atoms += 1
                        point = 0.5 + array([x, y, z]) + amp * randn(3)
                        s.write('B %f %f %f\n' % tuple(point))
    elif struct == 'FCC':
        for x in range(divs):
            for y in range(divs):
                for z in range(divs):
                    atoms += 1
                    point = array([x, y, z]) + amp * randn(3)
                    s.write('A %f %f %f\n' % tuple(point))
                    if True: #x != dm1 and y != dm1:
                        atoms += 1
                        point = array([0.5, 0.5, 0]) + array([x, y, z]) + amp * randn(3)
                        s.write('B %f %f %f\n' % tuple(point))
                    if True: #x != dm1 and z != dm1:
                        atoms += 1
                        point = array([0.5, 0, 0.5]) + array([x, y, z]) + amp * randn(3)
                        s.write('B %f %f %f\n' % tuple(point))
                    if True: #y != dm1 and z != dm1:
                        atoms += 1
                        point = array([0, 0.5, 0.5]) + array([x, y, z]) + amp * randn(3)
                        s.write('B %f %f %f\n' % tuple(point))
    elif struct == 'SC':
        for x in range(divs):
            for y in range(divs):
                for z in range(divs):
                    atoms += 1
                    point = array([x, y, z]) + amp * randn(3)
                    s.write('A %f %f %f\n' % tuple(point))

    sys.stdout.write('%d\ncomment\n' % atoms)
    sys.stdout.write(s.getvalue())

