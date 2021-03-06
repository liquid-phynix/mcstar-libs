#!/usr/bin/env python2

LIBS='/home/mcstar/src/mcstar-lib'
CXX='g++-4.8'
CCAP='35'

SRC='main.cu'
OPTS='-lineinfo -g -ccbin=%s -std=c++11 -I%s -arch=sm_%s -lcurand' % (CXX, LIBS, CCAP)

COMPILE='nvcc %s -o {output} %s' % (OPTS, SRC)

targets=[('main',)]
for cv,cn in zip([0, 1], ['nocycle', 'cycle']):
    for fv in ['float', 'double']:
        for iv,iname in zip([0, 1], ['ooplace', 'inplace']):
            targets.append(('fft_%s_%s_%s' % (cn, fv, iname), cv, fv, iv))

import os, sys
import numpy as np
from subprocess import Popen, PIPE

def build(targetindex):
    maintime = os.stat(SRC).st_mtime
    t = targets[targetindex]
    try:
        if os.stat(t[0]).st_mtime >= maintime:
            return
    except os.error: pass
    # compile it
    system=COMPILE.format(output=t[0])
    print '> compiling | %s' % system
    os.system(system)

class TestFailure(Exception):
    pass

class Tests(object):
    def __init__(self, sel):
        self.execindex = None
        for i,t in enumerate(targets):
            if t[1:] == sel:
                self.execindex = i
        if self.execindex is None:
            raise ValueError('this kind of executable isnt generated')
        build(self.execindex)
    def call_exec(self, shape, infile, outfile):
        exe = targets[self.execindex][0]
        exelist = ['./%s' % exe, str(shape[2]), str(shape[1]), str(shape[0]), infile, outfile]
        print '> calling | %s' % str(exelist)
        p = Popen(exelist, stdout=PIPE, stderr=PIPE)
        p.wait()
        print '> stdout |\n%s' % p.stdout.read()
        print '> stderr |\n%s' % p.stderr.read()
        if p.returncode != 0:
            raise TestFailure('testee\'s error status=%d' % p.returncode)

class TestA(Tests):
    # CYCLE, PREC, INPLACE
    def __init__(self, prec='float', inplace=1):
        self.prec = prec
        self.name = 'TestA'
        super(TestA, self).__init__((1, prec, inplace))
        self.files_to_unlink = []
    def __del__(self):
        #print 'to unlink: %s' % str(self.files_to_unlink)
        for f in self.files_to_unlink:
            try:
                os.unlink(f)
            except os.error: pass
    def __call__(self):
        outfile_fmt = 'outfile_%d.npy'
        l1 = [36, 20, 625, 30, 30, 27, 24, 60, 27, 10]
        l2 = [75, 20, 10, 40, 10, 36, 6, 30, 100, 18]
        l3 = [60, 45, 9, 16, 25, 27, 150, 150, 6, 75]
        shapes = zip(l1, l2, l3)
        for i,shape in enumerate(shapes):
            outfile = outfile_fmt % i
            arr = np.random.randn(*shape).astype('float32' if self.prec == 'float' else 'float64')
            np.save(infile, arr)
            os.stat('.')
            self.files_to_unlink.append(infile)

            self.call_exec(shape, infile, outfile)
            os.stat('.')
            self.files_to_unlink.append(outfile)

            arr2 = None
            try:
                arr2 = np.load(outfile)
            except os.error:
                raise TestFailure('testee\'s output file wasnt loadable')
            err = np.max(np.abs(arr - arr2 / np.prod(shape)))
            print '> difference is %.18e' % err
            maxerr = 1e-5 if self.prec == 'float' else 1e-14
            if err > maxerr:
                raise TestFailure('test concluded with too great an error')


if __name__ == '__main__':
    tests = [TestA()]

    for test in tests:
        try:
            test()
        except TestFailure as e:
            print '>> test *%s* failed with |%s|' % (test.name, e.args[0])
            sys.exit(1)
    print '> all tests passed, have a nice day :)'
    sys.exit(0)
