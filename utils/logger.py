from __future__ import absolute_import
import numpy as np

__all__ = ['Logger', 'LoggerMonitor']

class Logger(object):
    '''Save training process to log file with simple plot function.'''

    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if title is None else title
        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {name: [] for name in self.names}

                for line in self.file:
                    numbers = line.rstrip().split('\t')
                    for i, num in enumerate(numbers):
                        self.numbers[self.names[i]].append(float(num))
                self.file.close()

                # Resume tracking based on 'Valid Top1 Acc'
                if 'Valid Top1 Acc' in self.numbers and self.numbers['Valid Top1 Acc']:
                    self.resume_epoch = int(np.argmax(self.numbers['Valid Top1 Acc']))
                    self.resume_acc = max(self.numbers['Valid Top1 Acc'])
                    self.last_epoch = int(float(self.numbers['Epoch'][-1]))
                else:
                    self.resume_epoch = 0
                    self.resume_acc = 0
                    self.last_epoch = 0

                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'a')

    def set_names(self, names):
        if not self.resume:
            self.names = names
            self.numbers = {name: [] for name in names}
            for name in names:
                self.file.write(name + '\t')
            self.file.write('\n')
            self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for idx, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num) + '\t')
            self.numbers[self.names[idx]].append(float(num))
        self.file.write('\n')
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()

class LoggerMonitor(object):
    '''Load and visualize multiple logs.'''

    def __init__(self, paths):
        '''paths is a dict: {title: filepath}'''
        self.loggers = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)
