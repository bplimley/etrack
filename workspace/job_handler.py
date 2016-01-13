#!/usr/bin/python

# Handle file processing jobs.

import os
import glob
import time
import multiprocessing
import ipdb as pdb
import types


class JobHandler(object):
    """
    Create an object representing a file-by-file processing job.

    Required inputs:
      work_function: function to process one file. Should take as inputs:
        ...
      loadglob

    Possible input arguments:
      in_place_flag: operate on files in-place. (default: False)
        in other words, the loadfile is opened in r+ mode, and there is no
        savefile.
      phflag: create placeholder files. (default: True)
        if phpath/phglob are not given, then phpath = savepath and
        phglob = 'ph_' + saveglob
      doneflag: create done-files. (default: False)
        if donepath/doneglob are not given, then donepath = savepath and
        doneglob = 'done_' + saveglob
      verbosity: default verbosity, passed to work_function (int) (default: 1)
      dry_run: default state of doing a dry-run (bool) (default: False)
      n_threads: default number of processes to run (multiprocessing) (int)
        (default: 1)

      loadpath, loadglob: location to load data from. (default loadpath: '')
      savepath, saveglob: location to save new results file to
      phpath, phglob: location of placeholder file to indicate in-progress
      donepath, doneglob: location of done-file to confirm completion of a file

    Notes:
      loadpath/loadglob are required.
      savepath/saveglob are not required, but if doneflag is False then these
        are needed for output checking.
        (savepath/saveglob are irrelevant if in_place_flag is True.)
      doneflag becomes True if donepath/doneglob are provided.
    """

    def __init__(self, work_function=None, **kwargs):
                #  in_place_flag=False, phflag=False, doneflag=False,
                #  loadpath=None, savepath=None, phpath=None, donepath=None,
                #  loadglob=None, saveglob=None, phglob=None, doneglob=None,
                #  verbosity=1, dry_run=False, n_threads=None):
        """
        Initialize a job handler.
        """

        self.input_handling(work_function, **kwargs)

    def input_handling(self, work_function, **kwargs):
        """
        Check input args and write values into self.
        """

        # work_function (required)
        if not isinstance(work_function, types.FunctionType):
            raise RuntimeError('JobHandler requires a valid work function')
        self.do_work = work_function

        # in_place_flag (defaults to False)
        if 'in_place_flag' in kwargs:
            self.in_place_flag = bool(kwargs['in_place_flag'])
        else:
            self.in_place_flag = False

        # phflag (defaults to True)
        if 'phflag' in kwargs:
            self.phflag = bool(kwargs['phflag'])
        else:
            self.phflag = True

        # doneflag (defaults to False)
        if 'doneflag' in kwargs:
            self.doneflag = bool(kwargs['doneflag'])
        else:
            self.doneflag = False

        # verbosity (defaults to 1)
        if 'verbosity' in kwargs:
            self.default_verbosity = int(kwargs['verbosity'])
        else:
            self.default_verbosity = 1

        # dry_run (defaults to False)
        if 'dry_run' in kwargs:
            self.default_dry_run = bool(kwargs['dry_run'])
        else:
            self.default_dry_run = False

        # n_threads (defaults to 1)
        if 'n_threads' in kwargs:
            try:
                self.default_threads = int(kwargs['n_threads'])
            except TypeError:
                # e.g. if n_threads is None
                self.default_threads = 1
        else:
            self.default_threads = 1

        # ~~~ path and filename args ~~~
        # loadglob (required)
        if 'loadglob' not in kwargs or kwargs['loadglob'] is None:
            raise RuntimeError('JobHandler requires a loadglob')
        elif not isstrlike(kwargs['loadglob']):
            raise RuntimeError('loadglob should be a string')
        loadpath_from_glob, loadglob = os.path.split(kwargs['loadglob'])
        self.loadglob = loadglob

        # loadpath (defaults to '')
        if 'loadpath' in kwargs and kwargs['loadpath'] is not None:
            if not isstrlike(kwargs['loadpath']):
                raise RuntimeError('loadpath should be a string')
            loadpath = kwargs['loadpath']
        else:
            loadpath = ''
        if os.path.samefile(loadpath, loadpath_from_glob):
            self.loadpath = loadpath
        else:
            self.loadpath = os.path.join(loadpath, loadpath_from_glob)

        # saveglob
        if 'saveglob' in kwargs:
            if isstrlike(kwargs['saveglob']):
                savepath_from_glob, saveglob = os.path.split(
                    kwargs['saveglob'])
            else:
                raise RuntimeError('saveglob should be a string')
            self.saveglob = kwargs['saveglob']
            self.saveflag = True
        else:
            self.saveglob = None
            self.saveflag = False

        # savepath (defaults to '' if saveglob exists)
        if self.saveflag:
            if 'savepath' in kwargs and kwargs['savepath'] is not None:
                if not isstrlike(kwargs['savepath']):
                    raise RuntimeError('savepath should be a string')
                savepath = kwargs['savepath']
            else:
                savepath = ''
            if os.path.samefile(savepath, savepath_from_glob):
                self.savepath = savepath
            else:
                self.savepath = os.path.join(savepath, savepath_from_glob)
        else:
            self.savepath = None

        # phglob
        if self.phflag:
            if 'phglob' in kwargs:
                pass
        # TODO
        # .......

    def start(self, verbosity=1, dry_run=False, n_threads=None):

        if n_threads is None:
            n_threads = self.default_threads

        # args...

        # loadfilelist...

        p = multiprocessing.Pool(processes=n_threads)
        p.map(self.do_work, flist)


def isstrlike(data):
    """
    isinstance(data, str) or isinstance(data, unicode)
    """
    return isinstance(data, str) or isinstance(data, unicode)
