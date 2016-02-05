#!/usr/bin/python

# Handle file processing jobs.

from __future__ import print_function
import os
import glob
import time
import multiprocessing
import numpy as np
import ipdb as pdb
import types
import functools


def checkfile(filepath):
    """
    Check whether a file exists. filepath includes path and name.
    """

    return os.path.isfile(filepath)


def phcheck(phfilepath, v=1):
    """
    Check whether a placeholder file exists at phfilepath.
    """

    if checkfile(phfilepath):
        phfile = os.path.split(phfilepath)[-1]
        vprint(v, 2, 'Found placeholder {}, skipping at {}'.format(
            phfile, time.ctime()))
        return True
    else:
        return False


def donecheck(donefilepath, v=1):
    """
    Check whether a done file exists at donefilepath.
    """

    if checkfile(donefilepath):
        donefile = os.path.split(donefilepath)[-1]
        vprint(v, 2, 'Found donefile {}, skipping at {}'.format(
            donefile, time.ctime()))
        return True
    else:
        return False


def savecheck(savefilepath, v=1):
    """
    Check whether a save file exists at savefilepath.
    """

    if checkfile(savefilepath):
        savefile = os.path.split(savefilepath)[-1]
        vprint(v, 2, 'Found savefile {}, skipping at {}'.format(
            savefile, time.ctime()))
        return True
    else:
        return False


def writeph(phfilepath, v=1):
    """
    Write a placeholder file at phfilepath.
    """

    phfile = os.path.split(phfilepath)[-1]
    vprint(v, 3, 'Creating placeholder {} at {}'.format(
        phfile, time.ctime()))
    with open(phfilepath, 'w') as phf:
        phf.write('placeholder')
    return None


def clearph(phfilepath, v=1):
    """
    Write a placeholder file at phfilepath.
    """

    phfile = os.path.split(phfilepath)[-1]
    vprint(v, 3, 'Removing placeholder {} at {}'.format(
        phfile, time.ctime()))
    try:
        os.remove(phfile)
    except OSError:
        vprint(v, 1, '! Missing placeholder {} at {}'.format(
            phfile, time.ctime()))
    return None


def writedone(donefilepath, v=1):
    """
    Write a donefile at donefilepath.
    """

    donefile = os.path.split(donefilepath)[-1]
    vprint(v, 3, 'Creating donefile {} at {}'.format(
        donefile, time.ctime()))
    with open(donefilepath, 'w') as donef:
        donef.write('done')
    return None


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
        if phpath/phglob are not given, then phpath = savepath
        and ph file will be 'ph_' + saveglob
      doneflag: create done-files. (default: False)
        if donepath/doneglob are not given, then donepath = savepath and
        doneglob = 'done_' + saveglob
      verbosity: default verbosity, passed to work_function (int) (default: 1)
      n_threads: default number of processes to run (multiprocessing) (int)
        (default: 1)
      only_numeric: flag for requiring * in globs to be only filled by digits

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
            raise JobError('JobHandler requires a valid work function')
        self.vanilla_work_function = work_function

        # in_place_flag (defaults to False)
        if 'in_place_flag' in kwargs:
            self.in_place_flag = bool(kwargs['in_place_flag'])
        else:
            self.in_place_flag = False

        # missing load error
        if 'loadpath' not in kwargs or 'loadglob' not in kwargs:
            raise JobError('JobHandler requires a loadpath and loadglob')

        # missing save error
        if not self.in_place_flag and 'saveglob' not in kwargs:
            raise JobError('JobHandler requires a savefile unless ' +
                           'in_place_flag is True')

        # only_numeric
        if 'only_numeric' in kwargs:
            self.only_numeric = kwargs['only_numeric']
        else:
            self.only_numeric = True

        # phflag (defaults to True)
        if 'phflag' in kwargs:
            self.phflag = bool(kwargs['phflag'])
        else:
            self.phflag = True

        # doneflag (defaults to False)
        if 'doneflag' in kwargs:
            self.doneflag = bool(kwargs['doneflag'])
        elif ('donepath' in kwargs or 'doneglob' in kwargs):
            self.doneflag = True
        else:
            self.doneflag = False

        # verbosity (defaults to 1)
        if 'verbosity' in kwargs:
            self.default_verbosity = int(kwargs['verbosity'])
        else:
            self.default_verbosity = 1

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
        if not isstrlike(kwargs['loadglob']):
            raise JobError('loadglob should be a string')
        self.loadglob = kwargs['loadglob']

        # loadpath (defaults to '')
        if 'loadpath' in kwargs and kwargs['loadpath'] is not None:
            if not isstrlike(kwargs['loadpath']):
                raise JobError('loadpath should be a string')
            self.loadpath = kwargs['loadpath']
        else:
            self.loadpath = ''

        # saveglob
        if 'saveglob' in kwargs:
            if not isstrlike(kwargs['saveglob']):
                raise JobError('saveglob should be a string')
            self.saveglob = kwargs['saveglob']
            self.savefilefunc = get_filename_function(
                self.loadglob, self.saveglob,
                only_numeric=self.only_numeric)
            # savepath (defaults to '')
            if 'savepath' in kwargs:
                if not isstrlike(kwargs['savepath']):
                    raise JobError('savepath should be a string')
                self.savepath = kwargs['savepath']
            else:
                self.savepath = self.loadpath
        else:
            self.savepath = None
            self.savefilefun = None

        if self.phflag:
            # phglob
            if 'phglob' in kwargs:
                if not isstrlike(kwargs['phglob']):
                    raise JobError('phglob should be a string')
                self.phglob = kwargs['phglob']
            else:
                self.phglob = 'ph_' + self.loadglob
            self.phfilefunc = get_filename_function(
                self.loadglob, self.phglob, only_numeric=self.only_numeric)
            # phpath (defaults to savepath if phglob exists)
            if 'phpath' in kwargs:
                if not isstrlike(kwargs['phpath']):
                    raise JobError('phpath should be a string')
                self.phpath = kwargs['phpath']
            else:
                self.phpath = self.savepath
        else:
            self.phpath = None
            self.phfilefunc = None

        if self.doneflag:
            # doneglob
            if 'doneglob' in kwargs:
                if not isstrlike(kwargs['doneglob']):
                    raise JobError('doneglob should be a string')
                self.doneglob = kwargs['doneglob']
            else:
                self.doneglob = 'done_' + self.saveglob
            self.donefilefunc = get_filename_function(
                self.loadglob, self.doneglob, only_numeric=self.only_numeric)
            # donepath (defaults to savepath if doneglob exists)
            if 'donepath' in kwargs:
                if not isstrlike(kwargs['donepath']):
                    raise JobError('donepath should be a string')
                self.donepath = kwargs['donepath']
            else:
                self.donepath = self.savepath
        else:
            self.donepath = None
            self.donefilefunc = None

    def start(self, verbosity=None, dry_run=False, n_threads=None):

        if verbosity is None:
            verbosity = self.default_verbosity
        if n_threads is None:
            n_threads = self.default_threads

        do_work = functools.partial(
            enhanced_work_function,
            vanilla_work_function=self.vanilla_work_function,
            v=verbosity, dry_run=dry_run,
            loadpath=self.loadpath, in_place_flag=self.in_place_flag,
            savefilefunc=self.savefilefunc, savepath=self.savepath,
            doneflag=self.doneflag,
            donefilefunc=self.donefilefunc, donepath=self.donepath,
            phflag=self.phflag, phfilefunc=self.phfilefunc, phpath=self.phpath)

        flist_with_path = glob.glob(os.path.join(self.loadpath, self.loadglob))
        flist = [os.path.split(f)[-1] for f in flist_with_path]
        flist.sort()

        vprint(verbosity, 2, '~~~ Beginning job with {} threads at {}'.format(
            n_threads, time.ctime()))

        if n_threads == 1:
            [do_work(f) for f in flist]
        else:
            p = multiprocessing.Pool(processes=n_threads)
            p.map(do_work, flist)

    def remove_ph_files(self):
        if self.phflag:
            rm_list = glob.glob(os.path.join(self.phpath, self.phglob))
            for f in rm_list:
                os.remove(f)

    def remove_done_files(self):
        if self.doneflag:
            rm_list = glob.glob(os.path.join(self.donepath, self.doneglob))
            for f in rm_list:
                os.remove(f)

    def remove_save_files(self, i_am_sure=False):
        if i_am_sure and not self.in_place_flag:
            rm_list = glob.glob(os.path.join(self.savepath, self.saveglob))
            for f in rm_list:
                os.remove(f)
        elif not self.in_place_flag:
            print("If you want me to remove savefiles, " +
                  "you have to tell me you're sure...")

    def remove_load_files(self, i_am_sure=False, totally_sure=False):
        if i_am_sure and totally_sure:
            rm_list = glob.glob(os.path.join(self.loadpath, self.loadglob))
            for f in rm_list:
                os.remove(f)
        else:
            print("If you want me to remove LOADfiles, " +
                  "you have to tell me you're sure, totally sure...")

    def remove_all_files(self, i_am_sure=False, totally_sure=False):
        self.remove_ph_files()
        self.remove_done_files()
        self.remove_save_files(i_am_sure=i_am_sure)
        self.remove_load_files(i_am_sure=i_am_sure, totally_sure=totally_sure)


def do_job(
        vanilla_work_function, loadpath, loadglob,
        v=1, dry_run=False, only_numeric=True,
        in_place_flag=False, savepath=None, saveglob=None,
        doneflag=None, donepath=None, doneglob=None,
        phflag=True, phpath=None, phglob=None,
        n_threads=multiprocessing.cpu_count()):

    if not isinstance(vanilla_work_function, types.FunctionType):
        raise JobError('requires a valid work function')

    if loadpath is None:
        loadpath = ''
    elif not isstrlike(loadpath):
        raise JobError('loadpath should be a string')
    if not isstrlike(loadglob):
        raise JobError('loadglob should be a string')

    if not in_place_flag and (saveglob is None or savepath is None):
        raise JobError(
            'JobHandler requires a savepath and saveglob unless ' +
            'in_place_flag is True')

    if doneflag is None and doneglob is not None:
        doneflag = True
    elif doneflag is None:
        doneflag = False
    if doneflag and donepath is None:
        donepath = savepath
    if doneflag and doneglob is None:
        doneglob = 'done_' + saveglob

    if phflag and phpath is None:
        phpath = savepath
    if phflag and phglob is None:
        phglob = 'ph_' + loadglob

    if not in_place_flag:
        savefilefunc = get_filename_function(loadglob, saveglob,
                                             only_numeric=only_numeric)
    else:
        savefilefunc = None
    if doneflag:
        donefilefunc = get_filename_function(loadglob, doneglob,
                                             only_numeric=only_numeric)
    else:
        donefilefunc = None
    if phflag:
        phfilefunc = get_filename_function(loadglob, phglob,
                                           only_numeric=only_numeric)
    else:
        phfilefunc = None

    partial_func = functools.partial(
        enhanced_work_function,
        vanilla_work_function=vanilla_work_function,
        v=v, dry_run=dry_run,
        loadpath=loadpath, in_place_flag=in_place_flag,
        savefilefunc=savefilefunc, savepath=savepath,
        doneflag=doneflag, donefilefunc=donefilefunc, donepath=donepath,
        phflag=phflag, phfilefunc=phfilefunc, phpath=phpath)

    flist_with_path = glob.glob(os.path.join(loadpath, loadglob))
    flist = [os.path.split(f)[-1] for f in flist_with_path]
    flist.sort()

    vprint(v, 2, '~~~ Beginning job with {} threads at {}'.format(
        n_threads, time.ctime()))
    p = multiprocessing.Pool(processes=n_threads)
    p.map(partial_func, flist)


def enhanced_work_function(
        filename, vanilla_work_function=None,
        v=1, dry_run=False,
        loadpath=None, in_place_flag=None,
        savefilefunc=None, savepath=None,
        doneflag=None, donefilefunc=None, donepath=None,
        phflag=None, phfilefunc=None, phpath=None):

    vprint(v, 3, ('Entering work function with v={}, dry_run={}, ' +
           'loadfile={}').format(v, dry_run, filename))

    # construct the relevant file paths and names
    loadfile = os.path.join(loadpath, filename)
    # if savepath == loadpath,
    # the glob will try to load test_24_save.h5 as well as test_24.h5.
    # It will throw a GlobError. Catch it and skip the file.
    try:
        if doneflag:
            donefile = os.path.join(donepath, donefilefunc(filename))
            vprint(v, 4, 'Donefile is {}'.format(donefile))
        if phflag:
            phfile = os.path.join(phpath, phfilefunc(filename))
            vprint(v, 4, 'Placeholder is {}'.format(phfile))
        if not in_place_flag:
            savefile = os.path.join(savepath, savefilefunc(filename))
            vprint(v, 4, 'Savefile is {}'.format(savefile))
    except GlobError:
        vprint(v, 3, 'Glob mismatch on {}, skipping at {}'.format(
            filename, time.ctime()))
        return None

    # skip?
    if doneflag and donecheck(donefile, v=v):
        return None
    if phflag and phcheck(phfile, v=v):
        return None
    if (not in_place_flag and not doneflag and savecheck(savefile, v=v)):
        # savefile only causes skip if not using donefiles
        return None

    # make placeholder
    if phflag:
        if dry_run:
            vprint(v, 3, '[dry_run] Creating placeholder {} at {}'.format(
                phfile, time.ctime()))
        else:
            writeph(phfile, v=v)
    # perform work
    vprint(v, 1, 'Starting {} at {}'.format(loadfile, time.ctime()))
    if in_place_flag:
        vanilla_work_function(loadfile, verbosity=v, dry_run=dry_run)
    else:
        vanilla_work_function(loadfile, savefile, verbosity=v, dry_run=dry_run)

    vprint(v, 2, 'Finishing {} at {}'.format(loadfile, time.ctime()))
    # finished
    if doneflag:
        if dry_run:
            vprint(v, 3, '[dry_run] Writing donefile {} at {}'.format(
                donefile, time.ctime()))
        else:
            writedone(donefile, v=v)
    if phflag:
        if dry_run:
            vprint(v, 3, '[dry_run] Removing placeholder {} at {}'.format(
                phfile, time.ctime()))
        else:
            clearph(phfile, v=v)

    vprint(v, 4, 'Exiting work function from {} at {}'.format(
        loadfile, time.ctime()))


def vprint(verbosity, vmin, textstring):
    if verbosity >= vmin:
        print(textstring)


def get_filename_function(inputglob, outputglob, only_numeric=True):
    """
    Generate a function which turns a filename of form inputglob into
    a filename of form outputglob.

    E.g.:
    func = get_filename_function('asdf_*.h5', 'qwerty_*_done.h5')
    func('asdf_24_6.h5')
    # returns 'qwerty_24_6_done.h5'
    """

    input_split = split_glob(inputglob)
    output_split = split_glob(outputglob)
    if len(input_split) != len(output_split):
        raise GlobError('Input and output globs need same number of *!')

    def filename_function(filename):
        contents = get_glob_content(filename, inputglob,
                                    only_numeric=only_numeric)
        output_filename = put_glob_content(contents, outputglob)
        return output_filename

    return filename_function


def split_glob(globname):
    """
    Return a list of N+1 parts, where N is the number of asterisks in the glob.

    E.g.:
    split_glob('asdf_*_*.h5')
    # []'asdf_', '_', '.h5']
    """

    parts = []
    ind = 0
    while '*' in globname[ind:]:
        ind2 = globname.find('*', ind)
        parts.append(globname[ind:ind2])
        ind = ind2 + 1      # don't include the asterisk
    parts.append(globname[ind:])

    return parts


def get_glob_content(filename, globname, only_numeric=True):
    """
    Return a list of N strings from filename that are in the place of
    the glob asterisks. N is the number of asterisks.

    If filename doesn't match globname, raise a JobError.

    E.g.:
    get_glob_content('MultiAngle_24_12.h5', MultiAngle_*_*.h5)
    # ['24', '12']
    """

    if not isstrlike(globname):
        raise GlobError('globname must be a string type')
    if not isstrlike(filename):
        raise GlobError('filename must be a string type')

    parts = split_glob(globname)

    if not filename.startswith(parts[0]):
        raise GlobError("filename doesn't match glob")
    content = []
    ind = len(parts[0])

    for part in parts[1:]:
        ind2 = filename.find(part, ind)
        if ind2 == -1:
            raise GlobError("filename doesn't match glob")
        content.append(filename[ind:ind2])
        if only_numeric and not content[-1].isdigit():
            raise GlobError("filename doesn't match glob")
        ind = ind2 + len(part)

    return content


def put_glob_content(contents, globname):
    """
    Put the contents list (list of string-like) into the * of the glob.
    """

    parts = split_glob(globname)
    if len(contents) + 1 != len(parts):
        raise GlobError("contents don't match number of * in globname")

    out = parts[0]
    for content, part in zip(contents, parts[1:]):
        out += content
        out += part

    return out


def isstrlike(data):
    """
    isinstance(data, str) or isinstance(data, unicode)
    """
    return isinstance(data, str) or isinstance(data, unicode)


class JobError(Exception):
    pass


class GlobError(Exception):
    pass


################################################################
#                       Testing
################################################################

def test_split_glob():
    parts = split_glob('MultiAngle_*_*.h5')
    assert len(parts) == 3
    assert parts[0] == 'MultiAngle_'
    assert parts[1] == '_'
    assert parts[2] == '.h5'

    parts = split_glob('asdf.asdf')
    assert len(parts) == 1
    assert parts[0] == 'asdf.asdf'


def test_get_glob_content():
    # basic
    globname = 'MultiAngle_*_*.h5'
    filename = 'MultiAngle_24_12.h5'
    c = get_glob_content(filename, globname)
    assert len(c) == 2
    assert c[0] == '24'
    assert c[1] == '12'

    globname = 'MultiAngle_*_*_.h5'
    filename = 'MultiAngle_24_12.h5'
    try:
        c = get_glob_content(filename, globname)
    except GlobError:
        pass
    else:
        raise AssertionError('get_glob_content() failed to raise error')

    globname = 'MultiAngle_*_a_*.h5'
    filename = 'MultiAngle_24_12.h5'
    try:
        c = get_glob_content(filename, globname)
    except GlobError:
        pass
    else:
        raise AssertionError('get_glob_content() failed to raise error')

    globname = 'MultiAngles_*_*.h5'
    filename = 'MultiAngle_24_12.h5'
    try:
        c = get_glob_content(filename, globname)
    except GlobError:
        pass
    else:
        raise AssertionError('get_glob_content() failed to raise error')

    # only_numeric
    globname = 'MultiAngle_*_*.h5'
    filename = 'MultiAngle_24_12_.h5'
    try:
        c = get_glob_content(filename, globname)
    except GlobError:
        pass
    else:
        raise AssertionError('get_glob_content() failed to raise error')

    globname = 'MultiAngle_*_*.h5'
    filename = 'MultiAngle_24_12_.h5'
    c = get_glob_content(filename, globname, only_numeric=False)
    assert len(c) == 2
    assert c[0] == '24'
    assert c[1] == '12_'


def test_put_glob_contents():
    # basic
    globname = 'MultiAngle_*_*.h5'
    contents = ['24', '12']
    assert put_glob_content(contents, globname) == 'MultiAngle_24_12.h5'

    # round trip
    globname = 'MultiAngle_*_*.h5'
    filename = 'MultiAngle_24_12.h5'
    content = get_glob_content(filename, globname)
    new_filename = put_glob_content(content, globname)
    assert new_filename == filename

    # check mismatch
    globname = 'MultiAngle_*_*.h5'
    contents = ['24', '12', 'extra']
    try:
        put_glob_content(contents, globname)
    except GlobError:
        pass
    else:
        raise AssertionError('put_glob_contents() failed to raise error')


def test_isstrlike():
    assert isstrlike('qwertyuioxcvbnm1234567890~!@##$%@#$#%^$&^%*&,/.,;][p]')
    assert isstrlike(u'qwertyuioxcvbnm1234567890~!@##$%@#$#%^$&^%*&,/.,;][p]')
    assert isstrlike(123) is False
    assert isstrlike(3.14) is False
    assert isstrlike(['a', 'b', 'c']) is False
    assert isstrlike(('a', 'b', 'c')) is False
    assert isstrlike({'a': 1, 'b': 2}) is False


def test_get_filename_function():
    inputglob = 'MultiAngle_*_*.h5'
    outputglob = 'finished_*_and_*_asdf.h5'
    func = get_filename_function(inputglob, outputglob)
    inputfilename = 'MultiAngle_24_12.h5'
    assert func(inputfilename) == 'finished_24_and_12_asdf.h5'


def get_test_work_function(mintime=4, maxtime=5,
                           myverbosity=False, nosave=False):

    if nosave:
        def test_work(loadfile, verbosity=1, dry_run=False):
            if myverbosity:
                print(('+ test_work loadfile={} verbosity={} ' +
                       'dry_run={} +').format(
                      loadfile, verbosity, dry_run))
            sleeptime = mintime + np.random.random() * (maxtime - mintime)
            time.sleep(sleeptime)
            if myverbosity:
                print(('/ test_work loadfile={} verbosity={} ' +
                       'dry_run={} /').format(
                      loadfile, verbosity, dry_run))
    else:
        def test_work(loadfile, savefile, verbosity=1, dry_run=False):
            if myverbosity:
                print(('+ test_work loadfile={} savefile={} verbosity={} ' +
                       'dry_run={} +').format(
                      loadfile, savefile, verbosity, dry_run))
            sleeptime = mintime + np.random.random() * (maxtime - mintime)
            time.sleep(sleeptime)
            if not dry_run:
                with open(savefile, 'w') as s:
                    s.write(' ')
            if myverbosity:
                print(('/ test_work loadfile={} savefile={} verbosity={} ' +
                       'dry_run={} /').format(
                      loadfile, savefile, verbosity, dry_run))

    return test_work


def create_test_files(writepath, writeglob, n):
    if not os.path.isdir(writepath):
        os.mkdir(writepath)
    for i in range(n):
        filename = os.path.join(
            writepath, put_glob_content([str(i)], writeglob))
        pytouch(filename)


def pytouch(flist):
    if isinstance(flist, list) or isinstance(flist, tuple):
        for fpath in flist:
            with open(fpath, 'w') as f:
                f.write(' ')
    elif isstrlike(flist):
        with open(flist, 'w') as f:
            f.write(' ')


def remove_test_files(rmpath, rmglob, n):
    for i in range(n):
        filename = os.path.join(rmpath, put_glob_content(str(i), rmglob))
        os.remove(filename)


def test_run_job():

    def test1():
        # pdb.set_trace()
        # separate dirs, default settings, starting clean
        jh = test_job(do_work,
                      './testload', 'test_*.h5',
                      './testsave', 'test_*_save.h5',
                      n, {'verbosity': v})
        t1 = time.time()
        jh.start(n_threads=n_threads)
        dt = time.time() - t1
        assert dt > n / n_threads * mintime
        assert dt < n / n_threads * maxtime
        jh.remove_all_files(i_am_sure=True, totally_sure=True)

    def test2():
        # separate dirs, default settings, starting with save and ph
        jh = test_job(do_work,
                      './testload', 'test_*.h5',
                      './testsave', 'test_*_save.h5',
                      n, {'verbosity': v})
        if n_threads == 1:
            pytouch(['./testsave/test_1_save.h5', './testsave/test_2_save.h5',
                     './testsave/ph_test_7.h5'])
        elif n_threads == 4:
            pytouch(['./testsave/test_1_save.h5',
                     './testsave/test_2_save.h5',
                     './testsave/test_3_save.h5',
                     './testsave/test_4_save.h5',
                     './testsave/ph_test_7.h5',
                     './testsave/ph_test_8.h5',
                     './testsave/ph_test_9.h5',
                     './testsave/ph_test_10.h5'])
        t1 = time.time()
        jh.start(n_threads=n_threads)
        dt = time.time() - t1
        if n_threads == 1:
            assert dt > (n - 3) * mintime
            assert dt < (n - 3) * maxtime
        elif n_threads == 4:
            assert dt > (n - 8) / n_threads * mintime
            assert dt < (n - 8) / n_threads * maxtime
        jh.remove_all_files(i_am_sure=True, totally_sure=True)

    def test3():
        # same dir, defaults, starting clean
        jh = test_job(do_work,
                      './testload', 'test_*.h5',
                      './testload', 'test_*_save.h5',
                      n, {'verbosity': v})
        t1 = time.time()
        jh.start(n_threads=n_threads)
        dt = time.time() - t1
        assert dt > n / n_threads * mintime
        assert dt < n / n_threads * maxtime
        jh.remove_all_files(i_am_sure=True, totally_sure=True)

    def test4():
        # no ph, start with save (and ph)
        jh = test_job(do_work,
                      './testload', 'test_*.h5',
                      './testsave', 'test_*_save.h5',
                      n, {'verbosity': v, 'phflag': False})
        if n_threads == 1:
            pytouch(['./testsave/test_1_save.h5', './testsave/test_2_save.h5',
                     './testsave/ph_test_7.h5'])
        elif n_threads == 4:
            pytouch(['./testsave/test_1_save.h5',
                     './testsave/test_2_save.h5',
                     './testsave/test_3_save.h5',
                     './testsave/test_4_save.h5',
                     './testsave/ph_test_7.h5',
                     './testsave/ph_test_8.h5',
                     './testsave/ph_test_9.h5',
                     './testsave/ph_test_10.h5'])
        t1 = time.time()
        jh.start(n_threads=n_threads)
        dt = time.time() - t1
        # the created ph shouldn't cause a skip
        if n_threads == 1:
            assert dt > (n - 2) * mintime
            assert dt < (n - 2) * maxtime
        elif n_threads == 4:
            assert dt > (n - 4) / n_threads * mintime
            assert dt < (n - 4) / n_threads * maxtime
        jh.remove_all_files(i_am_sure=True, totally_sure=True)

    def test5():
        # donefiles, start with save and ph and done
        jh = test_job(do_work,
                      './testload', 'test_*.h5',
                      './testsave', 'test_*_save.h5',
                      n, {'verbosity': v, 'doneflag': True})
        if n_threads == 1:
            pytouch(['./testsave/test_1_save.h5',
                     './testsave/test_2_save.h5',
                     './testsave/test_3_save.h5',
                     './testsave/test_4_save.h5',
                     './testsave/ph_test_6.h5',
                     './testsave/ph_test_7.h5',
                     './testsave/done_test_0_save.h5'])
        elif n_threads == 4:
            pytouch(['./testsave/test_1_save.h5',
                     './testsave/test_2_save.h5',
                     './testsave/test_3_save.h5',
                     './testsave/test_4_save.h5',
                     './testsave/done_test_12_save.h5',
                     './testsave/done_test_13_save.h5',
                     './testsave/done_test_14_save.h5',
                     './testsave/done_test_15_save.h5',
                     './testsave/ph_test_7.h5',
                     './testsave/ph_test_8.h5',
                     './testsave/ph_test_9.h5',
                     './testsave/ph_test_10.h5'])
        t1 = time.time()
        jh.start(n_threads=n_threads)
        dt = time.time() - t1
        # created savefiles shouldn't be skipped
        if n_threads == 1:
            assert dt > (n - 3) * mintime
            assert dt < (n - 3) * maxtime
        elif n_threads == 4:
            assert dt > (n - 8) / n_threads * mintime
            assert dt < (n - 8) / n_threads * maxtime
        jh.remove_all_files(i_am_sure=True, totally_sure=True)

    def test6():
        # dry_run
        jh = test_job(do_work,
                      './testload', 'test_*.h5',
                      './testsave', 'test_*_save.h5',
                      n, {'verbosity': v})
        t1 = time.time()
        jh.start(dry_run=True, n_threads=n_threads)
        dt = time.time() - t1
        assert dt > n / n_threads * mintime
        assert dt < n / n_threads * maxtime
        assert not os.path.exists('./testsave/test_3_save.h5')
        jh.remove_all_files(i_am_sure=True, totally_sure=True)

    def test7():
        # in place. start with save and ph and done
        jh = test_job(do_work,
                      './testload', 'test_*.h5',
                      './testsave', 'test_*_save.h5',
                      n, {'verbosity': v, 'in_place_flag': True,
                          'doneflag': True})
        if n_threads == 1:
            pytouch(['./testsave/test_1_save.h5',
                     './testsave/test_2_save.h5',
                     './testsave/test_3_save.h5',
                     './testsave/test_4_save.h5',
                     './testsave/ph_test_6.h5',
                     './testsave/ph_test_7.h5',
                     './testsave/done_test_0_save.h5'])
        elif n_threads == 4:
            pytouch(['./testsave/test_1_save.h5',
                     './testsave/test_2_save.h5',
                     './testsave/test_3_save.h5',
                     './testsave/test_4_save.h5',
                     './testsave/done_test_12_save.h5',
                     './testsave/done_test_13_save.h5',
                     './testsave/done_test_14_save.h5',
                     './testsave/done_test_15_save.h5',
                     './testsave/ph_test_7.h5',
                     './testsave/ph_test_8.h5',
                     './testsave/ph_test_9.h5',
                     './testsave/ph_test_10.h5'])
        t1 = time.time()
        jh.start(n_threads=n_threads)
        dt = time.time() - t1
        # created savefiles shouldn't be skipped
        if n_threads == 1:
            assert dt > (n / n_threads - 3) * mintime
            assert dt < (n / n_threads - 3) * maxtime
        elif n_threads == 4:
            assert dt > (n - 8) / n_threads * mintime
            assert dt < (n - 8) / n_threads * maxtime
        jh.remove_all_files(i_am_sure=True, totally_sure=True)

    # single threaded
    n_threads = 1
    v = 2
    n = 10
    mintime = 4
    maxtime = 4.25
    # check that the number of files processed is unambiguous (single thread)
    assert (maxtime - mintime) * n < mintime
    do_work = get_test_work_function(mintime=mintime, maxtime=maxtime)

    # temp test
    do_job(do_work, loadpath='./testload', loadglob='test_*.h5',
           v=2, savepath='./testsave', saveglob='test_*_save.h5')
    pdb.set_trace()

    # pre-clean, in case last run was interrupted
    jh = test_job(do_work,
                  './testload', 'test_*.h5',
                  './testsave', 'test_*_save.h5',
                  n, {})
    jh.remove_all_files(i_am_sure=True, totally_sure=True)
    # test1()
    # test2()
    # test3()
    # test4()
    # test5()
    # test6()
    do_work = get_test_work_function(
        mintime=mintime, maxtime=maxtime, nosave=True)
    # test7()

    # multi threaded
    n_threads = 4
    n = 20
    # check that the number of files processed is unambiguous (multi thread)
    assert (maxtime - mintime) * (float(n) / n_threads) < mintime
    # pre-clean, in case last run was interrupted
    do_work = get_test_work_function(mintime=mintime, maxtime=maxtime)
    jh = test_job(do_work,
                  './testload', 'test_*.h5',
                  './testsave', 'test_*_save.h5',
                  n, {})
    jh.remove_all_files(i_am_sure=True, totally_sure=True)
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
    do_work = get_test_work_function(
        mintime=mintime, maxtime=maxtime, nosave=True)
    test7()


def test_job(work_function, loadpath, loadglob,
             savepath, saveglob,
             n, handler_kwargs):
    # setup
    print(' ')
    create_test_files(loadpath, loadglob, n)
    create_test_files(savepath, saveglob, 0)

    # "real work"
    jh = JobHandler(
        work_function,
        loadpath=loadpath, loadglob=loadglob,
        savepath=savepath, saveglob=saveglob,
        **handler_kwargs)

    return jh


class testclass(object):
    def amethod(self, filename):
        print('anotherfunc' + filename)


def mapfunc(c, v, dry_run, filename):
    if v > 2:
        print('v')
    if not dry_run:
        c.amethod(filename)


def maptest():

    c = testclass()
    f = functools.partial(mapfunc, c, 3, False)
    flist = ['asdf' + str(n) + '.test' for n in range(4)]

    f('./asdf.test')

    p = multiprocessing.Pool(processes=4)
    p.map(f, flist)


if __name__ == '__main__':
    # maptest()

    test_isstrlike()
    test_split_glob()
    test_get_glob_content()
    test_put_glob_contents()
    test_get_filename_function()

    test_run_job()

    if False:
        pdb.set_trace()
