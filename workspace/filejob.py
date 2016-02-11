#!/usr/bin/python

# Handle file processing jobs.

from __future__ import print_function
import os
import glob
import time
import multiprocessing
import numpy as np
import ipdb as pdb


# #########    begin copy to script file    ####################

def run_main():

    multi_flag = True   # run in parallel - turn off to debug
    _, loadpath, savepath, loadglob, saveglob, v, n_proc = file_vars()

    flist = glob.glob(os.path.join(loadpath, loadglob))
    flist.sort()

    if multi_flag:
        p = multiprocessing.Pool(processes=n_proc, maxtasksperchild=5)
        p.map(runfile, flist, chunksize=5)
    else:
        [runfile(f) for f in flist]


def file_vars():
    """
    Define file path and globs here. Also server flag, and verbosity.

    Gets loaded in run_main() as well as runfile(loadname).
    """

    server_flag = True
    if server_flag:
        n_threads = 11
        loadpath = './loadpath'
        savepath = './savepath'
    else:
        # e.g. LBL desktop
        n_threads = 4
        loadpath = './loadpath'
        savepath = './savepath'
    loadglob = 'load_*.h5'
    saveglob = 'save_*.h5'

    v = 1   # verbosity

    return server_flag, loadpath, savepath, loadglob, saveglob, v, n_threads


def runfile(loadname):
    """
    To use:
      1. make a copy in your script
      2. edit this function name to e.g. runmyjob(loadname)
      3. edit work function name from main_work_function below
      4. edit paths, globs, and flags
      5. in the main script:
         flist = glob.glob(os.path.join(loadpath, loadglob))
         p = multiprocessing.Pool(processes=n, maxtasksperchild=25)
         p.map(runmyjob, flist, chunksize=25)
    """

    # drop the path part of loadname, if it is given
    server_flag, loadpath, savepath, loadglob, saveglob, v, _ = file_vars()

    in_place_flag = False
    phflag = True
    doneflag = False

    # setup
    opts = JobOptions(
        loadpath=loadpath, loadglob=loadglob,
        savepath=savepath, saveglob=saveglob,
        in_place_flag=in_place_flag, phflag=phflag, doneflag=doneflag)
    # decide to skip or not; construct full filenames
    loadfile, savefile = opts.pre_job_tasks(loadname)
    if loadfile is not None and savefile is not None:
        # do the work
        main_work_function(loadfile, savefile)
        # clean up
        opts.post_job_tasks(loadname)


def main_work_function(loadfile, savefile):
    # for testing only!
    f = get_test_work_function(
        mintime=4, maxtime=4.25, myverbosity=False, nosave=False)
    return f(loadfile, savefile)

# ###########     end copy to script file     #####################


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
        os.remove(phfilepath)
    except OSError:
        vprint(v, 1, '! Missing placeholder {} at {}'.format(
            phfilepath, time.ctime()))
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


class JobOptions(object):
    """
    Create an object representing options for a file-by-file processing job.

    Required inputs:

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

    def __init__(self, **kwargs):
        """
        Check input args and write values into self.
        """

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
            self.v = int(kwargs['verbosity'])
        else:
            self.v = 1

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
                if self.in_place_flag:
                    # no savepath
                    self.phpath = self.loadpath
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
                if self.in_place_flag:
                    # no saveglob or savepath
                    self.doneglob = 'done_' + self.loadglob
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
                if self.in_place_flag:
                    # no saveglob or savepath
                    self.donepath = self.loadpath
                else:
                    self.donepath = self.savepath
        else:
            self.donepath = None
            self.donefilefunc = None

    def pre_job_tasks(self, loadname):
        """
        Take care of things before one file of the job starts.

        Includes checking save, ph, done files, printing messages.

        Inputs:
          loadname: the filename (NOT path) of file to load
          opts: a JobOptions isinstance

        Outputs:
          loadfile: the full path of the file to load
          savefile: the full path of the file to save

        If the file is to be skipped, then loadfile and savefile are None
        """

        loadfile = os.path.join(self.loadpath, loadname)

        if self.doneflag:
            donename = self.donefilefunc(loadname)
            donefile = os.path.join(self.donepath, donename)
            if donecheck(donefile, v=self.v):
                return None, None
        if self.in_place_flag:
            savefile = None
        else:
            savename = self.savefilefunc(loadname)
            savefile = os.path.join(self.savepath, savename)
            if not self.doneflag:
                # savecheck is verbose, so these ifs are nested
                if savecheck(savefile, v=self.v):
                    return None, None
        if self.phflag:
            phname = self.phfilefunc(loadname)
            phfile = os.path.join(self.phpath, phname)
            if phcheck(phfile, v=self.v):
                return None, None
            writeph(phfile, v=self.v)

        return loadfile, savefile

    def post_job_tasks(self, loadname):
        """
        Clean up after one file finishes.
        """

        if self.doneflag:
            donename = self.donefilefunc(loadname)
            donefile = os.path.join(self.donepath, donename)
            writedone(donefile, v=self.v)
        if self.phflag:
            phname = self.phfilefunc(loadname)
            phfile = os.path.join(self.phpath, phname)
            clearph(phfile, v=self.v)

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


def remove_test_files():
    paths = ['./testload', './testsave']
    for p in paths:
        flist = glob.glob(os.path.join(p, '*'))
        for f in flist:
            os.remove(f)


def default_work(loadfile, savefile):
    f = get_test_work_function(mintime=4, maxtime=4.25,
                               myverbosity=False, nosave=False)
    return f(loadfile, savefile)


def default_work_nosave(loadfile):
    f = get_test_work_function(mintime=4, maxtime=4.25,
                               myverbosity=False, nosave=True)
    return f(loadfile)


######################################################
#                    Test runfiles                   #
######################################################

def run_test_file_A(loadname):
    # separate dirs, default settings
    # test scripts 1 and 2

    # paths, globs, flags
    loadpath = './testload'
    loadglob = 'test_*.h5'
    savepath = './testsave'
    saveglob = 'test_*_save.h5'
    in_place_flag = False
    phflag = True
    doneflag = False

    # setup
    opts = JobOptions(
        loadpath=loadpath, loadglob=loadglob,
        savepath=savepath, saveglob=saveglob,
        in_place_flag=in_place_flag, phflag=phflag, doneflag=doneflag,
        verbosity=2)
    # decide to skip or not; construct full filenames
    loadfile, savefile = opts.pre_job_tasks(loadname)
    if loadfile is not None and savefile is not None:
        print('Starting ' + loadname + ' at ' + str(time.ctime()))
        # do the work
        default_work(loadfile, savefile)
        # clean up
        opts.post_job_tasks(loadname)
    else:
        print('--Skipping ' + loadname + ' at ' + time.ctime())


def run_test_file_B(loadname):
    # same dir, default settings
    # test script 3

    # paths, globs, flags
    loadpath = './testload'
    loadglob = 'test_*.h5'
    savepath = loadpath
    saveglob = 'test_*_save.h5'
    in_place_flag = False
    phflag = True
    doneflag = False

    # setup
    opts = JobOptions(
        loadpath=loadpath, loadglob=loadglob,
        savepath=savepath, saveglob=saveglob,
        in_place_flag=in_place_flag, phflag=phflag, doneflag=doneflag,
        verbosity=2)
    # decide to skip or not; construct full filenames
    loadfile, savefile = opts.pre_job_tasks(loadname)
    if loadfile is not None and savefile is not None:
        print('Starting ' + loadname + ' at ' + str(time.ctime()))
        # do the work
        default_work(loadfile, savefile)
        # clean up
        opts.post_job_tasks(loadname)
    else:
        print('--Skipping ' + loadname + ' at ' + time.ctime())


def run_test_file_C(loadname):
    # separate dirs, no PH
    # test script 4

    # paths, globs, flags
    loadpath = './testload'
    loadglob = 'test_*.h5'
    savepath = './testsave'
    saveglob = 'test_*_save.h5'
    in_place_flag = False
    phflag = False
    doneflag = False

    # setup
    opts = JobOptions(
        loadpath=loadpath, loadglob=loadglob,
        savepath=savepath, saveglob=saveglob,
        in_place_flag=in_place_flag, phflag=phflag, doneflag=doneflag,
        verbosity=2)
    # decide to skip or not; construct full filenames
    loadfile, savefile = opts.pre_job_tasks(loadname)
    if loadfile is not None and savefile is not None:
        print('Starting ' + loadname + ' at ' + str(time.ctime()))
        # do the work
        default_work(loadfile, savefile)
        # clean up
        opts.post_job_tasks(loadname)
    else:
        print('--Skipping ' + loadname + ' at ' + time.ctime())


def run_test_file_D(loadname):
    # donefiles (and ph and save)
    # test script 5

    # paths, globs, flags
    loadpath = './testload'
    loadglob = 'test_*.h5'
    savepath = './testsave'
    saveglob = 'test_*_save.h5'
    in_place_flag = False
    phflag = True
    doneflag = True

    # setup
    opts = JobOptions(
        loadpath=loadpath, loadglob=loadglob,
        savepath=savepath, saveglob=saveglob,
        in_place_flag=in_place_flag, phflag=phflag, doneflag=doneflag,
        verbosity=2)
    # decide to skip or not; construct full filenames
    loadfile, savefile = opts.pre_job_tasks(loadname)
    if loadfile is not None and savefile is not None:
        print('Starting ' + loadname + ' at ' + str(time.ctime()))
        # do the work
        default_work(loadfile, savefile)
        # clean up
        opts.post_job_tasks(loadname)
    else:
        print('--Skipping ' + loadname + ' at ' + time.ctime())


def run_test_file_E(loadname):
    # in_place_flag and donefiles (and ph and save)
    # test script 6

    # paths, globs, flags
    loadpath = './testload'
    loadglob = 'test_*.h5'
    in_place_flag = True
    phflag = True
    doneflag = True

    # setup
    opts = JobOptions(
        loadpath=loadpath, loadglob=loadglob,
        in_place_flag=in_place_flag, phflag=phflag, doneflag=doneflag,
        verbosity=2)
    # decide to skip or not; construct full filenames
    loadfile, _ = opts.pre_job_tasks(loadname)
    if loadfile is not None:
        print('Starting ' + loadname + ' at ' + str(time.ctime()))
        # do the work
        default_work_nosave(loadfile)
        # clean up
        opts.post_job_tasks(loadname)
    else:
        print('--Skipping ' + loadname + ' at ' + time.ctime())


######################################################
#                    Test scripts                    #
######################################################

def test1():
    # separate dirs, default settings, starting clean
    loadpath = './testload'
    loadglob = 'test_*.h5'
    mintime = 4
    maxtime = 4.25
    n = 20
    create_test_files(loadpath, loadglob, n)
    flist = [os.path.split(f)[-1]
             for f in glob.glob(os.path.join(loadpath, loadglob))]
    flist.sort()
    multi_process = True

    if multi_process:
        print('Multiprocessing')
        p = multiprocessing.Pool(processes=4)

    print('Starting Test 1...')
    t1 = time.time()
    if multi_process:
        p.map(run_test_file_A, flist, chunksize=1)
    else:
        [run_test_file_A(f) for f in flist]
    dt = time.time() - t1
    print(dt)
    assert dt > n / 4 * mintime
    assert dt < n / 4 * maxtime
    remove_test_files()


def test2():
    # separate dirs, default settings, starting with save and ph
    loadpath = './testload'
    loadglob = 'test_*.h5'
    mintime = 4
    maxtime = 4.25
    n = 20
    create_test_files(loadpath, loadglob, n)
    flist = [os.path.split(f)[-1]
             for f in glob.glob(os.path.join(loadpath, loadglob))]
    p = multiprocessing.Pool(processes=4)

    pytouch(['./testsave/test_1_save.h5',
             './testsave/test_2_save.h5',
             './testsave/test_3_save.h5',
             './testsave/test_4_save.h5',
             './testsave/ph_test_7.h5',
             './testsave/ph_test_8.h5',
             './testsave/ph_test_9.h5',
             './testsave/ph_test_10.h5'])

    t1 = time.time()
    p.map(run_test_file_A, flist, chunksize=1)
    dt = time.time() - t1
    assert dt > (n / 4 - 2) * mintime
    assert dt < (n / 4 - 2) * maxtime
    remove_test_files()


def test3():
    # same dirs, default settings, starting clean
    loadpath = './testload'
    loadglob = 'test_*.h5'
    mintime = 4
    maxtime = 4.25
    n = 20
    create_test_files(loadpath, loadglob, n)
    flist = [os.path.split(f)[-1]
             for f in glob.glob(os.path.join(loadpath, loadglob))]
    p = multiprocessing.Pool(processes=4)

    t1 = time.time()
    p.map(run_test_file_B, flist, chunksize=1)
    dt = time.time() - t1
    assert dt > n / 4 * mintime
    assert dt < n / 4 * maxtime
    remove_test_files()


def test4():
    # separate dirs, no ph, start with ph and save
    loadpath = './testload'
    loadglob = 'test_*.h5'
    mintime = 4
    maxtime = 4.25
    n = 20
    create_test_files(loadpath, loadglob, n)
    flist = [os.path.split(f)[-1]
             for f in glob.glob(os.path.join(loadpath, loadglob))]
    p = multiprocessing.Pool(processes=4)

    pytouch(['./testsave/test_1_save.h5',
             './testsave/test_2_save.h5',
             './testsave/test_3_save.h5',
             './testsave/test_4_save.h5',
             './testsave/ph_test_7.h5',
             './testsave/ph_test_8.h5',
             './testsave/ph_test_9.h5',
             './testsave/ph_test_10.h5'])

    t1 = time.time()
    p.map(run_test_file_C, flist, chunksize=1)
    dt = time.time() - t1
    assert dt > (n / 4 - 1) * mintime
    assert dt < (n / 4 - 1) * maxtime
    remove_test_files()


def test5():
    # separate dirs, donefiles, start with done and ph and save
    loadpath = './testload'
    loadglob = 'test_*.h5'
    mintime = 4
    maxtime = 4.25
    n = 20
    create_test_files(loadpath, loadglob, n)
    flist = [os.path.split(f)[-1]
             for f in glob.glob(os.path.join(loadpath, loadglob))]
    p = multiprocessing.Pool(processes=4)

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
    p.map(run_test_file_D, flist, chunksize=1)
    dt = time.time() - t1
    assert dt > (n / 4 - 2) * mintime
    assert dt < (n / 4 - 2) * maxtime
    remove_test_files()


def test6():
    # in_place, donefiles, start with done and ph and save
    loadpath = './testload'
    loadglob = 'test_*.h5'
    mintime = 4
    maxtime = 4.25
    n = 20
    create_test_files(loadpath, loadglob, n)
    flist = [os.path.split(f)[-1]
             for f in glob.glob(os.path.join(loadpath, loadglob))]
    flist.sort()
    multi_process = True
    if multi_process:
        p = multiprocessing.Pool(processes=4)

    # all files in dir testload, because there is no save dir
    pytouch(['./testload/test_1_save.h5',
             './testload/test_2_save.h5',
             './testload/test_3_save.h5',
             './testload/test_4_save.h5',
             './testload/done_test_12.h5',
             './testload/done_test_13.h5',
             './testload/done_test_14.h5',
             './testload/done_test_15.h5',
             './testload/ph_test_7.h5',
             './testload/ph_test_8.h5',
             './testload/ph_test_9.h5',
             './testload/ph_test_10.h5'])

    t1 = time.time()
    if multi_process:
        p.map(run_test_file_E, flist, chunksize=1)
    else:
        [run_test_file_E(f) for f in flist]
    dt = time.time() - t1
    assert dt > (n / 4 - 2) * mintime
    assert dt < (n / 4 - 2) * maxtime
    remove_test_files()


def run_test_scripts():
    # start clean
    loadpath = './testload'
    loadglob = 'test_*.h5'
    savepath = './testsave'
    saveglob = 'test_*_save.h5'
    create_test_files(loadpath, loadglob, 20)
    create_test_files(savepath, saveglob, 0)
    remove_test_files()

    print('Test 1...')
    test1()
    print('...success!\n\nTest 2...')
    test2()
    print('...success!\n\nTest 3...')
    test3()
    print('...success!\n\nTest 4...')
    test4()
    print('...success!\n\nTest 5...')
    test5()
    print('...success!\n\nTest 6...')
    test6()
    print('...success!\n')


if __name__ == '__main__':
    test_isstrlike()
    test_split_glob()
    test_get_glob_content()
    test_put_glob_contents()
    test_get_filename_function()

    run_test_scripts()

    if False:
        pdb.set_trace()
