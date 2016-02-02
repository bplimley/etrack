#!/usr/bin/python

# Handle file processing jobs.

from __future__ import print_function
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
        if phpath/phglob are not given, then phpath = savepath
        and ph file will be 'ph_' + saveglob
      doneflag: create done-files. (default: False)
        if donepath/doneglob are not given, then donepath = savepath and
        doneglob = 'done_' + saveglob
      verbosity: default verbosity, passed to work_function (int) (default: 1)
      dry_run: default state of doing a dry-run (bool) (default: False)
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
        self.full_work_function = work_function

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

        # initialize do_work with defaults
        self.set_work_function(
            verbosity=self.default_verbosity, dry_run=self.default_dry_run)

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

    def set_work_function(self, verbosity, dry_run):

        self.do_work = self.get_work_function(
            v=verbosity, dry_run=dry_run)

    def start(self, verbosity=1, dry_run=False, n_threads=None):

        if n_threads is None:
            n_threads = self.default_threads

        self.set_work_function(verbosity=verbosity, dry_run=dry_run)

        flist_with_path = glob.glob(os.path.join(self.loadpath, self.loadglob))
        flist = [os.path.split(f)[-1] for f in flist_with_path]
        flist.sort()

        vprint(verbosity, 2, '~~~ Beginning job with {} threads at {}'.format(
            n_threads, time.ctime()))

        if n_threads == 1:
            [self.do_work(f) for f in flist]
        else:
            p = multiprocessing.Pool(processes=n_threads)
            p.map(self.do_work, flist)

    def get_work_function(self, v=1, dry_run=False):
        """
        Make a version of self.full_work_function, with hidden input args.

        self.full_work_function(filename, verbosity, dry_run)
        self.short_work_function(filename)
        """

        def short_work_function(filename):
            vprint(v, 3, ('Entering work function with v={}, dry_run={}, ' +
                   'loadfile={}').format(v, dry_run, filename))

            # construct the relevant file paths and names
            loadfile = os.path.join(self.loadpath, filename)
            # if savepath == loadpath,
            # the glob will try to load test_24_save.h5 as well as test_24.h5.
            # It will throw a GlobError. Catch it and skip the file.
            try:
                if self.doneflag:
                    donefile = os.path.join(
                        self.donepath, self.donefilefunc(filename))
                    vprint(v, 4, 'Donefile is {}'.format(donefile))
                if self.phflag:
                    phfile = os.path.join(
                        self.phpath, self.phfilefunc(filename))
                    vprint(v, 4, 'Placeholder is {}'.format(phfile))
                if not self.in_place_flag:
                    savefile = os.path.join(
                        self.savepath, self.savefilefunc(filename))
                    vprint(v, 4, 'Savefile is {}'.format(savefile))
            except GlobError:
                vprint(v, 3, 'Glob mismatch on {}, skipping at {}'.format(
                    filename, time.ctime()))
                return None

            # skip?
            if self.doneflag and os.path.exists(donefile):
                vprint(v, 2, 'Found donefile {}, skipping at {}'.format(
                    donefile, time.ctime()))
                return None
            if self.phflag and os.path.exists(phfile):
                vprint(v, 2, 'Found placeholder {}, skipping at {}'.format(
                    phfile, time.ctime()))
                return None
            if not self.in_place_flag and os.path.exists(savefile):
                vprint(v, 2, 'Found savefile {}, skipping at {}'.format(
                    savefile, time.ctime()))
                return None

            # make placeholder
            if self.phflag:
                vprint(v, 3, 'Creating placeholder {} at {}'.format(
                    phfile, time.ctime()))
                with open(phfile, 'w') as phf:
                    phf.write('placeholder')
            # perform work
            vprint(v, 1, 'Starting {} at {}'.format(loadfile, time.ctime()))
            if self.in_place_flag:
                self.full_work_function(loadfile, verbosity=v, dry_run=dry_run)
            else:
                self.full_work_function(
                    loadfile, savefile, verbosity=v, dry_run=dry_run)

            vprint(v, 2, 'Finished {} at {}'.format(loadfile, time.ctime()))
            # finished
            if self.doneflag:
                vprint(v, 3, 'Writing donefile {} at {}'.format(
                    donefile, time.ctime()))
                with open(donefile, 'w') as df:
                    df.write('completed')
            if self.phflag:
                vprint(v, 3, 'Removing placeholder {} at {}'.format(
                    phfile, time.ctime()))
                os.remove(phfile)

            vprint(v, 4, 'Exiting work function from {} at {}'.format(
                loadfile, time.ctime()))

        return short_work_function

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
            rm_list = glob.glob(os.path.join(self.savepath, self.saveglob))
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


def test_work(loadfile, savefile, verbosity=1, dry_run=False):
    print(
        '+ test_work loadfile={} savefile={} verbosity={} dry_run={} +'.format(
            loadfile, savefile, verbosity, dry_run))
    sleeptime = 4 + np.random.random() * 4
    time.sleep(sleeptime)
    with open(savefile, 'w') as s:
        s.write(' ')
    print(
        '/ test_work loadfile={} savefile={} verbosity={} dry_run={} /'.format(
            loadfile, savefile, verbosity, dry_run))


def create_test_files(writepath, writeglob, n):
    if not os.path.isdir(writepath):
        os.mkdir(writepath)
    for i in range(n):
        filename = os.path.join(
            writepath, put_glob_content([str(i)], writeglob))
        with open(filename, 'w') as f:
            f.write(' ')


def remove_test_files(rmpath, rmglob, n):
    for i in range(n):
        filename = os.path.join(rmpath, put_glob_content(str(i), rmglob))
        os.remove(filename)


def test_run_job():
    v = 4

    # single thread, separate dirs, default settings
    test_job('./testload', 'test_*.h5',
             './testsave', 'test_*_save.h5',
             20, {},
             {'verbosity': v})

    #


def test_job(loadpath, loadglob,
             savepath, saveglob,
             n, handler_kwargs, start_kwargs):
    # setup
    create_test_files(loadpath, loadglob, n)
    create_test_files(savepath, saveglob, 0)

    # "real work"
    jh = JobHandler(
        test_work,
        loadpath=loadpath, loadglob=loadglob,
        savepath=savepath, saveglob=saveglob,
        **handler_kwargs)
    jh.start(**start_kwargs)

    # teardown
    jh.remove_all_files(i_am_sure=True, totally_sure=True)


if __name__ == '__main__':
    import numpy as np

    test_isstrlike()
    test_split_glob()
    test_get_glob_content()
    test_put_glob_contents()
    test_get_filename_function()

    test_run_job()
