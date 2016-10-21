

import os
import h5py
import socket
import matplotlib.pyplot as plt


def run_main():
    """
    Go through the default file, show events based on user input.
    """

    loadfile = 'MultiAngle_algs_100_1.h5'
    loadfull = os.path.join(get_loadpath(), loadfile)
    prompt = '[#], [n]ext, [p]revious, [q]uit. [Enter] for next: '
    pn = 'pix10_5noise15'

    with h5py.File(loadfull, 'r') as f:
        usr_input = '0'
        fig = None

        while not usr_input.startswith('q'):
            skip = False

            # parse the input
            try:
                # assume it's a number
                indnum = int(usr_input)
            except ValueError:
                # it's a string command
                if usr_input == '' or usr_input.startswith('n'):
                    indnum += 1
                elif usr_input.startswith('p'):
                    indnum -= 1
                else:
                    print('### Unknown input')
                    skip = True
            indstr = '{:05d}'.format(indnum)

            # show the event
            if not skip:
                try:
                    print('Event {}'.format(indstr))
                    fig = show_event(indstr, *objs)
                except KeyError:
                    print('### KeyError on {}/{}'.format(indstr, pn))

            # next input
            usr_input = raw_input(prompt)

            # close previous plot if exists
            try:
                plt.close(fig)
            except TypeError:
                # fig is None
                pass

def show_event(pn, ind):
    """
    Plot stuff for one event. pn is the h5py group of the pixelnoise.
    """

    print(ind)


def get_loadpath():
    """
    Return a system-dependent path for the file to load.
    """

    if socket.gethostname() == 'plimley-Vostro-mint17':
        # LBL desktop
        loadpath = '/media/plimley/TEAM 7B/algs_10.5_batch01'
    elif socket.gethostname() == 'plimley-zenbook-mint':
        # laptop
        loadpath = '/home/plimley/Documents/research/data/algs_10.5_batch01'
    else:
        raise RuntimeError("I don't know what system I'm on")

    return loadpath


if __name__ == '__main__':
    run_main()
