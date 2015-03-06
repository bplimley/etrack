# ccd_cal.py

import numpy as np
import math
from astropy.io import fits
import types

class CcdImage:

    def __init__(self, filename=None):
        """Initialize by loading a FITS image file."""

        if type(filename) is types.NoneType:
            pass
        else:
            hdulist = fits.open(filename)
            # Assume file contains only one HDU object
            self.header = hdulist[0].header
            self.unpack_header()
            self.raw = np.uint16(hdulist[0].data)


    def unpack_header(self):
        """Unpack shape, timestamp, and exposure_time from header."""

        # x,y consistent with MATLAB and python indexing
        self.raw_shape = (self.header['NAXIS2'], self.header['NAXIS1'])
        self.timestamp = self.header['DATE-OBS']
        self.exposure_time = self.header['EXPTIME']


    def define_quadrants(self,layout=None):
        """Define quadrant areas, to correspond to gain values.

        Quadrant areas are defined by the following attributes:
          quadrant_positions: (((x1,x2),(y1,y2)), ((x1,x2),(y1,y2)), ...)
          position_offsets: ((xoffset,yoffset), (xoffset,yoffset), ...)
          cal_shape: (xsize,ysize)
        Currently, all these attributes are manually defined, although
        position_offsets and cal_shape could in principle be computed
        from quadrant_positions

        For quadrant qi, the position transformation is:
          qp = quadrant_positions
          po = position_offsets
          cal[ qp[qi][0][0]-po[qi][0] : qp[qi][0][1]-po[qi][0],
               qp[qi][1][0]-po[qi][1] : qp[qi][1][1]-po[qi][1] ] = raw[
               qp[qi][0][0]           : qp[qi][0][1],
               qp[qi][1][0]           : qp[qi][1][1] ]
        """

        if type(layout) is types.NoneType:
            # auto detect...
            if self.raw_shape==(1454,726):
                layout = 'CCD1'
            elif self.raw_shape==(3518,3522):
                layout = 'Yigong'

        if layout.upper() == 'CCD1':
            # mini-SNAP size
            # assume device CCD1 (used in coincidence experiment, etc.)
            # convention is that gains start at "top left" and go clockwise
            # ... don't forget python 0-indexing
            self.quadrant_positions = (
                ( (727,1454), (0,363)),
                ( (727,1454), (363,726)),
                ( (0,727), (363,726)),
                ( (0,727), (0,363)))
            self.position_offsets = (
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0))
            self.cal_shape = (1454, 726)
        elif layout.upper() == 'YIGONG':
            # full SNAP
            # assume beam diagnostics device in Yigong's cryostat
            # two-corner readout
            # includes buffers around readout areas
            self.quadrant_positions = [
                [slice(2,1755), slice(12,3518)],
                [slice(1763,3516), slice(12,3518)]]
            self.position_offsets = [
                [2, 12],
                [10, 12]]
            self.cal_shape = (3506, 3506)
        else:
            raise LookupError, "Quadrant layout preset `" + layout + "` not found"


    def calibrate(self, median_image=None):
        """Calibrate image.

        median_image should be supplied as input.
        Otherwise, only the median of each quadrant will be subtracted.
        """

        assert hasattr(self,'gains')
        assert hasattr(self,'quadrant_positions')
        if type(median_image) is types.NoneType:
            median_image = np.zeros(self.raw.shape)

        # self.cal = np.float32(self.raw.copy())
        # self.cal -= np.float32(median_image)
        self.cal = np.float32(np.zeros(self.cal_shape))

        qp = self.quadrant_positions
        po = self.position_offsets
        batch_bl_offset = np.zeros(len(qp))
        bl_offset_adjust = np.zeros(len(qp))
        self.blacklevel_offsets = np.zeros(len(qp))
        for qi in range(len(qp)):
            # set up slices for position transformation
            raw_slice_inds = ((qp[qi][0][0], qp[qi][0][1]),
                              (qp[qi][1][0], qp[qi][1][1]))
            raw_slices = (
                slice(raw_slice_inds[0][0], raw_slice_inds[0][1]),
                slice(raw_slice_inds[1][0], raw_slice_inds[1][1]))
            cal_slice_inds = (
                (raw_slice_inds[0][0] - po[qi][0],
                 raw_slice_inds[0][1] - po[qi][0]),
                (raw_slice_inds[1][0] - po[qi][1],
                 raw_slice_inds[1][1] - po[qi][1]))
            cal_slices = (
                slice(cal_slice_inds[0][0], cal_slice_inds[0][1]),
                slice(cal_slice_inds[1][0], cal_slice_inds[1][1]))
            # move quadrant into cal image
            self.cal[cal_slices[0],cal_slices[1]] = self.raw[
                raw_slices[0],raw_slices[1]]
            # subtract median image
            self.cal[cal_slices[0],cal_slices[1]] -= median_image[
                raw_slices[0],raw_slices[1]]
            # measure black level offsets for batch and image in ADC units
            batch_bl_offset[qi] = np.median(
                median_image[raw_slices[0], raw_slices[1]])
            bl_offset_adjust[qi] = np.median(
                self.cal[cal_slices[0],cal_slices[1]])
            self.blacklevel_offsets[qi] = sum(
                [batch_bl_offset[qi], bl_offset_adjust[qi]])
#             print("median image:")
#             print(batch_bl_offset[qi])
#             print("this quadrant:")
#             print(bl_offset_adjust[qi])
            # subtract quadrant median
            self.cal[cal_slices[0],cal_slices[1]] -= bl_offset_adjust[qi]
            # apply gain
            self.cal[cal_slices[0],cal_slices[1]] *= self.gains[qi]


def compute_median_image(these_images):
    """Compute the median image out of one batch.

    these_images should be a list of CcdImage objects.

    ***needs work***
    """

    image_block = [these_images[i].raw for i in len(these_images)]

    median_image = np.median(image_block,axis=3)

    return median_image


def test_cal(filename):
    """Test script for calibration"""

    # 2014-12-4 on T400: '~/Dropbox/python/cs137_00511.fits'
    img = CcdImage(filename)
    img.gains = [1,1,1,1]
    img.define_quadrants()
    img.calibrate()

    return img
