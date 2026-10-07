"""
calc_flow contains the functions necessary to calculate optical flow in 2D and
3D, as well as a function for parsing files and their metadata.

GPU acceleration
----------------
All numerically heavy steps (separable Gaussian filtering, the Lucas-Kanade
structure-tensor solve, and the eigenvalue-based reliability map) can be run
on an NVIDIA GPU through CuPy by passing useGPU=True (or 'auto') to
calc_flow2D, calc_flow3D or process_flow. Inputs and outputs are always NumPy
arrays, so the GPU path is a drop-in replacement for the CPU path.

Requirements for the GPU path: an NVIDIA GPU, and a CuPy build matching your
CUDA toolkit, e.g.  pip install cupy-cuda12x   (or conda install -c
conda-forge cupy). Without CuPy, the code runs on the CPU exactly as before.
"""

import math
import numpy as np
from scipy.ndimage import correlate1d
from pathlib import Path
import os
import re
import tifffile as tf
import pandas as pd
from datetime import datetime
import sys
from natsort import natsorted

try:  # CuPy is optional
    import cupy as _cp
    from cupyx.scipy.ndimage import correlate1d as _cp_correlate1d
except Exception:  # ImportError, or a broken CUDA installation
    _cp = None
    _cp_correlate1d = None


###############################################################################
# Backend selection and shared helpers
###############################################################################

class _Backend:
    """Bundle of the array module and the 1D correlation routine to use."""
    def __init__(self, xp, correlate, is_gpu):
        self.xp = xp
        self.correlate = correlate
        self.is_gpu = is_gpu

    def to_cpu(self, a):
        return _cp.asnumpy(a) if self.is_gpu else a

    def release(self):
        """Return cached GPU memory to the driver (no-op on the CPU)."""
        if self.is_gpu:
            _cp.get_default_memory_pool().free_all_blocks()
            _cp.get_default_pinned_memory_pool().free_all_blocks()


_CPU_BACKEND = _Backend(np, correlate1d, False)


def gpu_available():
    """True if CuPy is importable and at least one CUDA device is visible."""
    if _cp is None:
        return False
    try:
        return _cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _get_backend(useGPU):
    """
    Resolve the useGPU option.
      False / None : CPU (NumPy/SciPy)
      True         : GPU (CuPy); raises if CuPy/CUDA is unavailable
      'auto'       : GPU if available, otherwise CPU
    """
    if useGPU is None or useGPU is False:
        return _CPU_BACKEND
    if isinstance(useGPU, str):
        if useGPU.lower() != 'auto':
            raise ValueError("useGPU must be True, False, or 'auto'")
        return _Backend(_cp, _cp_correlate1d, True) if gpu_available() else _CPU_BACKEND
    if not gpu_available():
        raise RuntimeError(
            'useGPU=True was requested but CuPy with a CUDA device is not '
            'available. Install a CuPy build matching your CUDA toolkit '
            '(e.g. pip install cupy-cuda12x) or use useGPU=False.')
    return _Backend(_cp, _cp_correlate1d, True)


def _separable(be, arr, steps):
    """
    Apply a sequence of 1D correlations. steps is a list of (kernel, axis)
    pairs applied in order, with mode='nearest' boundaries.
    """
    for kernel, axis in steps:
        arr = be.correlate(arr, kernel, axis=axis, mode='nearest')
    return arr


def _temporal_derivative_and_center(be, images, NtSlice, tFil):
    """
    Return (dtI, center) for the central time point, where dtI is the
    temporally-filtered central frame (still needing spatial smoothing) and
    center is the central frame itself, both as float64 arrays on the active
    device.

    Only the central output of the temporal filter is ever used, so when the
    kernel fits inside the stack this is a single weighted sum over a window
    of frames (a tensordot -> a cuBLAS GEMV on the GPU) and only that window
    is moved to the device. If the stack is shorter than the kernel the
    original full 'nearest'-padded correlation is used.
    """
    xp = be.xp
    Nt = images.shape[0]
    r = len(tFil) // 2
    lo, hi = NtSlice - r, NtSlice + r + 1
    tF = xp.asarray(tFil)
    if lo >= 0 and hi <= Nt:
        stack = xp.asarray(images[lo:hi], dtype=xp.float64)
        dtI = xp.tensordot(tF, stack, axes=(0, 0))
        center = stack[r].copy()
    else:
        stack = xp.asarray(images, dtype=xp.float64)
        dtI = be.correlate(stack, tF, axis=0, mode='nearest')[NtSlice]
        center = stack[NtSlice].copy()
    del stack
    return dtI, center


def _min_eig_sym3(xp, a, b, c, e, f, i):
    """
    Smallest eigenvalue of real symmetric 3x3 matrices, evaluated elementwise
    with the closed-form (trigonometric) solution. Matrix layout:
        [a b c]
        [b e f]
        [c f i]
    The structure tensor is symmetric positive semi-definite, so all
    eigenvalues are real. Being purely elementwise, this runs on the GPU with
    no per-voxel LAPACK/cuSOLVER call and no 3x3 matrix batch in memory.
    """
    q = (a + e + i) / 3.0
    p1 = b * b + c * c + f * f
    p2 = (a - q) ** 2 + (e - q) ** 2 + (i - q) ** 2 + 2.0 * p1
    p = xp.sqrt(p2 / 6.0)
    safe_p = xp.where(p > 0, p, 1.0)
    # B = (A - qI) / p ; r = det(B) / 2
    ba, be_, bi = (a - q) / safe_p, (e - q) / safe_p, (i - q) / safe_p
    bb, bc, bf = b / safe_p, c / safe_p, f / safe_p
    detB = (ba * (be_ * bi - bf * bf)
            - bb * (bb * bi - bf * bc)
            + bc * (bb * bf - be_ * bc))
    r = xp.clip(detB / 2.0, -1.0, 1.0)
    phi = xp.arccos(r) / 3.0
    lam_min = q + 2.0 * p * xp.cos(phi + 2.0 * math.pi / 3.0)
    # p == 0 means the matrix is a multiple of the identity
    return xp.where(p > 0, lam_min, q)


###############################################################################
# 2D optical flow
###############################################################################

def calc_flow2D(images,xySig=3,tSig=1,wSig=4,useGPU=False):
    """
    Calculate two-dimensional optical flow fields from input images.

    The calc_flow2D function calculates optical flow velocities for a single
    z-slice. Surrounding images in time are necessary to perform the 
    calculations. To peform calculations on an entire timelapse, see the
    function parse_flow.

    This script uses the convention that (0,0) is located in the upper-left
    corner of an image. This is inline with conventions used in other programs
    (e.g., ImageJ/FIJI), but note that it means that positive y-velocities point
    down, which can be non-intuitive in some cases.

    ARGS:
    images: 3D numpy array with dimensions N_T, N_Y, N_X
             N_T should be odd as only the central timepoint will be analyzed.
             N_T must be greater than or equal to 3*tSig+1.
    xySig:  sigma value for smoothing in all spatial dimensions. Default 3.
             Larger values remove noise but remove spatial detail.
    tSig:   sigma value for smoothing in the temporal dimension. Default 1.
             Larger values remove noise but remove temporal detail.
    wSig:   sigma value for Lucas-Kanade neighborhood. Default is 4.
             Larger values include a larger neighboorhood in the
             Lucas-Kanade constraint and will smooth over small features.
    useGPU: False (default) runs on the CPU with NumPy/SciPy. True runs all
             filtering and linear algebra on an NVIDIA GPU using CuPy (raises
             an error if CuPy/CUDA is unavailable). 'auto' uses the GPU when
             available and otherwise falls back to the CPU.

    RETURNS (always NumPy arrays on the host):
    vx:    Velocity in the x direction, reported as pixels/frame
    vy:    Velocity in the y direction, reported as pixels/frame
    rel:   Reliability, the smallest eigenvalue of (A'wA)
            This is a measure of confidence in the linear algebra solution
            and can be used to mask the velocities for downstream analysis.
    """

    ### Check the function inputs ##############################################
    # Check that the images are 2D + time
    if not(len(images.shape)==3): 
        sys.exit('ERROR: Input image must be a 3D matrix with dimensions N_T, N_Y, N_X')
    # Check image size against tSig
    Nt = images.shape[0] 
    if Nt < 6*tSig+1:
        # The kernel size is 3*tSig in time (or 6*tSig total)
        # There is also a central pixel, so need at least 6*tSig+1 images in t
        sys.exit('ERROR: Input images will lead to edge effects. N_T must be >= 6*tSig+1')
    # Check for an odd number of frames
    if not(Nt % 2):        
        sys.exit('ERROR: Input images must have an odd number of timepoints. Only the central time point is analyzed')
    NtSlice = math.ceil(Nt/2)-1 # -1 because python indexing starts from 0

    be = _get_backend(useGPU)
    xp = be.xp
    try:
        ### Set up filters (tiny; built on the host, then moved to the device) ####
        x = np.arange(-math.ceil(3*xySig),math.ceil(3*xySig)+1)
        xySig2 = xySig/4
        y = np.arange(-math.ceil(3*xySig2),math.ceil(3*xySig2)+1)
        fderiv = np.exp(-x*x/2/xySig/xySig)/math.sqrt(2*math.pi)/xySig
        fsmooth = np.exp(-y*y/2/xySig2/xySig2)/math.sqrt(2*math.pi)/xySig2
        gderiv = x/xySig/xySig

        deriv = xp.asarray(fderiv*gderiv)   # derivative kernel
        smooth = xp.asarray(fsmooth)        # light smoothing kernel
        fx = xp.asarray(fderiv)             # Gaussian used for dtI spatial smoothing

        t = np.arange(-math.ceil(3*tSig),math.ceil(3*tSig)+1)
        ft = np.exp(-t*t/2/tSig/tSig)/math.sqrt(2*math.pi)/tSig
        tFil = ft*(t/tSig/tSig)

        # Structure tensor -- Lucas Kanade neighborhood filter
        wRange = np.arange(-math.ceil(3*wSig),math.ceil(3*wSig)+1)
        gw = xp.asarray(np.exp(-wRange*wRange/2/wSig/wSig)/math.sqrt(2*math.pi)/wSig)
        W = [(gw, 0), (gw, 1)]

        ### Spatial and Temporal Gradients #####################################
        # Axis 0 = y, axis 1 = x
        dtI, im = _temporal_derivative_and_center(be, images, NtSlice, tFil)
        dtI = _separable(be, dtI, [(fx, 0), (fx, 1)])
        dyI = _separable(be, im, [(deriv, 0), (smooth, 1)])
        dxI = _separable(be, im, [(smooth, 0), (deriv, 1)])
        del im

        ### Structure Tensor Inputs ############################################
        # Gaussian weighting for the Lucas-Kanade constraint.
        wdtx = _separable(be, dxI*dtI, W)
        wdty = _separable(be, dyI*dtI, W)
        del dtI
        wdxy = _separable(be, dxI*dyI, W)
        wdx2 = _separable(be, dxI*dxI, W)
        del dxI
        wdy2 = _separable(be, dyI*dyI, W)
        del dyI

        ### Optical Flow Calculations ##########################################
        # Equation is v = (A' w A)^-1 A' w b
        # A = -[dxI dyI]
        # b = [dtI]
        # w multiplication is incorporated in the structure tensor inputs above
        # A' w b = -[wdtx wdty]  (minus sign because of negative sign on A)
        # (A' w A) = [a=wdx2 b=wdxy ; c=wdxy d=wdy2]
        # A^-1 = [a b ; c d]^-1 = (1/det(A))[d - b; -c a]
        determinant = (wdx2*wdy2) - (wdxy*wdxy)
        invdet = (determinant+np.finfo(float).eps)**-1
        vx = invdet*((wdy2*-wdtx)+(-wdxy*-wdty))
        vy = invdet*((-wdxy*-wdtx)+(wdx2*-wdty))
        del wdtx, wdty, wdxy, invdet

        ### Eigenvalues for Reliability ########################################
        # solve det(A^T w A - lamda I) = 0. The discriminant of a symmetric
        # matrix is non-negative; clamp round-off so sqrt never returns NaN.
        trace = wdx2 + wdy2
        del wdx2, wdy2
        disc = xp.maximum(trace**2 - 4*determinant, 0)
        rel = (trace - xp.sqrt(disc))/2   # smaller of the two eigenvalues
        del disc, trace, determinant

        ### Return Outputs (always host arrays) ################################
        return be.to_cpu(vx), be.to_cpu(vy), be.to_cpu(rel)
    finally:
        be.release()


###############################################################################
# 3D optical flow
###############################################################################

def calc_flow3D(images,xyzSig=3,tSig=1,wSig=4,useGPU=False):
    """
    Calculate three-dimensional optical flow fields from input z-stacks.

    The calc_flow3D function calculates optical flow velocities for a single
    z-stack of images. Surrounding z-stacks in time are necessary to perform
    the calculations. To peform calculations on an entire timelapse, see
    the function parse_flow.

    This script uses the convention that (0,0) is located in the upper-left
    corner of an image. This is inline with conventions used in other
    programs (e.g., ImageJ/FIJI), but note that it means that positive
    y-velocities point down, which can be non-intuitive in some cases.

    ARGS:
    images: 4D array with dimensions N_T, N_Z, N_Y, N_X
             N_T should be odd as only the central timepoint will be analyzed.
             N_T must be greater than or equal to 6*tSig+1.
    xyzSig:  sigma value for smoothing in all spatial dimensions. Default 3.
             Larger values remove noise but remove spatial detail.
    tSig:   sigma value for smoothing in the temporal dimension. Default 1.
             Larger values remove noise but remove temporal detail.
    wSig:   sigma value for Lucas-Kanade neighborhood. Default is 4.
             Larger values include a larger neighboorhood in the
             Lucas-Kanade constraint and will smooth over small features.
    useGPU: False (default) runs on the CPU with NumPy/SciPy. True runs all
             filtering and linear algebra on an NVIDIA GPU using CuPy (raises
             an error if CuPy/CUDA is unavailable). 'auto' uses the GPU when
             available and otherwise falls back to the CPU. On the GPU the
             3x3 eigenvalue problem is solved with a closed-form elementwise
             expression rather than a batched eigensolver, and is done in
             float64 (the CPU path uses complex64 eigvals).

    RETURNS (always NumPy arrays on the host):
    vx:    Velocity in the x direction, reported as pixels/frame
    vy:    Velocity in the y direction, reported as pixels/frame
    vz:    Velocity in the z direction, reported as pixels/frame
    rel:   Reliability, the smallest eigenvalue of (A'wA)
            This is a measure of confidence in the linear algebra solution
            and can be used to mask the velocities for downstream analysis.
    """

    ### Check the function inputs ##############################################
    # Check that the images are 3D + time
    if not(len(images.shape)==4): 
        sys.exit('ERROR: Input image must be a 4D matrix with dimensions N_T, N_Z, N_Y, N_X')
    # Check image size against tSig
    Nt = images.shape[0] 
    if Nt < 6*tSig+1:
        # The kernel size is 3*tSig in time (or 6*tSig total)
        # There is also a central pixel, so need at least 6*tSig+1 images in t
        sys.exit('ERROR: Input images will lead to edge effects. N_T must be >= 6*tSig+1')
    # Check for an odd number of frames
    if not(Nt % 2):
        sys.exit('ERROR: Input images must have an odd number of timepoints. Only the central time point is analyzed')
    NtSlice = math.ceil(Nt/2)-1 # -1 because python indexing starts from 0

    be = _get_backend(useGPU)
    xp = be.xp
    try:
        ### Set up filters (tiny; built on the host, then moved to the device) ####
        x = np.arange(-math.ceil(3*xyzSig),math.ceil(3*xyzSig)+1)
        xyzSig2 = xyzSig/4
        y = np.arange(-math.ceil(3*xyzSig2),math.ceil(3*xyzSig2)+1)
        fderiv = np.exp(-x*x/2/xyzSig/xyzSig)/math.sqrt(2*math.pi)/xyzSig
        fsmooth = np.exp(-y*y/2/xyzSig2/xyzSig2)/math.sqrt(2*math.pi)/xyzSig2
        gderiv = x/xyzSig/xyzSig

        deriv = xp.asarray(fderiv*gderiv)   # derivative kernel
        smooth = xp.asarray(fsmooth)        # light smoothing kernel
        fx = xp.asarray(fderiv)             # Gaussian used for dtI spatial smoothing

        t = np.arange(-math.ceil(3*tSig),math.ceil(3*tSig)+1)
        ft = np.exp(-t*t/2/tSig/tSig)/math.sqrt(2*math.pi)/tSig
        tFil = ft*(t/tSig/tSig)

        # Structure tensor -- Lucas Kanade neighborhood filter
        wRange = np.arange(-math.ceil(3*wSig),math.ceil(3*wSig)+1)
        gw = xp.asarray(np.exp(-wRange*wRange/2/wSig/wSig)/math.sqrt(2*math.pi)/wSig)
        W = [(gw, 1), (gw, 2), (gw, 0)]

        ### Spatial and Temporal Gradients #####################################
        # Axis 0 = z, axis 1 = y, axis 2 = x. Order of application (y, x, z)
        # matches the original implementation.
        dtI, im = _temporal_derivative_and_center(be, images, NtSlice, tFil)
        dtI = _separable(be, dtI, [(fx, 1), (fx, 2), (fx, 0)])
        dyI = _separable(be, im, [(deriv, 1), (smooth, 2), (smooth, 0)])
        dxI = _separable(be, im, [(smooth, 1), (deriv, 2), (smooth, 0)])
        dzI = _separable(be, im, [(smooth, 1), (smooth, 2), (deriv, 0)])
        del im

        ### Structure Tensor Inputs ############################################
        # Time components
        wdtx = _separable(be, dxI*dtI, W)
        wdty = _separable(be, dyI*dtI, W)
        wdtz = _separable(be, dzI*dtI, W)
        del dtI

        # Spatial Components
        wdxy = _separable(be, dxI*dyI, W)
        wdxz = _separable(be, dxI*dzI, W)
        wdx2 = _separable(be, dxI*dxI, W)
        del dxI
        wdyz = _separable(be, dyI*dzI, W)
        wdy2 = _separable(be, dyI*dyI, W)
        del dyI
        wdz2 = _separable(be, dzI*dzI, W)
        del dzI

        ### Optical Flow Calculations ##########################################
        # Equation is v = (A' w A)^-1 A' w b
        # A = -[dxI dyI dzI]
        # b = [dtI]
        # w multiplication is incorporated in the structure tensor inputs above
        # A' w b = -[wdtx wdty wdtz]  (minus sign because of negative sign on A)
        # (A' w A) = [a=wdx2 b=wdxy c=wdxz ; d=wdxy e=wdy2 f=wdyz ; g=wdxz h=wdyz i=wdz2]
        # A^-1 = 1/determinant * [A D G ; B E H ; C F I];
        # using inverse notation from here: https://en.wikipedia.org/wiki/Invertible_matrix#Inversion_of_3_%C3%97_3_matrices
        #   A = wdy2*wdz2 - wdyz*wdyz; %ei-fh
        #   B = wdyz*wdxz - wdxy*wdz2; %fg-di
        #   C = wdxy*wdyz - wdy2*wdxz; %dh-eg
        #   D = wdxz*wdyz - wdxy*wdz2; %ch-bi
        #   E = wdx2*wdz2 - wdxz*wdxz; %ai-cg
        #   F = wdxy*wdxz - wdx2*wdyz; %bg-ah
        #   G = wdxy*wdyz - wdxz*wdy2; %bf-ce
        #   H = wdxz*wdxy - wdx2*wdyz; %cd-af
        #   I = wdx2*wdy2 - wdxy*wdxy; %ae-bd
        determinant = (wdx2*wdy2*wdz2) + (2*wdxy*wdxz*wdyz) - (wdy2*wdxz**2) - (wdz2*wdxy**2) - (wdx2*wdyz**2)
        ninvdet = -((determinant + np.finfo(float).eps)**-1)
        del determinant
        vx = ninvdet*((wdy2*wdz2 - wdyz*wdyz)*wdtx + (wdxz*wdyz - wdxy*wdz2)*wdty + (wdxy*wdyz - wdxz*wdy2)*wdtz)
        vy = ninvdet*((wdyz*wdxz - wdxy*wdz2)*wdtx + (wdx2*wdz2 - wdxz*wdxz)*wdty + (wdxz*wdxy - wdx2*wdyz)*wdtz)
        vz = ninvdet*((wdxy*wdyz - wdy2*wdxz)*wdtx + (wdxy*wdxz - wdx2*wdyz)*wdty + (wdx2*wdy2 - wdxy*wdxy)*wdtz)
        del wdtx, wdty, wdtz, ninvdet

        ### Eigenvalues for Reliability ########################################
        # (A' w A) = [a=wdx2 b=wdxy c=wdxz ; d=wdxy e=wdy2 f=wdyz ; g=wdxz h=wdyz i=wdz2]
        if be.is_gpu:
            # Closed-form, elementwise: no batched eigensolver needed.
            rel = _min_eig_sym3(xp, wdx2, wdxy, wdxz, wdy2, wdyz, wdz2)
            del wdx2, wdxy, wdxz, wdy2, wdyz, wdz2
        else:
            # Original CPU behaviour (LAPACK eigvals, complex64)
            w = np.array([[wdx2, wdxy, wdxz],[wdxy, wdy2, wdyz],[wdxz, wdyz, wdz2]])
            del wdx2, wdxy, wdxz, wdy2, wdyz, wdz2
            w = np.moveaxis(w,[0,1],[-1,-2])
            w = w.astype(np.complex64) # Allow for complex eignenvalues
            rel = np.linalg.eigvals(w)
            rel = np.real(np.amin(rel,axis=-1))

        ### Return Outputs (always host arrays) ################################
        return be.to_cpu(vx), be.to_cpu(vy), be.to_cpu(vz), be.to_cpu(rel)
    finally:
        be.release()

def process_flow(imDir,imName,fileType="SequenceT",spatialDimensions=3,xyzSig=3,tSig=1,wSig=4,useGPU=False):
    """
    Parse images for input into calc_flow2D or calc_flow3D.

    Function to organize commands to calc_flow given a single image or image
    sequence for processing. The script can parse two tif formats. If your
    data is another format, this is the function to change to adapt the code
    to your uses.

    OneTif files are assumed to be created using ImageJ when reading metadata.

    ARGS:
    imDir:              Full path to the folder of image(s) to process
    imName:             File name for the image(s) to process, excluding .tif
    fileType:           Either 'OneTif' or 'SequenceT.' In the case 'OneTif',
                        the entire timelapse and all z-slices are assumed to 
                        be saved in one single .tif file, and in the format 
                        generated by ImageJ. In the case of 'SequenceT', the
                        images are assumed to be saved as a sequence, with one
                        tif per timepoint (but all z-slices saved as one tif).
                        In this case, imName must be specified with a wildcard 
                        (.*) for the time label in the file names. For a list of
                        files with names:
                            myexperiment_t000_ch0.tif
                            myexperiment_t001_ch0.tif
                            myexperiment_t002_ch0.tif
                        the correct imName would be 'myexperiment_t.*_ch0'.
    spatialDimensions:  Either 2 (2D) or 3 (3D). Default 3.
    xyzSig:             sigma value for smoothing in all spatial dimensions. Default 3.
                          Larger values remove noise but remove spatial detail.
    tSig:               sigma value for smoothing in the temporal dimension. Default 1.
                          Larger values remove noise but remove temporal detail.
    wSig:               sigma value for Lucas-Kanade neighborhood. Default is 4.
                          Larger values include a larger neighboorhood in the
                          Lucas-Kanade constraint and will smooth over small features.
    useGPU:             False (default): CPU. True: run the calculations on an
                        NVIDIA GPU with CuPy (error if unavailable). 'auto':
                        use the GPU if available, otherwise the CPU.

    RETURNS:
    This function does not have explict returns, but saves several output files.
    Output files are saved in subfolders of the input imDir.
    Output files are prefaced with the input imName.
    *_parameters.csv:   Input parameters as a comma seprated file
    *_vx.tif:           Velocity in the x direction, reported as pixels/frame
    *_vy.tif:           Velocity in the y direction, reported as pixels/frame
    *_vz.tif:           Velocity in the z direction, reported as pixels/frame
    *_rel.tif:          Reliability, the smallest eigenvalue of (A'wA)
                          This is a measure of confidence in the linear algebra solution
                          and can be used to mask the velocities for downstream analysis.
    """

    ### Check Inputs and Set Up Paths ##########################################
    # Check that the directory exists
    imDir = Path(imDir)
    if not imDir.is_dir():
        sys.exit('ERROR: image path \'%s\' does not exist' % imDir)    
    # Check that the image files exist and that the type is correct
    # First get the list of relevant images
    imNamePattern = re.compile(imName + '.tif')
    fileList = []
    files = os.listdir(imDir)
    for f in files:
        m = imNamePattern.fullmatch(f)
        if m:
            fileList.append(f)
    # Now check that the number of files makes sense
    if len(fileList) == 0:
        sys.exit('ERROR: No image files found. imName: ' + imName + ' imDir: ' + str(imDir))
    if fileType=='OneTif':
        if len(fileList) > 1:
            sys.exit('ERROR: Type is OneTif but more than one file was found for imName: ' + imName)
    elif fileType=='SequenceT':
        if len(fileList) < 6*tSig+1: # Minimum requirment for calc_flow
            sys.exit('ERROR: Image sequence found for file name ' + imName + ' only contains ' + str(len(fileList)) + ' files. Minimum 6*tsig+1 ('+ str(6*tSig+1) + ') files required.')
    else:
        sys.exit('ERROR: fileType must be either OneTif or SequenceT.')

    # Make sure files are sorted in numerical order not necessarily ASCII order
    fileList = natsorted(fileList)

    # Fail early if the requested compute backend is unavailable
    _get_backend(useGPU)

    # Must be either 2D or 3D processing
    if spatialDimensions < 2 or spatialDimensions > 3:
        sys.exit('ERROR: Number of spatial dimensions must be either 2 or 3.')

    ### Metadata parsing and parameter saving ##################################
    meta = tf.TiffFile(imDir/ fileList[0]) # Assume first file is representative of whole set
    Ny = meta.pages[0].shape[0]
    Nx = meta.pages[0].shape[1]
    imj = meta.imagej_metadata
    if fileType=='OneTif':
        if not(imj):
            sys.exit('ERROR: fileType is OneTif, but no ImageJ metadata was detected')
        Nt = imj["frames"]
        if spatialDimensions==3:
            Nz = imj["slices"]
        elif spatialDimensions==2:
            Nz = 1
    elif fileType=='SequenceT':
        Nz = len(meta.pages)
        Nt = len(fileList)
        if spatialDimensions==2:
            if Nz != 1:
                sys.exit('ERROR: More than one z-slice detected for 2D processing')
        elif spatialDimensions==3:
            if Nz <= 1:
                sys.exit('ERROR: 3D processing requested but Nz = ' + str(Nz))
    
    NtChunk = 6*tSig+1
    if not(NtChunk%2):
        NtChunk = NtChunk+1
    NtSlice = math.ceil(NtChunk/2)-1

    # Set up the saving folder    
    # Main folder is inside the image directory. 
    if spatialDimensions==3:
        savedir = imDir / 'OpticalFlow3D'
    elif spatialDimensions==2:
        savedir = imDir / 'OpticalFlow2D'
    savedir.mkdir(exist_ok=True)
    # Subfolder is imNameSave.
    imNameSave = imName.replace('.*','')
    savedir = savedir / imNameSave
    savedir.mkdir(exist_ok=True)

    # Save parameters
    param = {'xyzSig': [xyzSig],
            'tiSig': [tSig],
            'wSig': [wSig],
            'Nx': [Nx],
            'Ny': [Ny],
            'Nz': [Nz],
            'Nt': [Nt]}
    param = pd.DataFrame(param)
    savename = imNameSave + '_parameters.csv'
    param.to_csv(savedir / savename, index=False)

    ### Processing Loop ########################################################
    print('Note: regardless of input filenames, the first image = frame 0.')
    print('If your file names start from 0, adjust indexing accordingly for reading the output files.')
    print(' ')

    # Because >6*tSig time frames are necessary for processing, some frames at
    # the start and the end of the timelapse will be ignored.
    for hh in range(0,NtSlice):
        print(str(datetime.now()) + ' - No data will be saved for frame ' + str(hh) + ' to avoid edge effects')
    
    # Loop through the files to be processed
    if fileType=='OneTif': # assuming a tif made with ImageJ containing all z and all t
        imName = imName + '.tif'
        allImages = tf.memmap(imDir / imName) # Does not load images into memory until called later

        if spatialDimensions == 3:    
            for hh in range(0,Nt-NtChunk+1): # If you have enough memory, this could become a parfor loop.
            
                loopStart = datetime.now()
                print(str(datetime.now()) + ' - Processing frame ' + str(hh+NtSlice) + '...')
            
                # Load images
                images = allImages[hh:hh+NtChunk,:]

                # Run the optical flow
                vx,vy,vz,rel = calc_flow3D(images ,xyzSig, tSig, wSig, useGPU)
            
                # Save this frame
                tstr = str(hh+NtSlice)
                tstr = tstr.zfill(4)
                tf.imwrite(str(savedir / imNameSave) + '_vx_t' + tstr + '.tiff',vx, photometric='minisblack')
                tf.imwrite(str(savedir / imNameSave) + '_vy_t' + tstr + '.tiff',vy, photometric='minisblack')
                tf.imwrite(str(savedir / imNameSave) + '_vz_t' + tstr + '.tiff',vz, photometric='minisblack')
                tf.imwrite(str(savedir / imNameSave) + '_rel_t' + tstr + '.tiff',rel, photometric='minisblack')
            
                del rel, vx, vy, vz, images
            
                framestime = datetime.now()
                print(str(datetime.now()) + ' - Frame ' + str(hh+NtSlice) + ' saved.  Duration: ' + str(framestime-loopStart))
    
        elif spatialDimensions == 2:    
            for hh in range(0,Nt-NtChunk+1):
        
                loopStart = datetime.now()
                print(str(datetime.now()) + ' - Processing frame ' + str(hh+NtSlice) + '...')
            
                # Load images
                images = allImages[hh:hh+NtChunk,:]
            
                # Run the optical flow
                vx,vy,rel = calc_flow2D(images ,xyzSig, tSig, wSig, useGPU)
            
                # Save this frame
                tstr = str(hh+NtSlice)
                tstr = tstr.zfill(4)
                tf.imwrite(str(savedir / imNameSave) + '_vx_t' + tstr + '.tiff',vx, photometric='minisblack')
                tf.imwrite(str(savedir / imNameSave) + '_vy_t' + tstr + '.tiff',vy, photometric='minisblack')
                tf.imwrite(str(savedir / imNameSave) + '_rel_t' + tstr + '.tiff',rel, photometric='minisblack')

                del rel, vx, vy, images
            
                framestime = datetime.now()
                print(str(datetime.now()) + ' - Frame ' + str(hh+NtSlice) + ' saved.  Duration: ' + str(framestime-loopStart))
    
        else:
            sys.exit('ERROR: Spatial Dimension must be 2 or 3.')
    
    elif fileType=='SequenceT': # assuming 1 tif per timepoint that is a z-stack (or slice for 2D)    
        if spatialDimensions == 3:    
            for hh in range(0,Nt-NtChunk+1):
        
                loopStart = datetime.now()
                print(str(datetime.now()) + ' - Processing frame ' + str(hh+NtSlice) + '...')
            
                # Load images
                images = tf.imread(imDir / fileList[hh])
                images = np.append([images],[tf.imread(imDir / fileList[hh+1])],axis=0)
                for jj in range(2,NtChunk):
                    images = np.append(images,[tf.imread(imDir / fileList[hh+jj])],axis=0)
            
                # Run the optical flow
                vx,vy,vz,rel = calc_flow3D(images ,xyzSig, tSig, wSig, useGPU)
            
                # Save this frame
                tstr = str(hh+NtSlice)
                tstr = tstr.zfill(4)
                tf.imwrite(str(savedir / imNameSave) + '_vx_t' + tstr + '.tiff',vx, photometric='minisblack')
                tf.imwrite(str(savedir / imNameSave) + '_vy_t' + tstr + '.tiff',vy, photometric='minisblack')
                tf.imwrite(str(savedir / imNameSave) + '_vz_t' + tstr + '.tiff',vz, photometric='minisblack')
                tf.imwrite(str(savedir / imNameSave) + '_rel_t' + tstr + '.tiff',rel, photometric='minisblack')
            
                del rel, vx, vy, vz, images
            
                framestime = datetime.now()
                print(str(datetime.now()) + ' - Frame ' + str(hh+NtSlice) + ' saved.  Duration: ' + str(framestime-loopStart))

        elif spatialDimensions == 2:    
            for hh in range(0,Nt-NtChunk+1):
        
                loopStart = datetime.now()
                print(str(datetime.now()) + ' - Processing frame ' + str(hh+NtSlice) + '...')
            
                # Load images
                images = tf.imread(imDir / fileList[hh])
                images = np.append([images],[tf.imread(imDir / fileList[hh+1])],axis=0)
                for jj in range(2,NtChunk):
                    images = np.append(images,[tf.imread(imDir / fileList[hh+jj])],axis=0)
            
                # Run the optical flow
                vx,vy,rel = calc_flow2D(images ,xyzSig, tSig, wSig, useGPU)
            
                # Save this frame
                tstr = str(hh+NtSlice)
                tstr = tstr.zfill(4)
                tf.imwrite(str(savedir / imNameSave) + '_vx_t' + tstr + '.tiff',vx, photometric='minisblack')
                tf.imwrite(str(savedir / imNameSave) + '_vy_t' + tstr + '.tiff',vy, photometric='minisblack')
                tf.imwrite(str(savedir / imNameSave) + '_rel_t' + tstr + '.tiff',rel, photometric='minisblack')
            
                del rel, vx, vy, images
            
                framestime = datetime.now()
                print(str(datetime.now()) + ' - Frame ' + str(hh+NtSlice) + ' saved.  Duration: ' + str(framestime-loopStart))
    
        else:
            sys.exit('ERROR: Spatial Dimension must be 2 or 3')
    
    # Because >6*tSig time frames are necessary for processing, some frames at
    # the start and the end of the timelapse will be ignored.
    for hh in range(Nt-NtSlice,Nt):
        print(str(datetime.now()) + ' - No data will be saved for frame ' + str(hh) + ' to avoid edge effects')

