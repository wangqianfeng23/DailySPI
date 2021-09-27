import numpy as np
from lmoments3 import distr
from scipy.stats import gamma
import scipy.stats
from statsmodels.sandbox.regression import gmm
from rpy2.robjects.packages import importr
import rpy2.robjects as r2
import logging
stats=importr('stats')

# ------------------------------------------------------------------------------
#set up a basic, global _logger
def get_logger(name, level):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d  %H:%M:%S",
    )
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger

# ------------------------------------------------------------------------------
def scale_values(
        values: np.ndarray,
        scale: int,
        periodicity: str,
):
    # we expect to operate upon a 1-D array, so if we've been passed a 2-D array
    # then we flatten it, otherwise raise an error
    shape = values.shape
    if len(shape) == 2:
        values = values.flatten()
    elif len(shape) != 1:
        message = "Invalid shape of input array: {shape}".format(shape=shape) + \
                  " -- only 1-D and 2-D arrays are supported"
        get_logger.error(message)
        raise ValueError(message)

    # if we're passed all missing values then we can't compute
    # anything, so we return the same array of missing values
    if (np.ma.is_masked(values) and values.mask.all()) or np.all(np.isnan(values)):
        return values

    # clip any negative values to zero
    if np.amin(values) < 0.0:
        get_logger.warn("Input contains negative values -- all negatives clipped to zero")
        values = np.clip(values, a_min=0.0, a_max=None)

    # get a sliding sums array, with each time step's value scaled
    # by the specified number of time steps
    scaled_values = sum_to_scale(values, scale)

    # reshape precipitation values to (years, 12) for monthly,
    # or to (years, 366) for daily
    if periodicity == 'daily':

        scaled_values = scaled_values.reshape((int(scaled_values.shape[0] / 365), 365))

    elif periodicity == 'monthly':

        scaled_values = scaled_values.reshape((int(scaled_values.shape[0] / 12), 12))

    else:

        raise ValueError("Invalid periodicity argument: %s" % periodicity)

    return scaled_values

# ------------------------------------------------------------------------------
def sum_to_scale(
        values: np.ndarray,
        scale: int,
) -> np.ndarray:
    """
    Compute the moving average according to the given scale
    values:A one-dimensional array that needs to be convolved
    scale:Scale integer, it can be 30,60,90,120...
    """

    # don't bother if the number of values to sum is 1
    if scale == 1:
        return values

    # get the valid sliding summations with 1D convolution
    sliding_sums = np.convolve(values, np.ones(scale), mode="valid")

    # pad the first (n - 1) elements of the array with NaN values
    return np.hstack(([np.NaN] * (scale - 1), sliding_sums))/scale

# ------------------------------------------------------------------------------
def fit_gamma_para(
        x: np.ndarray,
        p0='TRUE',
        mass='TRUE',
        fix='TRUE',
):
    #The maximum likelihood method is used to estimate the parameters of the Gamma distribution
    #x:Value that has been convolved and reshape
    # and more distributions will be explored in the future.
    # p0，mass，and fix:The default Settings are used and more will be explored in the future
    b=x

    #Initial parameters
    result_alpa = np.zeros((365)) - 999
    result_beta = np.zeros((365)) - 999

    #Maximum likelihood fitting parameter
    f_alpa = np.zeros((365)) - 999
    f_beta = np.zeros((365)) - 999

    #The probability of zero
    f_P0 = np.zeros((365)) - 999

    r2.r(
        '''
        fnobj <- function(par, obs, ddistnam){
   -sum(do.call(ddistnam, c(list(obs), as.list(par), log=TRUE) ) )
   }
       '''
    )

    for i in range(0, 365):
        now = b[:, i]
        now = now[now == now]

        #If there is no data, the value is null
        if now.shape[0] == 0:
            result_alpa[i] = np.nan
            result_beta[i] = np.nan
            f_alpa[i] = np.nan
            f_beta[i] = np.nan
            f_P0[i] = np.nan
            continue

        # If all values are the same, the value is null
        elif np.all(now == now[0]):
            result_alpa[i] = np.nan
            result_beta[i] = np.nan
            f_alpa[i] = np.nan
            f_beta[i] = np.nan
            f_P0[i] = np.nan
            continue

        #Calculate the probability of zero
        npo = now[now == 0].shape[0]
        nn = now.shape[0]
        if mass:
            est = npo / (nn + 1)
        else:
            est = npo / nn

        #If 0 exists and the non-zero values are greater than three, the non-zero values are picked out for likelihood estimation.
        #Add a value close to zero at the end to prevent gaps between the data
        # Initial parameters were estimated by L-moments
        if est > 0 and (nn - npo > 3):
            now = now[now > 0]
            now = np.append(now, np.min(now) * 0.01)
            paras = distr.gam.lmom_fit(now)
            result_alpa[i] = paras['a']
            result_beta[i] = paras['scale']
            aa = r2.FloatVector(list(now))
            ss = r2.ListVector({"shape": paras['a'], "rate": paras['scale']})
            aal = stats.optim(par=ss, fn=r2.r.fnobj, obs=aa, ddistnam='dgamma')
            kkk = list(list(aal)[0])
            f_P0[i] = est
            f_alpa[i] = kkk[0]
            f_beta[i] = kkk[1]

        #If there is no value of 0, the parameter estimation is performed directly
        #Initial parameters were estimated by L-moments
        elif est == 0:
            now = now[now > 0]
            paras = distr.gam.lmom_fit(now)
            result_alpa[i] = paras['a']
            result_beta[i] = paras['scale']
            aa = r2.FloatVector(list(now))
            ss = r2.ListVector({"shape": paras['a'], "rate": paras['scale']})
            aal = stats.optim(par=ss, fn=r2.r.fnobj, obs=aa, ddistnam='dgamma')
            kkk = list(list(aal)[0])
            f_P0[i] = est
            f_alpa[i] = kkk[0]
            f_beta[i] = kkk[1]

        #If there are zero values and the number of non-zero values is less than three, the Moments method is used for initial parameter estimation
        elif est > 0 and (nn - npo <= 3) and (nn - npo >= 1):
            now = now[now > 0]
            now = np.append(now, np.min(now) * 0.01)
            n = now.shape[0]
            m = np.mean(now)
            v = np.var(now)
            shape = m ** 2 / v
            rate = m / v
            result_alpa[i] = shape
            result_beta[i] = rate
            aa = r2.FloatVector(list(now))
            ss = r2.ListVector({"shape": paras['a'], "rate": paras['scale']})
            aal = stats.optim(par=ss, fn=r2.r.fnobj, obs=aa, ddistnam='dgamma')
            kkk = list(list(aal)[0])
            f_P0[i] = est
            f_alpa[i] = kkk[0]
            f_beta[i] = kkk[1]

        #If all values are 0, the argument is null
        elif est > 0 and (nn == npo):
            result_alpa[i] = np.nan
            result_beta[i] = np.nan
            f_alpa[i] = np.nan
            f_beta[i] = np.nan
            f_P0[i] = np.nan

#Returns the paras and probability of a zero value
    return f_alpa,f_beta,f_P0

# ------------------------------------------------------------------------------
def caculate_SPI(
        x: np.ndarray,
        alpha: np.ndarray,
        belta: np.ndarray,
        p0:np.ndarray,
):
    """
        Fit values to a gamma distribution and transform the values to corresponding
        normalized sigmas.

        :param x: 2-D array of values, with each row typically representing a year
                       containing twelve columns representing the respective calendar
                       months, or 366 days per column as if all years were leap years
        :param alphas: pre-computed gamma fitting parameters
        :param belta: pre-computed gamma fitting parameters
        :param p0: pre-computed gamma fitting parameters
        :return: SPI
        """
    alldata=np.zeros(x.shape)-999

    for i in range(0,365):
        nowd=x[:,i]
        nowd = nowd[nowd == nowd]
        nnn = nowd.shape[0]
        nnnp0 = nowd[nowd == 0].shape[0]
        data = gamma.cdf(nowd, a=alpha[i], scale=1 / belta[i])
        spi = p0[i] + (1 - p0[i]) * data
        spi[nowd == 0] = (nnnp0 + 1) / (2 * (nnn + 1))
        if nnn == nnnp0:
            spi[nowd == 0] = np.nan
        alldata[x.shape[0]- nnn:, i] = spi

    alldata[alldata == -999] = np.nan
    SPI = scipy.stats.norm.ppf(alldata)
    return SPI
