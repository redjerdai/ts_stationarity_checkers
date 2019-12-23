# Time Series Stationarity Checkers v.0.0.1
# @author: redjerdai


import numpy
import itertools
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import pacf


# remake all of these methods to a single class!

def check_stationarity_adf(array, significance):

    # Augmented Dickey-Fuller test (ADF-test)

    # Wiki Article:     https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test
    # From Docs:        https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html

    test_detrended_result = adfuller(x=array, regression='c')
    test_trended_result = adfuller(x=array, regression='ct')

    if test_detrended_result.pvalue <= test_trended_result.pvalue:

        # we suppose that results for detrended are better than for trended

        if test_detrended_result.pvalue < significance:

            # stationary without trend

            result = 'N'

        else:

            # non stationary

            result = 'U'

    else:

        # we suppose that results for trended are better than for detrended

        if test_trended_result.pvalue < significance:

            # stationary with trend

            result = 'T'

        else:

            # non stationary

            result = 'U'

    #
    return result


def check_stationarity_kpss(array, significance):

    # Kwiatkowski–Phillips–Schmidt–Shin test (KPSS-test)

    # Wiki Article:     https://en.wikipedia.org/wiki/KPSS_test
    # From Docs:        https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.kpss.html

    test_detrended_result = kpss(x=array, regression='c')
    test_trended_result = kpss(x=array, regression='ct')

    if test_detrended_result.p_value >= test_trended_result.p_value:

        # we suppose that results for detrended are better than for trended

        if test_detrended_result.p_value >= significance:

            # stationary without trend

            result = 'N'

        else:

            # non stationary

            result = 'U'

    else:

        # we suppose that results for trended are better than for detrended

        if test_trended_result.p_value >= significance:

            # stationary with trend

            result = 'T'

        else:

            # non stationary

            result = 'U'

    #
    return result


def check_stationarity_with_arima(array, significance):

    # Testing with ARIMA model

    # Wiki Article:     https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average
    # From Docs:        https://www.statsmodels.org/dev/generated/statsmodels.tsa.arima_model.ARIMA.html

    # the main idea of this method is the following
    # at first, we create search space with possible arima params p, i and q
    # then we loop them all fitting arima models
    # we select as the best the model with the minimal information criterion value (now aic is used)
    # then we take a look at it's i parameter:
    # if it is equal to zero we suppose that the time series are stationary
    # otherwise they are said to be non stationary
    # that's all folks!

    # note: in the following implementation arimas consider only detrended case
    # so trended case should be added in future!

    # create search space
    p_variants, i_variants, q_variants = numpy.array(numpy.arange(5)), numpy.array(numpy.arange(2)), numpy.array(numpy.arange(5))
    search_space = list(itertools.product(p_variants, i_variants, q_variants))

    # loop through arimas
    information_criterias = []
    for j in numpy.arange(len(search_space)):
        model = ARIMA(endog=array, order=search_space[j])
        model = model.fit(trend='c')
        information_criterias.append(model.aic)

    # gain order of integration of the best model
    the_i = search_space[information_criterias.index(min(information_criterias))][1]

    if the_i == 0:

        # stationary without trend

        result = 'N'

    else:

        # non stationary

        result = 'U'

    # we have to check significance! add it!

    #
    return result


def check_stationarity_with_pacf(array, significance):

    # Testing for stationarity with a Partial Autocorrelation Function

    # Wiki Article:     https://en.wikipedia.org/wiki/Partial_autocorrelation_function
    # From Docs:        https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.pacf.html

    # compute pacf
    pacf_values = pacf(x=array)

    # pls check if pacf applicable at all for stationarity problems
    # pls check if you use significance correctly

    # add a case for trended!

    # find the maximal pike order
    mx_pike_order = len(pacf_values) - 1
    mx_pike_found = False
    while not mx_pike_found:
        if mx_pike_found > 0 and abs(pacf_values[mx_pike_order]) < significance:
            mx_pike_order = mx_pike_order - 1
        else:
            mx_pike_found = True

    if mx_pike_found == 0:

        # stationary without trend

        result = 'N'

    else:

        # non stationary

        result = 'U'

    return result

# add checker with spectral density
