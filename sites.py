import math
import os
import sys

import scipy.io as io
from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.signal import medfilt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline


def get_tresp_function(channel_names):
    path = os.path.abspath('SITESpy/tresp_2019.txt')
    data = pd.read_csv(path, delimiter='\s+', index_col=False)
    data['T,K'] = np.power(10, data.TMP)
    return data[['T,K'] + channel_names]


def get_tresp_intpltd(channel_names):
    data = get_tresp_function(channel_names)
    result = {}
    for channel in channel_names:
        result[channel] = interp1d(x=data['T,K'], y=data[channel],
                                   kind='quadratic')
    return result


K_B = 1.380649E-16
S0 = 1.9E15
N0 = 1E9
P0 = 0.138


class SITES():

    def __init__(self, t_min, t_max, n_bins, channels):
        self.__t_min = t_min
        self.__t_max = t_max
        self.__n_bins = n_bins
        self.__channels = channels
        self.__t_bins, self.__dt = self.__create_log_bins()
        self.__resp = self.__get_resp_matrix()
        self.__chan_uncer = self.__calc_chan_uncer()

    def __create_log_bins(self):
        t_bins = np.linspace(np.log10(self.__t_min), np.log10(
            self.__t_max), self.__n_bins + 1)
        t_bins = np.power(10, t_bins)
        dt = [t2 - t1 for t2, t1 in zip(t_bins[1:], t_bins[:-1])]

        return t_bins, np.array(dt)

    def __get_resp_matrix(self):
        resp = get_tresp_intpltd(self.__channels)
        resp_matrix = []

        for channel in self.__channels:
            row = []
            for t_mid in [t + 0.5 * dt for t, dt in zip(self.__t_bins[:-1], self.__dt)]:
                row.append(resp[channel](t_mid))

            resp_matrix.append(row)

        return np.array(resp_matrix)

    def __calc_chan_uncer(self):
        e = {'304': 0.5, '131': 0.5, '94': 0.5, '171': 0.25,
             '193': 0.25, '211': 0.25, '335': 0.25}
        uncert = []
        for channel in self.__channels:
            df = get_tresp_function([channel])
            df[channel + 'sh'] = df[channel].shift(-1)
            df['dt'] = -df['T,K'].diff(-1)
            df['int'] = 0.5 * (df[channel + 'sh'] + df[channel]) * df['dt']
            num = df.loc[(df['T,K'] < self.__t_min) | (
                (df['T,K'] > self.__t_max)), 'int'].sum()
            den = df.loc[(df['T,K'] >= self.__t_min) & (
                (df['T,K'] <= self.__t_max)), 'int'].sum()
            er = num / den

            uncert.append(np.sqrt(e[channel]**2 + er**2))

        return np.array(uncert)

    def __get_channel_weights(self, i):
        er = self.__chan_uncer
        er_i = np.divide(1, i)
        w = np.divide(1, np.sqrt(er_i + er * er))
        return w

    def calc_intens(self, dem):
        return self.__resp @ (dem*self.__dt)

    def calc_intens_without_edge(self, dem):
        return self.__resp[:, 1:-1] @ (dem*self.__dt)[1:-1]

    def calc_dem(self, i, tol=0.1, max_iter=300, kernel_width=3.2):
        w = self.__get_channel_weights(i)
        w_sum = w.sum()
        # Calculate relative response matrix
        self.__rel_resp = (w[:, None] * self.__resp) / (w @ self.__resp)
        # Calculate matrices for calculation
        a = self.__rel_resp * (self.__resp * self.__dt)
        b = ((self.__resp*self.__resp) @ (self.__dt*self.__dt))
        ker = self.g_kernel(kernel_width)
        dem = np.zeros(self.__n_bins)
        res = i.copy()
        er = 1
        n_iter = 0
        while er > tol and n_iter < max_iter:
            delta = a * (res / b)[:, None]
            delta = delta.sum(axis=0)
            dem += self.g_filter(delta, ker)
            dem[dem <= 0] = 0
            res = i - self.calc_intens(dem)
            er = np.dot(w, np.abs(res / i)) / w_sum
            n_iter += 1

        er_resp = self.__chan_uncer
        er_i = np.divide(1, i)
        dem_err = np.sqrt(np.sum((er_i + er_resp * er_resp)[:, None] * self.__rel_resp, axis=0))
        
        return dem, er, dem_err

    def g_kernel(self, sigma):
        radius = int(4 * float(sigma) + 0.5)
        exponent_range = np.arange(1)
        sigma2 = sigma * sigma
        x = np.arange(-radius, radius+1)
        phi_x = np.exp(-0.5 / sigma2 * x ** 2)
        phi_x = phi_x / phi_x.sum()
        return phi_x[::-1]

    def g_filter(self, x, ker):
        return np.convolve(x, ker, mode='same')

    def calc_em(self, dem):
        return np.sum(dem * self.__dt)

    def calc_temp(self, dem):
        t_mid = np.array(
            [t + 0.5 * dt for t, dt in zip(self.__t_bins[:-1], self.__dt)])
        return np.sum((dem * self.__dt) * t_mid) / np.sum(dem * self.__dt)

    def calc_energy_isobaric(self, dem):
        t_mid = np.array(
            [t + 0.5 * dt for t, dt in zip(self.__t_bins[:-1], self.__dt)])
        emissions = dem * self.__dt
        energy = (emissions * (t_mid * t_mid)).sum()
        energy *= 1.5 * (K_B**2) * S0 / P0
        return energy

    def test_int(self, A=1.4e21, wt=0.9e6, tc=1.4e6):
        t = np.array(
            [t + 0.5 * dt for t, dt in zip(self.__t_bins[:-1], self.__dt)])
        dem = A * np.exp(-np.power(np.divide(t - tc, wt), 2))
        return self.calc_intens(dem), dem

    def test_int2(self, A1=1.6e21, wt1=0.35e6, tc1=0.8e6, A2=1.6E21, wt2 = 0.15e6, tc2 = 1.2e6, const=1e20):     
        t = np.array(
            [t + 0.5 * dt for t, dt in zip(self.__t_bins[:-1], self.__dt)])
        dem = A1 * np.exp(-np.power(np.divide(t - tc1, wt1), 2))

        dem += A2 * np.exp(-np.power(np.divide(t - tc2, wt2), 2))
        dem += const
        return self.calc_intens(dem), dem

    def get_temps(self):
        return np.array([t + 0.5 * dt for t, dt in zip(self.__t_bins[:-1], self.__dt)])
