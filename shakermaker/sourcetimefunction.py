import numpy as np
import abc
import scipy.signal as sig
from scipy.interpolate import interp1d
from scipy.fft import fft, irfft, next_fast_len


def _get_fft_kernel_size(n_signal, kernel_len):
    return next_fast_len(n_signal + kernel_len - 1)


class SourceTimeFunction(metaclass=abc.ABCMeta):

    def __init__(self, dt=-1):
        self._dt = dt
        self._data = None
        self._t = None
        self._fft_kernel = None
        self._fft_kernel_size = None
        self._fft_kernel_valid_for_dt = None

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        assert value > 0, "SourceTimeFunction - dt must be > 0. Got dt = {}".format(value)

        self._dt = value
        self._generate_data()
        self._fft_kernel = None

    @property
    def data(self):
        if self._data is None:
            self._generate_data()
        return self._data

    @property
    def t(self):
        if self._t is None:
            self._generate_data()
        return self._t

    def _get_fft_kernel(self, n_signal):
        if self._fft_kernel is None or self._fft_kernel_valid_for_dt != self._dt:
            kernel = self.data
            n_fft = _get_fft_kernel_size(n_signal, len(kernel))
            self._fft_kernel = fft(kernel, n=n_fft)
            self._fft_kernel_size = n_fft
            self._fft_kernel_valid_for_dt = self._dt
        return self._fft_kernel, self._fft_kernel_size

    @abc.abstractmethod
    def _generate_data(self):
        raise NotImplementedError('derived class must define method generate_data')

    def convolve(self, val, t, debug=False):
        if len(self.data) == 1:
            val_stf = val*self.data[0]
        else:
            dt_old = t[1] - t[0]
            dt_new = self.dt
            t_resampled = np.arange(t[0], t[-1], dt_new)
            val_resampled = interp1d(t, val, bounds_error=False, fill_value=(val[0], val[-1]))(t_resampled)
            val_stf_resampled = sig.convolve(val_resampled, self.data, mode="full")[0:len(val_resampled)] * dt_new 
            val_stf = interp1d(t_resampled, val_stf_resampled, bounds_error=False, fill_value=(val[0], val[-1]))(t)
            if debug:
                import matplotlib.pylab as plt

                plt.figure(1)
                plt.plot(t, val, label="original val")
                plt.plot(t_resampled, val_resampled, ".", label="resampled val")

                t1 = t_resampled[val_stf_resampled.argmax()]
                t2 = self.t[self.data.argmax()]
                plt.legend()

                plt.figure(2)
                plt.plot(t, val,  label="original val")
                plt.plot(t_resampled, val_resampled, ".", label="original val")
                plt.plot(t_resampled, val_stf_resampled, ".", label="convolved resampled val")
                plt.plot(t, val_stf, label="convolved val")
                plt.plot(self.t-t2+t1, self.data, label="original STF")
                plt.legend()
                plt.show()

        return val_stf

    def convolve_fft(self, val, t):
        if len(self.data) == 1:
            return val * self.data[0]
        
        dt_new = self.dt
        n_signal = len(val)
        fft_kernel, n_fft = self._get_fft_kernel(n_signal)
        
        val_fft = fft(val, n=n_fft)
        val_conv_fft = val_fft * fft_kernel
        val_conv = irfft(val_conv_fft, n=n_fft)[:n_signal]
        return val_conv * dt_new

    def convolve_batch(self, seis, t):
        if len(self.data) == 1:
            return seis * self.data[0]
        
        dt_new = self.dt
        n_signal = seis.shape[-1]
        fft_kernel, n_fft = self._get_fft_kernel(n_signal)
        
        seis_fft = fft(seis, n=n_fft, axis=-1)
        seis_conv_fft = seis_fft * fft_kernel
        seis_conv = irfft(seis_conv_fft, n=n_fft, axis=-1)
        
        result = seis_conv[..., :n_signal] * dt_new
        
        boundaries = (seis[..., 0], seis[..., -1])
        
        return result

    def convolve_batch_with_resample(self, seis, t, t_resampled):
        if len(self.data) == 1:
            return seis * self.data[0]
        
        n_comp = seis.shape[0]
        n_new = len(t_resampled)
        
        dt_new = self.dt
        fft_kernel, n_fft = self._get_fft_kernel(n_new)
        
        val_resampled = np.zeros((n_comp, n_new), dtype=np.float64)
        for i in range(n_comp):
            val_resampled[i] = interp1d(t, seis[i], bounds_error=False, 
                                        fill_value=(seis[i, 0], seis[i, -1]))(t_resampled)
        
        seis_fft = fft(val_resampled, n=n_fft, axis=-1)
        seis_conv_fft = seis_fft * fft_kernel
        seis_conv = irfft(seis_conv_fft, n=n_fft, axis=-1)
        val_stf_resampled = seis_conv[..., :n_new] * dt_new
        
        val_stf = np.zeros((n_comp, len(t)), dtype=np.float64)
        for i in range(n_comp):
            val_stf[i] = interp1d(t_resampled, val_stf_resampled[i], bounds_error=False,
                                  fill_value=(seis[i, 0], seis[i, -1]))(t)
        
        return val_stf[0], val_stf[1], val_stf[2]
