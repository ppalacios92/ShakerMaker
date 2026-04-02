import numpy as np
import abc
import scipy.signal as sig
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# GPU acceleration support (optional, requires CuPy)
# ---------------------------------------------------------------------------
_GPU_AVAILABLE = False
_cp = None
_gpu_fftconvolve = None

def _init_gpu():
    """Initialize GPU support. Called once when use_gpu=True is first requested."""
    global _GPU_AVAILABLE, _cp, _gpu_fftconvolve
    if _cp is not None:
        return _GPU_AVAILABLE
    try:
        import cupy as cp
        from cupyx.scipy.signal import fftconvolve as gpu_fftconv
        _cp = cp
        _gpu_fftconvolve = gpu_fftconv
        _GPU_AVAILABLE = True
        # Test GPU is actually accessible
        _ = cp.array([1.0])
        return True
    except (ImportError, Exception) as e:
        _GPU_AVAILABLE = False
        return False

def gpu_available():
    """Check if GPU acceleration is available."""
    return _init_gpu()


class SourceTimeFunction(metaclass=abc.ABCMeta):

    def __init__(self, dt=-1):
        self._dt = dt
        self._data = None
        self._t = None

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        assert value > 0, "SourceTimeFunction - dt must be > 0. Got dt = {}".format(value)

        self._dt = value
        self._generate_data()

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

    @abc.abstractmethod
    def _generate_data(self):
        raise NotImplementedError('derived class must define method generate_data')

    def convolve(self, val, t, debug=False, use_gpu=False):
        """Convolve signal with the source time function.
        
        :param val: Input signal to convolve.
        :type val: numpy array
        :param t: Time vector corresponding to val.
        :type t: numpy array
        :param debug: If True, plot debug figures.
        :type debug: bool
        :param use_gpu: If True, use GPU acceleration (requires CuPy).
        :type use_gpu: bool
        :returns: Convolved signal.
        :rtype: numpy array
        """
        if len(self.data) == 1:
            val_stf = val * self.data[0]
        else:
            dt_old = t[1] - t[0]
            dt_new = self.dt
            t_resampled = np.arange(t[0], t[-1], dt_new)
            val_resampled = interp1d(t, val, bounds_error=False, fill_value=(val[0], val[-1]))(t_resampled)
            
            # Use FFT-based convolution (O(n log n) vs O(n^2))
            # GPU path if requested and available
            if use_gpu and _init_gpu():
                val_resampled_gpu = _cp.asarray(val_resampled)
                data_gpu = _cp.asarray(self.data)
                conv_result = _gpu_fftconvolve(val_resampled_gpu, data_gpu, mode="full")
                val_stf_resampled = _cp.asnumpy(conv_result)[0:len(val_resampled)] * dt_new
            else:
                # CPU FFT convolution (still much faster than direct convolution)
                val_stf_resampled = sig.fftconvolve(val_resampled, self.data, mode="full")[0:len(val_resampled)] * dt_new
            
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

    def convolve_batch(self, signals, t, use_gpu=False):
        """Convolve multiple signals with the source time function in batch.
        
        This is more efficient than calling convolve() multiple times because
        it reuses the resampled STF and time vectors.
        
        :param signals: List of input signals to convolve [z, e, n].
        :type signals: list of numpy arrays
        :param t: Time vector corresponding to signals.
        :type t: numpy array
        :param use_gpu: If True, use GPU acceleration (requires CuPy).
        :type use_gpu: bool
        :returns: List of convolved signals.
        :rtype: list of numpy arrays
        """
        n_signals = len(signals)
        
        if len(self.data) == 1:
            # Dirac delta case: simple scaling
            return [sig * self.data[0] for sig in signals]
        
        dt_old = t[1] - t[0]
        dt_new = self.dt
        t_resampled = np.arange(t[0], t[-1], dt_new)
        n_resampled = len(t_resampled)
        
        # Batch resample all signals at once using matrix operations
        signals_resampled = np.empty((n_signals, n_resampled), dtype=np.float64)
        for i, s in enumerate(signals):
            signals_resampled[i, :] = interp1d(t, s, bounds_error=False, fill_value=(s[0], s[-1]))(t_resampled)
        
        # Batch convolution
        if use_gpu and _init_gpu():
            # GPU batch convolution
            signals_gpu = _cp.asarray(signals_resampled)
            data_gpu = _cp.asarray(self.data)
            results = []
            for i in range(n_signals):
                conv_result = _gpu_fftconvolve(signals_gpu[i], data_gpu, mode="full")
                results.append(_cp.asnumpy(conv_result)[0:n_resampled] * dt_new)
            conv_resampled = np.array(results)
        else:
            # CPU FFT batch convolution
            conv_resampled = np.empty((n_signals, n_resampled), dtype=np.float64)
            for i in range(n_signals):
                conv_resampled[i, :] = sig.fftconvolve(signals_resampled[i], self.data, mode="full")[0:n_resampled] * dt_new
        
        # Batch resample back to original time grid
        results = []
        for i in range(n_signals):
            val_stf = interp1d(t_resampled, conv_resampled[i], bounds_error=False, 
                              fill_value=(signals[i][0], signals[i][-1]))(t)
            results.append(val_stf)
        
        return results
