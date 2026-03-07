import numpy as np
from shakermaker.sourcetimefunction import SourceTimeFunction

class SRF2(SourceTimeFunction):
    """
    The SRF2 Source Time Function

    Implements the provided srf2 functional form as a slip rate function.
    
    Parameters:
        Tr, Tp, Te : float
            The parameter from the srf2 function.
        dt : float
            Time step for the source time function.
    """
    def __init__(self, Tr, Tp, Te, dt, a, b):
        SourceTimeFunction.__init__(self)
        self._Tp = Tp
        self._Tr = Tr
        self._Te = Te
        self._dt = dt
        self._a = a
        self._b = b

    
    def _generate_data(self):
        t, svf = self._srf2_function()
        self._t = t
        self._data = svf

    def _srf2_function(self):
        a = self._a
        b = self._b

        t = np.arange(0, self._Tr, self._dt)
        Nt = len(t)
        svf = 0 * t

        i1 = t < self._Tp
        svf[i1] = t[i1] / self._Tp * np.sqrt(a + b / self._Tp**2) * np.sin(np.pi * t[i1] / (2 * self._Tp))
        i2 = np.logical_and(self._Tp <= t, t < self._Te)
        svf[i2] = np.sqrt(a + b / t[i2]**2)
        i3 = t >= self._Te
        svf[i3] = np.sqrt(a + b / t[i3]**2) * np.sin(5/3 * np.pi * (self._Tr - t[i3]) / self._Tr)

        A = np.trapz(svf, dx=self._dt)

        svf /= A

        return t, svf

# Register the new SRF2 class with SourceTimeFunction
SourceTimeFunction.register(SRF2)
