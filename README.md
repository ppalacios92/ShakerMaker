![ShakerMaker](/docs/source/images/logo.png)

ShakerMaker is intended to provide a simple tool allowing earthquake engineers and seismologists to easily use the frequency-wavenumber method (FK) to produce ground-motion datasets for analysis using the Domain Reduction Method (DRM). DRM motions are stored directly into the H5DRM format.

The FK method, the core of ShakerMaker, is implemented in fortran (originally from http://www.eas.slu.edu/People/LZhu/home.html with several modifications), and interfaced with python through f2py wrappers. Classes are built on top of this wrapper to simplify common modeling tasks such as crustal model specification, generation of source faults (from simple point sources to full kinematic rupturespecifications), generating single recording stations, grids and other arrays of recording stations and stations arranged to meet the requirements of the DRM. Filtering and simple plotting tools are provided to ease model setup. 

ShakerMaker includes the Finite Fault Stochastic Process tool, (FFSP developed in Fortran), which allows for the idealization of a fault with a determined area and an associated event magnitude, with specific properties of strike, dip, and rake. Easy-to-use graphical functions are added for visualization of calculated metrics, as well as the determined statistics of the stochastic space performed to select the best model.

Finally, computation of motion traces is done by pairing all sources and all receivers, which is parallelized using MPI. This means that ShakerMaker can run on simple personal computers all the way up to large supercomputing clusters. 

Installation
------------

For now, only though the git repo::

	git clone git@github.com:jaabell/ShakerMaker.git

Use the `setup.py` script, using setuptools, to compile and install::

	sudo python setup.py install

If you dont' have sudo, you can install locally for your user with::

	sudo python setup.py install --user


Dependencies
------------

- `h5py`
- `f2py`
- `numpy`
- `scipy`
- `mpi4py` (optional but highly recommended for parallel computing of the response)
- `matplotlib` (optional, for plotting)

You can get all these packages with `pip`::

	sudo pip install mpi4py h5py f2py numpy scipy matplotlib

or, for your user::

	sudo pip install --user mpi4py f2py h5py numpy scipy matplotlib

Quickstart usage Shakermaker
----------------

Using ShakerMaker is simple. You need to specify a :class:`CrustModel` (choose from the available
predefined models or create your own), a :class:`SourceModel` (from a simple 
:class:`PointSource` to a complex fully-customizable extended source with :class:`MathyFaultPlane`) 
and, finally, a :class:`Receiver` specifying a place to record motions (and store them
in memory or text format).

In this simple example, we specify a simple strike-slip (strike=90, that is due east) 
point source at the origin and a depth of 4km, on a custom two-layer crustal model, 
and a single receiver 5km away to the north::

	from shakermaker.shakermaker import ShakerMaker
	from shakermaker.crustmodel import CrustModel
	from shakermaker.pointsource import PointSource 
	from shakermaker.faultsource import FaultSource
	from shakermaker.station import Station
	from shakermaker.stationlist import StationList
	from shakermaker.tools.plotting import ZENTPlot

	#Initialize two-layer CrustModel
	model = CrustModel(2)

	#Slow layer
	Vp=4.000			# P-wave speed (km/s)
	Vs=2.000			# S-wave speed (km/s)
	rho=2.600			# Density (gm/cm**3)
	Qp=10000.			# Q-factor for P-wave
	Qs=10000.			# Q-factor for S-wave
	thickness = 1.0		# Self-explanatory
	model.add_layer(thickness, Vp, Vs, rho, Qp, Qs)

	#Halfspace
	Vp=6.000
	Vs=3.464
	rho=2.700
	Qp=10000.
	Qs=10000.
	thickness = 0   #Zero thickness --> half space
	model.add_layer(thickness, vp, vs, rho, Qp, Qs)

	#Initialize Source
	source = PointSource([0,0,4], [90,90,0])
	fault = FaultSource([source], metadata={"name":"single-point-source"})


	#Initialize Receiver
	s = Station([0,4,0],metadata={"name":"a station"})
	stations = StationList([s], metadata=s.metadata)


These are fed into the shakermaker model class::

	model = ShakerMaker(crust, fault, stations)

Which is executed::

	model.run()

Results at the station can be readily visualized using the utility function :func:`Tools.Plotting.ZENTPlot`::

	from shakermaker.Tools.Plotting import ZENTPlot
	ZENTPlot(s, xlim=[0,60], show=True)

Yielding:

![ShakerMaker](/examples/example0_fig1.png)


## Quickstart usage FFSP tool

ShakerMaker has adopted the FFSP tool for stochastic generation of spatially-correlated fault ruptures `(Pengcheng Liu 2005 and adding the relationships found by Schmedes et al. (2010, 2013) by Chen Ji, 2020)`. To define a stochastic fault source, you need to specify the parameters in the :class:`FFSPSource` class. Since layered earth structures are required to consider stratigraphy and wave propagation properties, you also use the previously described :class:`CrustModel` class.
In this example, we generate a stochastic finite fault rupture by specifying the fault geometry, magnitude and rupture characteristics. The following parameters must be provided to :class:`FFSPSource`:

**Source Type and Frequency** - `id_sf_type`: Source type identifier (defines slip-rate function type) - `freq_min`, `freq_max`: Frequency band for ground motion simulation (Hz)

**Fault Geometry** - `fault_length`, `fault_width`: Fault plane dimensions (km) - `strike`, `dip`, `rake`: Fault orientation angles (degrees) - `pdip_max`, `prake_max`: Maximum perturbation for dip and rake angles (degrees) - `nsubx`, `nsuby`: Number of subfaults along strike and dip directions

**Hypocenter Location** - `x_hypc`, `y_hypc`, `depth_hypc`: Hypocenter position relative to fault plane (km) - `xref_hypc`, `yref_hypc`: Reference point for hypocenter coordinates

**Source Characteristics** - `magnitude`: Moment magnitude (Mw) - `fc_main_1`, `fc_main_2`: Corner frequencies for double-corner source spectrum (Hz) - `rv_avg`: Average rupture velocity (km/s) - `ratio_rise`: Ratio between peak time and rise time (tp/tr), controls slip-rate function shape

**Slip Distribution** - `nb_taper_trbl`: Taper zones [top, right, bottom, left] to smooth slip edges - `seeds`: Random seeds for stochastic slip generation - `id_ran1`, `id_ran2`: First and last index of stochastic realizations to generate (model numbers)

**Coordinate System and Output** - `angle_north_to_x`: Rotation angle of coordinate system (degrees) - `is_moment`: Flag for moment tensor calculation - `crust_model`: :class:`CrustModel` defining velocity structure for Green's functions - `work_dir`: Output directory path for results and temporary files - `cleanup`: Remove temporary files after computation (boolean) - `verbose`: Print detailed progress information (boolean)

	from shakermaker.crustmodel import CrustModel
	from shakermaker.ffspsource import FFSPSource
	
	# Model crustal
	crustal = CrustModel(3)
	# thickness, vp, vs, rho, Qa, Qb
	crustal.add_layer(15.5, 5.5, 3.14, 2.5, 1000.0, 1000.0)
	crustal.add_layer(31.5, 7.0, 4.0, 2.67, 1000.0, 1000.0)
	crustal.add_layer(0.0, 8.0, 4.57, 2.8, 1000.0, 1000.0)
	
	# Create FFSP source with all parameters from your .inp
	source = FFSPSource(
	    id_sf_type=8,  freq_min=0.01,  freq_max=24.0,
	    fault_length=30.0,   fault_width=16.0,
	    x_hypc=15.0,  y_hypc=8.0,  depth_hypc=8.0,
	    xref_hypc=0.0,  yref_hypc=0.0,
	    magnitude=6.5,  fc_main_1=0.09,  fc_main_2=3.0,
	    rv_avg=3.0,
	    ratio_rise=0.3,
	    strike=358.0,  dip=40.0,  rake=113.0,
	    pdip_max=15.0,   prake_max=30.0,
	    nsubx=256,   nsuby=128,
	    nb_taper_trbl=[5, 5, 5, 5],
	    seeds=[52, 448, 4446],
	    id_ran1=1,  id_ran2=2,
	    angle_north_to_x=0.0,
	    is_moment=3,
	    crust_model=crustal,
	    work_dir='./ffsp_output',
	    cleanup=False,  
	    verbose=True,
	)


![ShakerMaker](/examples/example1_fig1.png)
