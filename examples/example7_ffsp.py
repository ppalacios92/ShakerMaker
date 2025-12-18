from shakermaker.crustmodel import CrustModel
from shakermaker.ffspsource import FFSPSource

# Model crustal
# thickness, vp, vs, rho, Qa, Qb
crustal = CrustModel(4)    
vp,vs,rho,thick,Qa,Qb=1.32,   0.75,   2.4000 , 0.200,   1000.00000,   1000.00000
crustal.add_layer(thick, vp, vs, rho, Qa, Qb)
vp,vs,rho,thick,Qa,Qb=2.75,   1.57,   2.5000 , 0.800,   1000.00000,   1000.00000
crustal.add_layer(thick, vp, vs, rho, Qa, Qb)
vp,vs,rho,thick,Qa,Qb=5.50000,   3.140000,   2.5000 , 14.50000,   1000.00000,   1000.00000
crustal.add_layer(thick, vp, vs, rho, Qa, Qb)
vp,vs,rho,thick,Qa,Qb=7.00000,   4.000000,   2.6700 , 0.000000,   1000.00000,   1000.00000
crustal.add_layer(thick, vp, vs, rho, Qa, Qb)

# Create FFSP source
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
    id_ran1=1,  id_ran2=1,
    angle_north_to_x=0.0,
    is_moment=3,
    crust_model=crustal,
    output_name="FFSP_OUTPUT",
    verbose=True,
)
# Run FFSP
subfaults = source.run()
# Write results in a .h5 file
source.write_hdf5('results.h5')
# Write results in classic FFSP format
source.write_ffsp_format('FFSP_OUTPUT')