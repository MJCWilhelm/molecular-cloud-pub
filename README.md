# molecular-cloud-pub
The code used in arXiv:2101.07826.

Runs a model with coupled hydrodynamics, stellar dynamics, and protoplanetary disk evolution. Stars and disks are formed dynamically from sink particles that form in regions of high gas density. Stars with masses >1.9 MSun are FUV luminous, and provide a radiation field that drives external photoevaporation in disks. Encounters between stars can also result in dynamic disk truncation.

This model requires the AMUSE package, with at least the Fi, SeBa, ph4, and VADER codes built.

Running the model is as follows:

```
python gmc_star_formation_ppds.py {-d data_directory} {-i read_index} {-s imf_seed} {-t delay_time} {-e sfe}
```

With:
- `data_directory`: location of data directory (containing both the initial conditions and output)
- `read_index`: the index of data files to use as initial conditions (default 0, change for restart)
- `imf_seed`: the random seed for the IMF realization
- `delay_time`: the delay time parameter in Myr (see paper)
- `sfe`: fraction of gas mass to be converted into gas

We have provided the initial conditions of all runs in the output directory. Once the `_run{a}` suffix, with `{a}` the run number, is removed, it can be read by the main script. The IMF seeds for all runs are:

1. 89168967
2. 22323457
3. 84357089
4. 83236243
5. 92455439
6. 39203741
