This test runs a linear ITG instability calculation using a circular Miller geometry with Cyclone-base-case-like parameters.
This test uses a Boltzmann adiabatic electron response.

To run the test, simply use 
```
[/path/to/]gx itg_miller_adiabatic_electrons.in
```

To check the results, use
```
python plot-gams.py itg_miller_adiabatic_electrons.nc itg_miller_adiabatic_electrons_correct.nc 
```

