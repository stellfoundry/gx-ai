this test runs a linear ITG instability calculation using a Cyclone-base-case-like circular Miller geometry. this test uses adiabatic electrons.

to run the test, simply use 
```
[/path/to/]gx itg_miller_adiabatic_electrons.in
```

to check the results, use
```
python plot-gams.py itg_miller_adiabatic_electrons.nc itg_miller_adiabatic_electrons_correct.nc 
```

