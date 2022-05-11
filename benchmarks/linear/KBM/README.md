This test runs a linear kinetic ballooning mode instability calculations using a circular Miller geometry with Cyclone-base-case-like parameters.

To run the test, simply use 
```
[/path/to/]gx kbm_miller.in
```

To check the results, use
```
python plot-gams.py kbm_miller.nc kbm_miller_correct.nc 
```

