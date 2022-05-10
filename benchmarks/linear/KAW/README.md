This test runs a linear kinetic Alfven wave calculation in slab geometry.

To run the test, simply use 
```
[/path/to/]gx betahat10.0_kp0.01.in
```

To check the results, use
```
python plot-Pky-vs-time.py betahat10.0_kp0.01.nc betahat10.0_kp0.01_correct.nc
```

