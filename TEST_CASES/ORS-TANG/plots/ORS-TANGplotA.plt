set terminal x11 size 600,600
set size .8,.8
set origin .07,.15
set title "ENERGY PLOTS (original)"
set xlabel "Time/2 (s)"
set ylabel "Energy"
set xtics 0, .5, 4.000000
set mxtics 5
set ytics 0, 1
set mytics 5
set tics scale 3
set label "Nx=256   Ny=256   Nz=1   Boxsize=(2pi*10)^3\n\n\
nu=0.01   eta=0.01" at 0,-.9
set label "initial conditions:\n\
phi= -(cosx+cosy)\nA= .5(cos2x+2cosy)" at 1.5,-.9
plot [ ] [0:] "../outputs/TIMEoutputA.dat" using 1:2 title "total energy" with lines, \
"../outputs/TIMEoutputA.dat" using 1:3 title "kinetic energy" with lines, \
"../outputs/TIMEoutputA.dat" using 1:4 title "magnetic energy" with lines
set terminal postscript color solid
set output "ORS-TANGplotA.ps"
replot
pause -1 "press any key"
