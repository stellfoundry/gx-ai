A module that, from a VMEC wout file, can calculate the geometry coefficients for both tokamaks and stellarators.

Additionally, this module can calculate the geometry coefficients when the user varies tprim(a\_N/L\_T), fprim(a\_N/L\_n), beta or global shear for a given equilibrium according to the formalism of Greene and Chance for tokamaks, and Hegna and Nakajima for stellarators. This could have significant implications for a transport solver such as Tr3D as we currently do not change geometric coefficients when we calculate the heat/particle flux jacobian with respect to fprim and tprim.

All the calculations are performed in Boozer coordinates as the geometric quantities are more intuitive, making the analysis relatively straightforward.

Dependencies: booz\_xform, netcdf4. Both these packages can be installed using pip.

Currently, I have only compared the coefficients with GS2 for tokamak equilibria. I have to add tests to avoid sign-related issues.

