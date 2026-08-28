stop passing `dim` to hydro kernels, etc. make the first arg normal by default and the following two are transverse

improve HydroSolver doc string; list out what each literal means in a few words

the raising of exceptions in configs.py is stressful to me. i feel exceptions should be raised in the source code where they are relevant.

`# If limiting conservatives, then all PAD bounds must be in primitives` this warning is confusing

Remove resassignmetns in configs.py , they should be errors

Maybe shock detection should have its own PAD params
