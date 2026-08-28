stop passing `dim` to hydro kernels, etc. make the first arg normal by default and the following two are transverse

improve HydroSolver doc string; list out what each literal means in a few words

the raising of exceptions in configs.py is stressful to me. i feel exceptions should be raised in the source code where they are relevant.
