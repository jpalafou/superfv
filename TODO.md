stop passing `dim` to hydro kernels, etc. make the first arg normal by default and the following two are transverse

the importing of enums is stressful to me. should they all go in a special enum module? is it better to just use literals and enforce them in configs.py?

the raising of exceptions in configs.py is stressful to me. i feel exceptions should be raised in the source code where they are relevant.
