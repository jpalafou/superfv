changes:

bugs:
- [ ] !major bug: 'ZS = slope_limiter == "zhang-shu"' should be ZS = limiting_scheme == "zhang-shu"'
- [ ] !major bug: no way to apply dirichlet boundary conditions with gauss-legendre fluxes. update advection tests once this is done
- [ ] fix: the rest of the tests
- [ ] clean up hydro/riemann solver modules
- [ ] fix: 2DZS + RK4

optimizations:
- [ ] preallocate padded arrays?
- [ ] timer should pause during snapshot
- [ ] look into cleaning up interpolation cache

cosmetics:
- [ ] make gauss-legendre a boolean option
- [ ] remove '_thing' naming convention unless the thing is actually hidden on purpose
- [ ] fix: progress bar for allow_overshoot=True
- [ ] switch from f"v{dim}" style to "v" + dim style for slight performance improvement
- [ ] expand README.md

very ambitious changes:
- [ ] multi-core support
- [ ] coverage
