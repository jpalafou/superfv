changes:
- [x] !major bug: no way to apply dirichlet boundary conditions with gauss-legendre fluxes. update advection tests once this is done
- [x] fix: GL asymmetry in slotted disk test
- [x] remove '_thing' naming convention unless the thing is actually hidden on purpose
- [x] !major bug: 'ZS = slope_limiter == "zhang-shu"' should be ZS = limiting_scheme == "zhang-shu"'
- [x] fix: 2DZS + RK4
- [x] fix: only working tests are pytest --ignore tests/test_SodShockTube1D.py
- [x] fix: CI pipeline

issues:
- [ ] write a test that compares periodic advection to time-dependent dirichlet boundary conditions
- [ ] test_ZS and test_MOOD
- [ ] redundant PAD checks in ZS_adaptive_dt and detect_troubles
- [ ] vestigial 'half-dt' option in MOOD
- [ ] move interpolation and integration functions to fv or some other modules

optimizations:
- [ ] preallocate padded arrays?
- [ ] timer should pause during snapshot
- [ ] look into cleaning up interpolation cache

cosmetics:
- [ ] plot_1d_slice and others should default to x=None, y=0.5, z=0.5
- [ ] clean up hydro/riemann solver modules
- [ ] make gauss-legendre a boolean option
- [ ] fix: progress bar for allow_overshoot=True
- [ ] switch from f"v{dim}" style to "v" + dim style for slight performance improvement
- [ ] expand README.md

very ambitious changes:
- [ ] multi-core support
- [ ] coverage
