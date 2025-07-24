changes:
- [x] !major bug: no way to apply dirichlet boundary conditions with gauss-legendre fluxes. update advection tests once this is done
- [x] fix: GL asymmetry in slotted disk test
- [x] remove '_thing' naming convention unless the thing is actually hidden on purpose
- [x] !major bug: 'ZS = slope_limiter == "zhang-shu"' should be ZS = limiting_scheme == "zhang-shu"'
- [x] fix: 2DZS + RK4
- [x] fix: only working tests are pytest --ignore tests/test_SodShockTube1D.py
- [x] fix: CI pipeline
- [x] for large p and N, the first step is actually the fastest
- [x] move interpolation and integration functions to fv or some other modules
- [x] look into cleaning up interpolation cache
- [x] adaptive timestep should revise "unew"
- [x] preallocate padded arrays?
- [x] plot_1d_slice and others should default to x=None, y=0.5, z=0.5
- [x] use PAD violations function for adaptive dt
- [x] ZhangShuConfig

issues:
- [ ] PAD asymmetry for 2D square
- [ ] buffer array size is arbitrary in 3D
- [ ] test_ZS and test_MOOD
- [ ] redundant PAD checks in ZS_adaptive_dt and detect_troubles
- [ ] vestigial 'half-dt' option in MOOD

optimizations:
- [ ] timer should pause during snapshot. maybe rephrase as clean up timer calls

cosmetics:
- [ ] sort and clean `ExplicitODESolver` and `FiniteVolumeSolver` methods
- [ ] move `self.timer.reset("current_step")` to called_at_beginning_of_step
- [ ] buffer fv_interpolate should be optional for the single-sweep case
- [ ] clean up hydro/riemann solver modules
- [ ] make gauss-legendre a boolean option
- [ ] fix: progress bar for allow_overshoot=True
- [ ] switch from f"v{dim}" style to "v" + dim style for slight performance improvement
- [ ] expand README.md

very ambitious changes:
- [ ] multi-core support
- [ ] coverage
