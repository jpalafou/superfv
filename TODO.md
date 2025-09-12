changes:
- [x] PAD should be checked on primitive variables in both a priori and a posteriori cases
- [x] NAD should optionally be performed on primitives or conservatives
- [x] at some point i created an Euler advection bug for flux recipes 2 and 3. take-away: SED is just sensitive for primitive variables
- [x] SED for MUSCL-minmod
- [x] Improve timer output and pause run_time during snapshots and minisnapshots
- [x] interpolate center in Zhang Shu limiter?
- [x] array allocation is a mess
- [x] `_integrate_for_fixed_number_of_steps` missing log every step option
- [x] do something about these divide-by-0 warnings (`warnings.filterwarnings("ignore", category=RuntimeWarning)`)

todo:
- [ ] MUSCL-Hancock fails to converge to second-order even when MUSCL_limiter=None. The is remedied by using `flux_recipe=1` and `.ssprk2()`, so the issue is probably with my predictor-corrector implementation.
- [ ] write solutions to disk
- [ ] automatically set inactive dim BCs to "none"
- [ ] rm 'inplace' from function names
