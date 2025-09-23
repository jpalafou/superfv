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
- [x] automatically set inactive dim BCs to "none"
- [x] write solutions to disk
- [x] ZS4 comparison with Teyssier for the 1D experiments
- [x] timing file should print with \n at the end to make catting more pleasant
- [x] rm wtflux backend for now
- [x] rm: log_every_step argument
- [x] add `ExplicitODESolver` args and documentation to `FiniteVolumeSolver.run`
- [x] rm 'inplace' from function names

todo:
- [ ] fix: hydro operators on nonsense ghost values
- [ ] ZS + flux_recipe=2 doesn't converge
