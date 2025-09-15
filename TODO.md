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

todo:
- [ ] rm 'inplace' from function names
- [ ] in `FiniteVolumeSolver.run`, switch `q_max` to `mode: Optional[int] = None` and allow -1 for MUSCL-Hancock
- [ ] add `ExplicitODESolver` args and documentation to `FiniteVolumeSolver.run`
- [ ] rm wtflux backend for now
- [ ] fix: hydro operators on nonsense ghost values

experiments:
- [ ] ZS4 comparison with Teyssier for the 1D experiments
