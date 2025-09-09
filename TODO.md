changes:
- [x] PAD should be checked on primitive variables in both a priori and a posteriori cases
- [x] NAD should optionally be performed on primitives or conservatives

todo:
- [ ] SED for MUSCL-minmod
- [ ] array allocation is a mess
- [ ] Improve timer output and pause run_time during snapshots and minisnapshots
- [ ] at some point i created an Euler advection bug for flux recipes 2 and 3

Timer output:

Routine             #calls      Total time (s)  % of run time
-------             -----       ---------       -------------
run                    1            60                100
take_step              67           57                95
compute_dt             67           21                2
apply_bc               134          0.2               1
riemann_solver         134          14                25
Zhang Shu limiter       0            0                  0
MOOD_loop              3            12.2               10
    detect troubles    30           6.1                5
    revise fluxes      30           6.1                5
snapshot                67          10                 -
minisnapshot            134         10                 -
