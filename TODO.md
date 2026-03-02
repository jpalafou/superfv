- [ ] log_limiter_scalars should default to false
- [ ] all RawKernel helpers should check that arrays are contiguous and float64 or int32
- [ ] all RawKernel helpers should that aren’t called in parent functions should be in test_modified_regions
- [ ] troubles only saved as (1, nx, ny, nz). PAD_violations and NAD_violations now
        saved separately as (nvars, nx, ny, nz).
- [ ] kernel helpers should annotate arrays with cp.ndarray, not ArrayLike
- [ ] all the `hasattr(self.xp, "cuda")` inside solvers should just be `self.cupy`
