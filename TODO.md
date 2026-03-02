- [ ] log_limiter_scalars should default to false
- [ ] all RawKernel helpers should check that arrays are contiguous and float64 or int32
- [x] has_shock should have a check that it is int32 type
- [ ] all RawKernel helpers should that aren’t called in parent functions should be in test_modified_regions
- [x] PAD kernel
- [x] conservative interpolation kernel
- [ ] troubles only saved as (1, nx, ny, nz). PAD_violations and NAD_violations now
        saved separately as (nvars, nx, ny, nz).
