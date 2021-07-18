One full trajectory, which is broken into SDFs.
Extrapolation regions too.

Simplifying assumption: timepoints are aligned


'amp_range': (args.amp_range_min, args.amp_range_max),
'freq_range': (args.freq_range_min, args.freq_range_max),
'samp_range': (args.samp_range_min, args.samp_range_max),
'end_range': (args.end_range_min, args.end_range_max),
'min_amp_delta': args.min_amp_delta,
'min_freq_delta': args.min_freq_delta,
'noise': args.noise,
'extrap_len': args.extrap_len,
'extrap_time': args.extrap_time,
'max_cp': args.max_cp,
'n_traj_train': args.n_traj_train,
'n_traj_val': args.n_traj_val,
'n_traj_test': args.n_traj_test, (Per max_cp!)