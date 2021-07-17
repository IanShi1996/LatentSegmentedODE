import numpy as np


def transform_sine(tp, freq, amp, phase=0, noise=0.):
    data = amp * np.sin(tp * freq + phase * np.pi)
    return data + noise


def generate_amps(n_cp, amp_range, min_delta):
    valid_amp = False
    amps = None
    while not valid_amp:
        amps = np.random.uniform(amp_range[0], amp_range[1], n_cp + 1)

        valid_amp = True

        for i in range(len(amps) - 1):
            if np.abs(np.abs(amps[i]) - np.abs(amps[i + 1])) < min_delta:
                valid_amp = False
        for i in range(len(amps)):
            if np.abs(amps[i]) < min_delta:
                valid_amp = False
    return amps


def generate_freqs(n_cp, freq_range, min_delta):
    valid_freqs = False
    freqs = None
    while not valid_freqs:
        freqs = np.random.uniform(freq_range[0], freq_range[1], n_cp + 1)

        valid_freqs = True

        for i in range(len(freqs) - 1):
            if np.abs(np.abs(freqs[i]) - np.abs(freqs[i + 1])) < min_delta:
                valid_freqs = False
    return freqs


def generate_timepoints(n_samp, end):
    return np.sort(np.random.uniform(0.01, end, n_samp))


def generate_resampled_set(n_traj, samp_range, max_end, amp_range, freq_range,
                           max_cp, min_amp_delta, min_freq_delta, noise):
    tp_bank = generate_timepoints(samp_range[1] * (max_cp + 2),
                                  max_end * (max_cp + 2))

    output = []

    for n_cp in range(max_cp + 1):
        for _ in range(n_traj):
            seg_data = np.array([])
            hyb_data = np.array([])
            seg_tps = np.array([])
            cps = []

            cp_counter = 0

            amps = generate_amps(n_cp, amp_range, min_amp_delta)
            freqs = generate_freqs(n_cp, freq_range, min_freq_delta)
            phases = np.random.uniform(0, 2, n_cp + 1)

            hyb_counter = 0

            for i in range(n_cp + 1):
                n_samp = np.random.randint(samp_range[0], samp_range[1])

                noise_samp = np.random.normal(0, noise)

                seg_tp = tp_bank[:n_samp]
                seg_d = transform_sine(seg_tp, freqs[i], amps[i], phases[i],
                                       noise_samp)

                seg_data = np.concatenate([seg_data, seg_d], axis=0)

                if len(seg_tps) > 0:
                    seg_tps = np.concatenate([seg_tps, seg_tp + seg_tps[-1]])
                else:
                    seg_tps = seg_tp
                cps.append(n_samp + cp_counter)

                last_tp = (np.abs(tp_bank - seg_tps[-1])).argmin()
                hyb_tp = tp_bank[hyb_counter:last_tp] - tp_bank[hyb_counter]
                hyb_d = transform_sine(hyb_tp, freqs[i], amps[i], phases[i],
                                       noise_samp)
                hyb_data = np.concatenate([hyb_data, hyb_d], axis=0)

                cp_counter += n_samp
                hyb_counter = last_tp

            hyb_tps = tp_bank[:hyb_counter]
            output.append((seg_data, seg_tps, cps[:-1], hyb_data, hyb_tps))
    return output


def generate_test_set(n_traj, samp_range, end_range, amp_range, freq_range,
                      max_cp, min_amp_delta, min_freq_delta, noise, extrap_len,
                      extrap_time):
    #
    output = []

    for n_cp in range(max_cp + 1):
        for _ in range(n_traj):
            data = np.array([])
            tps = np.array([])
            cps = []

            cp_counter = 0

            amps = generate_amps(n_cp, amp_range, min_amp_delta)
            freqs = generate_freqs(n_cp, freq_range, min_freq_delta)
            phases = np.random.uniform(0, 2, n_cp + 1)

            for i in range(n_cp + 1):
                n_samp = np.random.randint(samp_range[0], samp_range[1])
                end = np.random.uniform(end_range[0], end_range[1])

                if i == n_cp:
                    n_samp += extrap_len
                    end += extrap_time

                tp = np.sort(np.random.uniform(0, end, n_samp))

                d = transform_sine(tp, freqs[i], amps[i], phases[i], noise)

                data = np.concatenate([data, d], axis=0)
                if len(tps) > 0:
                    tps = np.concatenate([tps, tp + tps[-1]])
                else:
                    tps = tp
                cps.append(n_samp + cp_counter)
                cp_counter += n_samp

            output.append((data, tps, cps[:-1]))
    return output


