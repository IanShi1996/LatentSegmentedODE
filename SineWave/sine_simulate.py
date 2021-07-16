import numpy as np
np.random.seed(2547)


def transform_sine(tp, freq, amp, phase=0, n_samp=1, noise=0.):
    """Evaluate sine wave at given time points.

    Args:
        tp (np.ndarray): Time points at which to evaluate sine wave.
        freq (float): Frequency of sine wave.
        amp (float): Amplitude of sine wave.
        phase (float, optional): Phase of sine wave in radians.
        n_samp (int, optional): Number of sine waves to generate.
        noise (float, optional): Std of normal noise to add to sine wave.

    Returns:
        np.ndarray: Sine wave evaluated at given time points.
    """
    data = amp * np.sin(tp * freq + phase * np.pi)

    data = np.repeat(np.expand_dims(data, 0), n_samp, 0)
    noise = np.random.normal(0, noise, (n_samp, data.shape[1]))

    return data + noise


def generate_sine_linear(n_samp, freq, amp, phase, end, noise_std=0.):
    """Generate sine wave with linear time steps.

    Args:
        n_samp (int): Number of samples in trajectory.
        freq (float): Frequency of sine wave.
        amp (float): Amplitude of sine wave.
        phase (float): Phase of sine wave in radians.
        end (float): Last time point of trajectory.
        noise_std (float, optional): Standard deviation of noise added to sine.
    Returns:
        np.ndarray: Sine wave observation times and data.
    """
    tp = np.linspace(0, end, n_samp)
    data = transform_sine(tp, freq, amp, phase, 1, noise_std)

    return tp, data


def generate_sine_uniform(n_samp, freq, amp, phase, end, noise_std=0.):
    """Generate sine wave with uniformly sampled time steps.
    Args:
        n_samp (int): Number of samples in trajectory.
        freq (float): Frequency of sine wave.
        amp (float): Amplitude of sine wave.
        phase (float): Phase of sine wave in radians.
        end (float): Last time point of trajectory.
        noise_std (float, optional): Standard deviation of noise added to sine.
    Returns:
        np.ndarray: Sine wave observation times and data.
    """
    tp = np.sort(np.random.uniform(0, end, n_samp))
    data = transform_sine(tp, freq, amp, phase, 1, noise_std)

    return tp, data


def generate_disjoint_tps(n_samp, start, end):
    """Generate three sets of disjoint time points.

    Generates three sets of disjoint time points. Time points are drawn from an
    uniform distribution. Intended for usage as training/validation/test set
    observation times for model training.

    Args:
        n_samp (int, int, int): Tuple specifying number of samples. Entries
            specify number of train/val/test samples respectively.
        start (float): First time point.
        end (float): Last time point.

    Returns:
        (np.ndarray, np.ndarray, np.ndarray): Time points in train/val/test set.
    """
    n_total = sum(n_samp)
    tps = np.random.uniform(start, end, n_total)

    train_time = np.sort(tps[:n_samp[0]])
    val_time = np.sort(tps[n_samp[0]:n_samp[0]+n_samp[1]])
    test_time = np.sort(tps[n_samp[0]+n_samp[1]:])

    return train_time, val_time, test_time


class SineSetGenerator:
    """Generates three sets of sine waves according to parameterization.

    Intended for use as train/val/test datasets for Latent Node models.
    Sine waves are generated using amplitude and frequency sampled from a
    uniform distribution bounded by input parameters. Each trajectory will
    share time points within a specific set. Time points can be made evenly
    spaced, or sampled under a uniform distribution. Differences in phase
    (and thus initial condition) can be included.

    Additional features such as changing data generation scheme to favour
    specific trajectories, or adding more sine wave parameters such as phase
    could be added in the future.

    Attributes:
        n_traj (int, int, int): Number of trajectories in train/val/test set.
        n_samp (int, int, int): Number of samples per trajectory in
            train/val/test set.
        amp (float, float): Minimum and maximum amplitudes.
        freq (float, float): Minimum and maximum frequencies.
        phase (boolean): Whether phase variation should be included.
        start (float): Initial time point for trajectories.
        end (float): Final time point for trajectories.
        noise (float): Standard deviation of simulated noise.
        tp_generation (string, string, string): Tuple representing time point
            generation strategy. Can be either L for linear or U for uniform.
    """

    def __init__(self, params):
        """Initialize SineSetGenerator.

        Expects a dictionary of many parameters. Currently does not check
        validity of these arguments, nor is backwards compatible.

        Args:
            params (dict): Dictionary of sine wave parameters.
        """
        self.n_traj = params['n_traj']
        self.n_samp = params['n_samp']

        self.amp = params['amp']
        self.freq = params['freq']
        self.phase = params['phase']
        self.start = params['start']
        self.end = params['end']
        self.noise = params['noise']

        self.tp_generation = params['tp_generation']

        self.train_time, self.val_time, self.test_time = self.gen_tps()
        self.train_data, self.val_data, self.test_data = self.gen_data()

    def gen_tps(self):
        """Generate observation times.

        Time points are shared across a set of data. Time points are
        sampled according to a uniform distribution by default, but can
        be switched to a linspace via the tp_generation flag.

        This default behavior should be modified when other time point
        generation schemes are required.

        Returns:
            (tuple): Time points of dataset.
        """
        tp = list(generate_disjoint_tps(self.n_samp, self.start, self.end))
        for i in range(len(self.tp_generation)):
            if self.tp_generation[i] == 'L':
                tp[i] = np.linspace(self.start, self.end, self.n_traj[i])

        return tuple(tp)

    def gen_data(self):
        """Generate data by uniformly sampling frequency and amplitude.

        Returns:
            (np.ndarray, np.ndarray, np.ndarray): Train, val, and test data.
        """
        train_data = []
        val_data = []
        test_data = []

        total_traj = sum(self.n_traj)

        freqs = np.random.uniform(self.freq[0], self.freq[1], total_traj)
        amps = np.random.uniform(self.amp[0], self.amp[1], total_traj)

        if self.phase:
            phase = np.random.uniform(0, 2, total_traj)
        else:
            phase = [0] * total_traj

        for i in range(self.n_traj[0]):
            train_d = transform_sine(self.train_time, freqs[i], amps[i],
                                     phase[i], noise=self.noise)
            train_data.append(train_d)

        for i in range(self.n_traj[0], self.n_traj[0]+self.n_traj[1]):
            val_d = transform_sine(self.val_time, freqs[i], amps[i],
                                   phase[i], noise=self.noise)
            val_data.append(val_d)

        for i in range(self.n_traj[0]+self.n_traj[1], total_traj):
            test_d = transform_sine(self.test_time, freqs[i], amps[i],
                                    phase[i], noise=self.noise)
            test_data.append(test_d)

        train_data = np.stack(train_data, 0)
        train_data = np.concatenate(train_data, 0)

        val_data = np.stack(val_data, 0)
        val_data = np.concatenate(val_data, 0)

        test_data = np.stack(test_data, 0)
        test_data = np.concatenate(test_data, 0)

        return train_data, val_data, test_data

    def get_train_set(self):
        """Retrieve training time points and data.

        Returns:
            (np.ndarray, np.ndarray): Training time points and data.
        """
        return self.train_time, self.train_data

    def get_test_set(self):
        """Retrieve test time points and data.

        Returns:
            (np.ndarray, np.ndarray): Test time points and data.
        """
        return self.test_time, self.test_data

    def get_val_set(self):
        """Retrieve validation time points and data.

        Returns:
            (np.ndarray, np.ndarray): Validation time points and data.
        """
        return self.val_time, self.val_data


def generate_classwise_sine(tp, n_traj, amps, freqs, phases, noise):
    """Generate sine trajectories which are clustered by class.

    Generated trajectories are clustered around specific frequency, amplitude,
    and phase combinations. These combinations are sorted in increasing order
    of amplitude, frequency, then phase.

    Args:
        tp (np.ndarray): Time points of all sine trajectories.
        n_traj (int): Number of trajectories per class.
        amps (list of float): Class amplitudes.
        freqs (list of float): Class frequencies.
        phases (list of float): Class phases.
        noise (float): Standard deviation of simulated noise.

    Returns:
        list of nd.array: List of all trajectories grouped by class.
    """
    data = []

    for amp in amps:
        for freq in freqs:
            for phase in phases:
                d = transform_sine(tp, freq, amp, phase, n_traj, noise)
                data.append(d)

    return data


def generate_piecewise_sine(n_samp, n_cp, amp_range, freq_range, phase_flag,
                            cp_min, amp_min, noise, end=None, manual_tp=None):
    """Generate a trajectory composed of piecewise sine waves.

    Each segment is generated using a sine wave with parameters sampled from
    a uniform distribution specified by method arguments. Trajectories
    must satisfy several conditions:

    1. Change points must be further than a minimum distance from the ends of
       the trajectory as well as each other.
    2. Amplitudes must be at least a certain magnitude apart. This prevents
       segments being reflections of each other, essentially meaning the
       underlying distribution does not change.
    3. Amplitudes must not be near zero.

    Time points must either be passed in, or can be generated by specifying the
    last time point of observation, allowing for uniform sampling between 0 and
    this time point to generate time points.

    Args:
        n_samp (int): Number of samples per trajectory.
        n_cp (int): Number of change points.
        amp_range (float, float): Minimum and maximum possible amplitude.
        freq_range (float, float): Minimum and maximum possible frequency.
        phase_flag (boolean): Whether phase modulation should be included.
        cp_min (float): Minimum distance between change points.
        amp_min (float): Minimum absolute difference between segment amplitude.
        noise (float): Standard deviation of injected random noise.
        end (float, optional): Last observed time point.
        manual_tp (np.ndarray, optional): Time points to evaluate sine wave.

    Raises:
        ValueError: Raised when both end and manual_tp are set to None.

    Returns:
        (np.ndarray, np.ndarray, np.ndarray): Array of trajectory time points,
            data points, and ground truth change point locations.
    """
    if end is None and manual_tp is None:
        raise ValueError("Either end or manual_tp must be specified.")

    amps, cps = None, None

    valid_cp = False
    while not valid_cp:
        cps = np.sort(np.random.choice(np.arange(cp_min, n_samp-cp_min), n_cp))
        cps = np.concatenate(([0], cps, [n_samp]), 0)

        valid_cp = True

        for i in range(len(cps) - 1):
            identical = cps[i] == cps[i+1]
            below_min = cps[i+1] - cps[i] < cp_min

            if identical or below_min:
                valid_cp = False

    valid_amp = False
    while not valid_amp:
        amps = np.random.uniform(amp_range[0], amp_range[1], n_cp+1)

        valid_amp = True

        for i in range(len(amps)-1):
            if np.abs(np.abs(amps[i]) - np.abs(amps[i+1])) < amp_min:
                valid_amp = False
        for i in range(len(amps)):
            if np.abs(amps[i]) < amp_min:
                valid_amp = False

    freqs = np.random.uniform(freq_range[0], freq_range[1], n_cp+1)

    traj_data = np.array([])
    traj_time = np.array([])

    for i in range(n_cp+1):
        amp = amps[i]
        freq = freqs[i]
        length = int(cps[i+1] - cps[i])

        if manual_tp is not None:
            tp = manual_tp[:length]
        else:
            tp = np.sort(np.random.uniform(0, end, n_samp))[:length]

        if phase_flag:
            phase = np.random.uniform(0, 2)
        else:
            phase = 0

        d = transform_sine(tp, freq, amp, phase, noise=noise).flatten()

        # Starts time from last observed time
        if i > 0:
            tp = tp + traj_time[-1]

        traj_time = np.concatenate((traj_time, tp))
        traj_data = np.concatenate((traj_data, d))

    cps = cps[1:len(cps)-1]

    return traj_time, traj_data, cps


def generate_test_set(generator_params, max_cp, n_traj, seg_samp, seg_len):
    """ Generate piecewise 1D sine trajectories as test data.

    Certain relaxations are typically made. These include:
      - Setting freq and amp ranges to match ranges in the model training set.
      - Setting a minimum change in amplitude between segments.
      - Setting a minimum absolute amplitude (must not be near zero).
      - Setting a minimum distance between change points.
      - Setting a minimum number of samples per segment.
      - Increasing minimum length per segment with number of change points.

    Opt to not inject noise at this stage. This allows for easier downstream
    testing of model sensitivity to noise.

    Args:
        generator_params (dict): Data generation parameters. See
            generate_piecewise_sine for required keys.
        max_cp (int): Maximum number of change points in data set.
        n_traj (int): Number of trajectories for each change point sub set.
        seg_samp (int): Samples per segment.
        seg_len (float): Length of each segment.

    Returns:
        (list of (data, tps, cps)): Test dataset. Each tuple contains the data,
            time points, and change points associated with each trajectory.
    """
    data = []

    for cp in range(max_cp + 1):
        for _ in range(n_traj):
            length = max(seg_len * 0.5, (cp+1) * seg_len * 0.5)
            n_samps = max(seg_samp, (cp+1) * int(seg_samp * 0.75))

            traj = generate_piecewise_sine(n_samps, cp,
                                           **generator_params,
                                           end=length)
            out = (traj[1].reshape((1, -1, 1)), traj[0], traj[2])
            data.append(out)

    return data
