import reservoirpy as rpy
import numpy as np
import pandas as pd
from itertools import cycle
from tqdm import tqdm
from datetime import datetime
import cftime
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

# Global Matplotlib style settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica'] + plt.rcParams['font.sans-serif']
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['legend.frameon'] = False
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.minor.width'] = 1.0
plt.rcParams['ytick.minor.width'] = 1.0
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['font.size'] = 14
plt.rcParams['legend.fontsize'] = 12


# =============================================================================
# Section 1 — Model Construction
# =============================================================================

def Create_New_ESN(units=4000, lr=1, sr=0.95, rc_connectivity=0.2,
                   noise_rc=0.02, noise_in=0.0, output_dim=1,
                   input_scaling=1, W=None, ridge=5e-6,
                   input_connectivity=0.1, use_raw_input=False, seed=None):
    """Create a standard Echo State Network (ESN) with Ridge readout.

    Parameters
    ----------
    units : int
        Number of reservoir neurons.
    lr : float
        Leak rate (0, 1].
    sr : float
        Spectral radius of the recurrent weight matrix.
    rc_connectivity : float
        Density of the recurrent weight matrix (ignored when W is given).
    noise_rc : float
        Noise injected into reservoir states at each step.
    noise_in : float
        Noise added to the input signal.
    output_dim : int
        Dimensionality of the readout output.
    input_scaling : float
        Global scaling factor applied to the input weight matrix.
    W : np.ndarray or None
        Pre-built recurrent weight matrix. If None, a random sparse matrix
        is generated using rc_connectivity.
    ridge : float
        L2 regularisation coefficient for the Ridge readout.
    input_connectivity : float
        Density of the input weight matrix.
    use_raw_input : bool
        Whether to concatenate the raw input to reservoir states before the
        readout. Stored as an attribute for downstream use.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    esn : reservoirpy model
        Assembled ESN (Reservoir >> Ridge).
    """
    if W is None:
        res0 = rpy.nodes.Reservoir(
            units=units, lr=lr, sr=sr,
            rc_connectivity=rc_connectivity,
            input_connectivity=input_connectivity,
            input_scaling=input_scaling,
            noise_rc=noise_rc, noise_in=noise_in, seed=seed)
    else:
        res0 = rpy.nodes.Reservoir(
            units=units, lr=lr, sr=sr,
            noise_rc=noise_rc, noise_in=noise_in, W=W,
            input_connectivity=input_connectivity,
            input_scaling=input_scaling, seed=seed)
    readout0 = rpy.nodes.Ridge(output_dim=output_dim, ridge=ridge)
    esn0 = res0 >> readout0
    esn0.use_raw_input = use_raw_input
    return esn0


def Create_Online_ESN(units=4000, lr=1, sr=0.95, rc_connectivity=0.2,
                      noise_rc=0.02, noise_in=0.02, output_dim=1,
                      W=None, alpha=5e-6, input_connectivity=0.1,
                      input_scaling=1, use_raw_input=False, seed=None):
    """Create an online-learning ESN with a FORCE (RLS) readout.

    Parameters
    ----------
    units : int
        Number of reservoir neurons.
    lr : float
        Leak rate (0, 1].
    sr : float
        Spectral radius of the recurrent weight matrix.
    rc_connectivity : float
        Density of the recurrent weight matrix (ignored when W is given).
    noise_rc : float
        Noise injected into reservoir states at each step.
    noise_in : float
        Noise added to the input signal.
    output_dim : int
        Dimensionality of the readout output.
    W : np.ndarray or None
        Pre-built recurrent weight matrix.
    alpha : float
        Regularisation parameter for the FORCE/RLS readout.
    input_connectivity : float
        Density of the input weight matrix.
    input_scaling : float
        Global scaling factor applied to the input weight matrix.
    use_raw_input : bool
        Whether to concatenate the raw input to reservoir states.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    esn : reservoirpy model
        Assembled online ESN (Reservoir >> FORCE).
    """
    if W is None:
        res0 = rpy.nodes.Reservoir(
            units=units, lr=lr, sr=sr,
            rc_connectivity=rc_connectivity,
            input_connectivity=input_connectivity,
            input_scaling=input_scaling,
            noise_rc=noise_rc, noise_in=noise_in, seed=seed)
    else:
        res0 = rpy.nodes.Reservoir(
            units=units, lr=lr, sr=sr,
            noise_rc=noise_rc, noise_in=noise_in, W=W,
            input_connectivity=input_connectivity,
            input_scaling=input_scaling, seed=seed)
    force = rpy.nodes.FORCE(output_dim=output_dim, alpha=alpha)
    esn0 = res0 >> force
    esn0.use_raw_input = use_raw_input
    return esn0


def Create_New_IPESN(units=4000, lr=1, sr=0.95, rc_connectivity=0.2,
                     noise_rc=0.0, noise_in=0.0, output_dim=1, W=None,
                     ridge=5e-6, use_raw_input=False, input_connectivity=0.1,
                     input_scaling=1, seed=None):
    """Create an Intrinsic Plasticity ESN (IP-ESN) with Ridge readout.

    The reservoir uses an IPReservoir node which adapts its activation
    function parameters during the warm-up phase to match a target output
    distribution, improving information transmission.

    Parameters
    ----------
    units : int
        Number of reservoir neurons.
    lr : float
        Leak rate (0, 1].
    sr : float
        Spectral radius of the recurrent weight matrix.
    rc_connectivity : float
        Density of the recurrent weight matrix (ignored when W is given).
    noise_rc : float
        Noise injected into reservoir states at each step.
    noise_in : float
        Noise added to the input signal.
    output_dim : int
        Dimensionality of the readout output.
    W : np.ndarray or None
        Pre-built recurrent weight matrix.
    ridge : float
        L2 regularisation coefficient for the Ridge readout.
    use_raw_input : bool
        Whether to concatenate the raw input to reservoir states.
    input_connectivity : float
        Density of the input weight matrix.
    input_scaling : float
        Global scaling factor applied to the input weight matrix.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    esn : reservoirpy model
        Assembled IP-ESN (IPReservoir >> Ridge).
    """
    if W is None:
        res0 = rpy.nodes.IPReservoir(
            units=units, lr=lr, sr=sr,
            rc_connectivity=rc_connectivity,
            input_connectivity=input_connectivity,
            noise_rc=noise_rc, input_scaling=input_scaling,
            noise_in=noise_in, seed=seed)
    else:
        res0 = rpy.nodes.IPReservoir(
            units=units, lr=lr, sr=sr,
            noise_rc=noise_rc, noise_in=noise_in, W=W,
            input_connectivity=input_connectivity,
            input_scaling=input_scaling, seed=seed)
    readout0 = rpy.nodes.Ridge(output_dim=output_dim, ridge=ridge)
    esn0 = res0 >> readout0
    esn0.use_raw_input = use_raw_input
    return esn0


def Create_Deep_ESN(units: list, lr: list, sr: list,
                    rc_connectivity: list = None, input_scaling: list = None,
                    W: list = None, Win: list = None,
                    noise_rc: list = None, noise_in: list = None,
                    input_connectivity: list = None,
                    output_dim=1, ridge=5e-6, seed=None,
                    deep_ip_list: list = None):
    """Create a multi-layer (deep) ESN with a shared Ridge readout.

    Each layer receives input from the previous layer. The readout collects
    states from all layers. Individual layers can be standard Reservoir or
    IPReservoir nodes, controlled per layer via deep_ip_list.

    Parameters
    ----------
    units : list of int
        Number of neurons in each reservoir layer.
    lr : list of float
        Leak rate for each layer.
    sr : list of float
        Spectral radius for each layer.
    rc_connectivity : list of float or None
        Recurrent connectivity for each layer. Defaults to 0.1 per layer.
        Ignored for layers where W is provided.
    input_scaling : list of float or None
        Input scaling for each layer.
    W : list of np.ndarray or None
        Pre-built recurrent weight matrices (per layer). None entries
        trigger random generation using rc_connectivity.
    Win : list of np.ndarray or None
        Pre-built input weight matrices (per layer). None entries trigger
        random generation using input_connectivity.
    noise_rc : list of float or None
        Reservoir noise for each layer. Defaults to 0.0.
    noise_in : list of float or None
        Input noise for each layer. Defaults to 0.0.
    input_connectivity : list of float or None
        Input connectivity for each layer. Defaults to 0.1 for the first
        layer and 1.0 for subsequent layers.
    output_dim : int
        Dimensionality of the shared readout output.
    ridge : float
        L2 regularisation coefficient for the Ridge readout.
    seed : int or None
        Random seed for reproducibility.
    deep_ip_list : list of bool or None
        Whether to use IPReservoir for each layer. Defaults to False for
        all layers.

    Returns
    -------
    esn : reservoirpy model
        Assembled deep ESN model.

    Raises
    ------
    ValueError
        If the lengths of units, lr, and sr do not match, or if optional
        list parameters have incompatible lengths.
    """
    if not (len(units) == len(lr) == len(sr)):
        raise ValueError("'units', 'lr', and 'sr' must have the same length.")
    if rc_connectivity is not None and len(rc_connectivity) != len(units):
        raise ValueError("'rc_connectivity' must have the same length as 'units'.")
    if W is not None and len(W) != len(units):
        raise ValueError("'W' must have the same length as 'units'.")
    if Win is not None and len(Win) != len(units):
        raise ValueError("'Win' must have the same length as 'units'.")
    if input_connectivity is not None and len(input_connectivity) != len(units):
        raise ValueError("'input_connectivity' must have the same length as 'units'.")
    if deep_ip_list is not None and len(deep_ip_list) != len(units):
        raise ValueError("'deep_ip_list' must have the same length as 'units'.")

    # Set defaults
    if noise_rc is None:
        noise_rc = [0.0] * len(units)
    if noise_in is None:
        noise_in = [0.0] * len(units)
    if input_connectivity is None:
        input_connectivity = [0.1] + [1.0] * (len(units) - 1)
    if rc_connectivity is None:
        rc_connectivity = [0.1] * len(units)
    if deep_ip_list is None:
        deep_ip_list = [False] * len(units)

    input_node = rpy.nodes.Input()
    readout = rpy.nodes.Ridge(output_dim=output_dim, ridge=ridge)

    path1 = []  # input-to-reservoir paths
    path2 = []  # reservoir-to-readout paths

    for i in range(len(units)):
        print(f"Layer {i+1}: units={units[i]}, sr={sr[i]}, lr={lr[i]}, ip_reservoir={deep_ip_list[i]}")

        reservoir_params = {
            "units": units[i], "sr": sr[i], "lr": lr[i],
            "noise_rc": noise_rc[i], "noise_in": noise_in[i],
            "seed": seed, "input_scaling": input_scaling[i],
        }

        if W is None or W[i] is None:
            reservoir_params["rc_connectivity"] = rc_connectivity[i]
        else:
            reservoir_params["W"] = W[i]

        if Win is None or Win[i] is None:
            reservoir_params["input_connectivity"] = input_connectivity[i]
        else:
            reservoir_params["Win"] = Win[i]

        if deep_ip_list[i]:
            reservoir = rpy.nodes.IPReservoir(**reservoir_params)
        else:
            reservoir = rpy.nodes.Reservoir(**reservoir_params)

        path1.append(input_node >> reservoir)
        path2.append(reservoir >> readout)
        input_node = reservoir  # chain layers sequentially

    esn = rpy.merge(*path1, *path2)
    return esn


def get_hyperparameters(hypers):
    """Extract and normalise a hyperparameter dictionary for ESN construction.

    Provides default values for optional keys and distinguishes between
    offline (Ridge), online (FORCE), IP, and deep ESN configurations.

    Parameters
    ----------
    hypers : dict
        Raw hyperparameter dictionary. Required keys: 'units', 'lr', 'sr',
        'rc_connectivity', 'noise_rc', 'noise_in', 'seed',
        'input_connectivity', 'input_scaling'.
        Optional keys (with defaults): 'output_dim' (None), 'use_raw_input'
        (False), 'online' (False), 'alpha' (1e-6), 'ip_reservoir' (False),
        'ridge' (None), 'deep' (False), 'deep_ip_list' (None).

    Returns
    -------
    params : dict
        Normalised parameter dictionary suitable for passing to
        get_esn_from_hypers.
    """
    params = {
        "units": hypers['units'],
        "lr": hypers['lr'],
        "sr": hypers['sr'],
        "rc_connectivity": hypers['rc_connectivity'],
        "noise_rc": hypers['noise_rc'],
        "noise_in": hypers['noise_in'],
        "seed": hypers['seed'],
        "input_connectivity": hypers['input_connectivity'],
        "input_scaling": hypers['input_scaling'],
        "output_dim": hypers.get('output_dim', None),
        "use_raw_input": hypers.get('use_raw_input', False),
        "online": hypers.get('online', False),
        "alpha": hypers.get('alpha', 1e-6) if hypers.get('online', False) else None,
        "ip_reservoir": hypers.get('ip_reservoir', False),
        "ridge": hypers.get('ridge', None) if not hypers.get('online', False) else None,
        "deep": hypers.get('deep', False),
        "deep_ip_list": hypers.get('deep_ip_list', None),
    }
    return params


def get_esn_from_hypers(hypers):
    """Instantiate the appropriate ESN variant from a hyperparameter dictionary.

    Dispatches to Create_Deep_ESN, Create_Online_ESN, Create_New_IPESN, or
    Create_New_ESN depending on the flags in hypers.

    Parameters
    ----------
    hypers : dict
        Hyperparameter dictionary as accepted by get_hyperparameters.
        Key flags controlling dispatch:
          - 'deep' (bool): build a multi-layer deep ESN.
          - 'online' (bool): use FORCE/RLS online learning readout.
          - 'ip_reservoir' (bool): use IPReservoir with Ridge readout.
          - (default): standard Reservoir with Ridge readout.

    Returns
    -------
    esn : reservoirpy model
        Constructed ESN model ready for training.
    """
    params = get_hyperparameters(hypers)

    if params["deep"]:
        esn = Create_Deep_ESN(
            units=params["units"], lr=params["lr"], sr=params["sr"],
            rc_connectivity=params["rc_connectivity"],
            noise_rc=params["noise_rc"], noise_in=params["noise_in"],
            ridge=params["ridge"],
            input_connectivity=params["input_connectivity"],
            input_scaling=params["input_scaling"],
            output_dim=params["output_dim"],
            seed=params["seed"], deep_ip_list=params["deep_ip_list"],
        )
    elif params["online"]:
        esn = Create_Online_ESN(
            units=params["units"], lr=params["lr"], sr=params["sr"],
            rc_connectivity=params["rc_connectivity"],
            noise_rc=params["noise_rc"], noise_in=params["noise_in"],
            input_connectivity=params["input_connectivity"],
            input_scaling=params["input_scaling"],
            output_dim=params["output_dim"], alpha=params["alpha"],
            seed=params["seed"], use_raw_input=params["use_raw_input"],
        )
    elif params["ip_reservoir"]:
        esn = Create_New_IPESN(
            units=params["units"], lr=params["lr"], sr=params["sr"],
            rc_connectivity=params["rc_connectivity"],
            noise_rc=params["noise_rc"], noise_in=params["noise_in"],
            input_connectivity=params["input_connectivity"],
            input_scaling=params["input_scaling"],
            output_dim=params["output_dim"], ridge=params["ridge"],
            seed=params["seed"], use_raw_input=params["use_raw_input"],
        )
    else:
        esn = Create_New_ESN(
            units=params["units"], lr=params["lr"], sr=params["sr"],
            rc_connectivity=params["rc_connectivity"],
            noise_rc=params["noise_rc"], noise_in=params["noise_in"],
            input_connectivity=params["input_connectivity"],
            input_scaling=params["input_scaling"],
            output_dim=params["output_dim"], ridge=params["ridge"],
            seed=params["seed"], use_raw_input=params["use_raw_input"],
        )
    return esn


# =============================================================================
# Section 2 — Training & Prediction
# =============================================================================

def pack_TS_anualTP(TS, omega=2*np.pi/12, order=3, bias=0):
    """Augment a time series with sinusoidal temporal-period (TP) features.

    Appends paired sin/cos columns at each harmonic up to the specified order,
    enabling the ESN to distinguish calendar months.

    Parameters
    ----------
    TS : np.ndarray, shape (T, n_vars)
        Input multivariate time series.
    omega : float
        Base angular frequency. Default 2π/12 (monthly annual cycle).
    order : int
        Number of harmonics to include (1 through order). Default 3.
    bias : float
        Constant offset added to each sinusoidal feature. Default 0.

    Returns
    -------
    TS_TP : np.ndarray, shape (T, n_vars + 2*order)
        Original time series concatenated with 2*order temporal features.
    """
    t = np.linspace(0, TS.shape[0] - 1, TS.shape[0], dtype=int)
    for i in range(1, order + 1):
        sint = (bias + np.sin(i * omega * t)).reshape(-1, 1)
        cost = (bias + np.cos(i * omega * t)).reshape(-1, 1)
        if i == 1:
            TP = np.hstack((sint, cost))
        else:
            TP = np.hstack((TP, sint, cost))
    TS_TP = np.hstack((TS, TP))
    return TS_TP


def get_RCTP(TS, steps=22, omega=2*np.pi/12, order=3, bias=0):
    """Generate a sinusoidal temporal-period (TP) feature matrix.

    Produces a TP array that covers the training period plus an additional
    `steps` months, so that temporal features can be updated during
    multi-step iterative forecasting.

    Parameters
    ----------
    TS : np.ndarray, shape (T, n_vars)
        Reference time series (only its length T is used).
    steps : int
        Number of additional forecast steps to extend the TP array.
    omega : float
        Base angular frequency. Default 2π/12.
    order : int
        Number of harmonics. Default 3.
    bias : float
        Constant offset added to each sinusoidal feature. Default 0.

    Returns
    -------
    TP : np.ndarray, shape (T + steps, 2*order)
        Temporal-period feature matrix spanning [0, T + steps).
    """
    t = np.linspace(0, TS.shape[0] - 1 + steps, TS.shape[0] + steps, dtype=int)
    for i in range(1, order + 1):
        sint = (bias + np.sin(i * omega * t)).reshape(-1, 1)
        cost = (bias + np.cos(i * omega * t)).reshape(-1, 1)
        if i == 1:
            TP = np.hstack((sint, cost))
        else:
            TP = np.hstack((TP, sint, cost))
    return TP


def TPRC_Forecast_Train_Test_Ensemble(
    TS, tl, wl=0, steps=22, dl=0, hypers=None,
    nmember=10, isReforecast=False, noise_ini=0.0
):
    """Train an ESN ensemble and run multi-step rolling forecasts in one call.

    Each ensemble member is independently initialised and trained on the
    same training window. Forecasting is performed iteratively: at each
    step the ESN prediction is fed back as input with updated TP features.

    Parameters
    ----------
    TS : np.ndarray, shape (T, n_vars)
        Full multivariate time series (training + test combined).
    tl : int
        Training length (number of time steps).
    wl : int
        Number of test steps to skip at the start of the test period
        (warm-up buffer for skill calculation).
    steps : int
        Number of forecast lead-time steps.
    dl : int
        Warm-up steps discarded at the start of training.
    hypers : dict
        Hyperparameter dictionary passed to get_esn_from_hypers.
    nmember : int
        Number of independent ensemble members to train.
    isReforecast : bool
        If True, run hindcasts over the training period instead of the
        test period.
    noise_ini : float
        Standard deviation of Gaussian noise added to the initial state
        at each forecast step (> 0 enables stochastic perturbation).

    Returns
    -------
    Ypred_mean : np.ndarray, shape (T_test, n_vars, steps)
        Ensemble-mean forecast array. Axis 2 indexes lead time.
    Ytest : np.ndarray, shape (T_test, n_vars + n_tp)
        Test period input array (time series + TP features).
    ensemble_predictions : np.ndarray, shape (nmember, T_test, n_vars + n_tp, steps)
        Full predictions from all ensemble members.
    """
    tp_omega = 2 * np.pi / 12
    tp_order = 2
    tp_bias = 0

    TP = get_RCTP(TS, steps=steps, omega=tp_omega, order=tp_order, bias=tp_bias)
    TS_TP = np.hstack((TS, TP[:-steps]))
    Xtrain = TS_TP[:tl - 1]
    Ytrain = TS[1:tl]

    if isReforecast:
        ensemble_predictions = np.zeros(
            (nmember, TS_TP[dl:tl].shape[0], TS_TP[dl:tl].shape[1], steps))
    else:
        ensemble_predictions = np.zeros(
            (nmember, TS_TP[tl + wl:].shape[0], TS_TP[tl + wl:].shape[1], steps))

    for m in tqdm(range(nmember)):
        member_esn = get_esn_from_hypers(hypers)
        member_esn = member_esn.fit(Xtrain, Ytrain, warmup=dl)

        if isReforecast:
            Ytest = TS_TP[dl:tl]
        else:
            Ytest = TS_TP[tl + wl:]

        x = Ytest
        Ypred = np.zeros((x.shape[0], x.shape[1], steps))

        for j in range(steps):
            if j == 0:
                Ypred[:, :, j] = x
            else:
                noise = np.random.normal(0, noise_ini, size=x.shape)
                x += noise
                ts_prediction = member_esn.run(x)
                if isReforecast:
                    x = np.hstack((ts_prediction, TP[dl + j: tl + dl + j]))
                else:
                    x = np.hstack((ts_prediction, TP[tl + wl + j: -steps + j]))
                Ypred[:, :, j] = x

        ensemble_predictions[m] = Ypred

    Ypred_mean = np.mean(ensemble_predictions, axis=0)
    return Ypred_mean, Ytest, ensemble_predictions


def TPRC_Train_Ensemble(TS, tl, dl=0, hypers=None, nmember=10,
                        tp_omega=2*np.pi/12, tp_order=2, tp_bias=0):
    """Train an ensemble of ESNs and return the trained models with TP features.

    Separates the training phase from forecasting so that the same set of
    trained models can be reused across multiple test periods or variable
    subsets.

    Parameters
    ----------
    TS : np.ndarray, shape (T, n_vars)
        Full multivariate time series.
    tl : int
        Training length.
    dl : int
        Warm-up steps discarded at the start of training.
    hypers : dict
        Hyperparameter dictionary for get_esn_from_hypers.
    nmember : int
        Number of independent ensemble members to train.
    tp_omega : float
        Base angular frequency for TP features.
    tp_order : int
        Number of harmonics for TP features.
    tp_bias : float
        Constant offset for TP features.

    Returns
    -------
    trained_models : list of reservoirpy models
        List of nmember trained ESN models.
    TP : np.ndarray, shape (T, 2*tp_order)
        Temporal-period feature array (covering the training period only,
        steps=0).
    """
    TP = get_RCTP(TS, steps=0, omega=tp_omega, order=tp_order, bias=tp_bias)
    TS_TP = np.hstack((TS, TP))
    Xtrain = TS_TP[:tl - 1]
    Ytrain = TS[1:tl]

    trained_models = []
    for m in tqdm(range(nmember), desc="Training ensemble members"):
        member_esn = get_esn_from_hypers(hypers)
        member_esn = member_esn.fit(Xtrain, Ytrain, warmup=dl)
        trained_models.append(member_esn)

    return trained_models, TP


def TPRC_Forecast_Ensemble(trained_models, Ytest, TP_test=None, steps=22,
                           noise_ini=0.0, tp_order=2, tp_omega=2*np.pi/12):
    """Generate multi-step ensemble forecasts from pre-trained ESN models.

    At each lead time j, the ESN output is concatenated with the TP features
    for time step j and used as input for the next step.

    Parameters
    ----------
    trained_models : list of reservoirpy models
        Pre-trained ESN ensemble from TPRC_Train_Ensemble.
    Ytest : np.ndarray, shape (T_test, n_vars)
        Initial conditions for the test period.
    TP_test : np.ndarray or None, shape (T_test + steps, 2*tp_order)
        Temporal-period features for the test period plus forecast horizon.
        If None, recomputed from Ytest.
    steps : int
        Number of forecast lead-time steps.
    noise_ini : float
        Standard deviation of Gaussian noise added to the initial state.
    tp_order : int
        Number of harmonics (used only when TP_test is None).
    tp_omega : float
        Base angular frequency (used only when TP_test is None).

    Returns
    -------
    Ypred_mean : np.ndarray, shape (T_test, n_vars, steps)
        Ensemble-mean forecast. Axis 2 indexes lead time.
    ensemble_predictions : np.ndarray, shape (nmember, T_test, n_vars, steps)
        Forecasts from all individual ensemble members.
    """
    nmember = len(trained_models)
    if TP_test is None:
        TP_test = get_RCTP(Ytest, steps=steps, omega=tp_omega, order=tp_order, bias=0)

    Ytest_TP = np.hstack((Ytest, TP_test[:-steps]))
    ensemble_predictions = np.zeros((nmember, Ytest.shape[0], Ytest.shape[1], steps))

    for m in tqdm(range(nmember), desc="Forecasting with ensemble"):
        member_esn = trained_models[m]
        x = Ytest_TP.copy()
        Ypred = np.zeros((Ytest.shape[0], Ytest.shape[1], steps))

        for j in range(steps):
            if j == 0:
                Ypred[:, :, j] = Ytest
                noise = np.random.normal(0, noise_ini, size=x.shape)
                x += noise
            else:
                ts_prediction = member_esn.run(x)
                x = np.hstack((ts_prediction, TP_test[j: Ytest.shape[0] + j]))
                Ypred[:, :, j] = ts_prediction

        ensemble_predictions[m] = Ypred

    Ypred_mean = np.mean(ensemble_predictions, axis=0)
    return Ypred_mean, ensemble_predictions


def dimension_addition_ensemble_forecast(ds, tl, hypers, wl=0, dl=0,
                                         retain_var=('Nino34', 'WWV'),
                                         nmembers=10, steps=22,
                                         tp_omega=2*np.pi/12, tp_order=2,
                                         noise_ini=0.0):
    """Mode-addition experiment: measure the skill gain from adding each extra variable.

    Starting from a baseline model trained on `retain_var`, the function
    trains one additional model for each remaining variable in ds, adding it
    to the retained set. Results are indexed by the name of the added variable.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing all climate-mode variables.
    tl : int
        Training length (number of time steps).
    hypers : dict or list of dict
        Hyperparameter dict(s). If a list, each dict trains nmembers members;
        total ensemble size = len(hypers) * nmembers.
    wl : int
        Number of test steps to skip at the start of the test period.
    dl : int
        Warm-up steps discarded during training.
    retain_var : list or tuple of str
        Variables always included in the input (baseline set).
    nmembers : int
        Number of ensemble members per hyperparameter set.
    steps : int
        Number of forecast lead-time steps.
    tp_omega : float
        Base angular frequency for TP features.
    tp_order : int
        Number of harmonics for TP features.
    noise_ini : float
        Noise amplitude for stochastic initial-condition perturbation.

    Returns
    -------
    results : dict
        Mapping from key → ensemble predictions array
        (n_members, T_test, n_vars, steps). Key 'baseline' corresponds to
        the retain_var-only model; other keys are the added variable names.
    results_mean : dict
        Ensemble-mean forecasts for each key, shape (T_test, n_vars, steps).
    """
    all_vars = list(ds.data_vars)
    for var in retain_var:
        if var not in all_vars:
            raise ValueError(f"{var} not found in dataset")
    other_vars = [var for var in all_vars if var not in retain_var]

    if isinstance(hypers, list):
        total_members = len(hypers) * nmembers
        print(f"Configuration: {len(hypers)} hyperparameter sets × {nmembers} members = {total_members} total members")
    else:
        print(f"Configuration: 1 hyperparameter set × {nmembers} members = {nmembers} total members")

    print(f"\nDimension Addition Ensemble Forecast - Baseline: {'+'.join(retain_var)}")
    print(f"Testing: {'+'.join(retain_var)} + {other_vars}\n")

    def dataset_to_array(sub_ds):
        return np.stack([sub_ds[var].values for var in sub_ds.data_vars], axis=1)

    def run_forecast_ensemble(combo_vars):
        combo_ds = ds[combo_vars]
        TS = dataset_to_array(combo_ds)
        Ytest = TS[tl + wl:]

        if isinstance(hypers, list):
            all_trained_models = []
            for hyper in hypers:
                trained_models, TP = TPRC_Train_Ensemble(
                    TS, tl, dl=dl, hypers=hyper, nmember=nmembers,
                    tp_omega=tp_omega, tp_order=tp_order, tp_bias=0)
                all_trained_models.extend(trained_models)
            trained_models = all_trained_models
        else:
            trained_models, TP = TPRC_Train_Ensemble(
                TS, tl, dl=dl, hypers=hypers, nmember=nmembers,
                tp_omega=tp_omega, tp_order=tp_order, tp_bias=0)

        TP_test = get_RCTP(TS, steps=steps, omega=tp_omega,
                           order=tp_order, bias=0)[tl + wl:]
        Ypred_mean, ensemble_predictions = TPRC_Forecast_Ensemble(
            trained_models, Ytest, TP_test=TP_test, steps=steps,
            noise_ini=noise_ini, tp_order=tp_order, tp_omega=tp_omega)
        return Ypred_mean, ensemble_predictions

    results = {}
    results_mean = {}

    print(f"Running baseline ({'+'.join(retain_var)})...")
    Ypred_mean, ensemble_predictions = run_forecast_ensemble(list(retain_var))
    results['baseline'] = ensemble_predictions
    results_mean['baseline'] = Ypred_mean
    print(f"✓ Baseline completed (dim={len(retain_var)}, members={ensemble_predictions.shape[0]})")

    print(f"\nTesting {len(other_vars)} combinations...")
    for x_var in tqdm(other_vars, desc='Testing combinations'):
        Ypred_mean, ensemble_predictions = run_forecast_ensemble(list(retain_var) + [x_var])
        results[x_var] = ensemble_predictions
        results_mean[x_var] = Ypred_mean

    return results, results_mean


def dimension_reduction_ensemble_forecast(ds, tl, hypers, dl=0, wl=0,
                                          consern_var='Nino34', exclude_dims=1,
                                          nmembers=10, steps=22,
                                          tp_omega=2*np.pi/12, tp_order=2,
                                          noise_ini=0.0, include_baseline=True):
    """Mode-decoupling experiment: measure the skill drop from removing each variable.

    Trains a baseline model with all variables, then re-trains with each
    possible combination of `exclude_dims` variables removed. Results are
    indexed by the names of the excluded variable(s).

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing all climate-mode variables.
    tl : int
        Training length (number of time steps).
    hypers : dict or list of dict
        Hyperparameter dict(s). If a list, total ensemble size = len(hypers)
        * nmembers.
    dl : int
        Warm-up steps discarded during training.
    wl : int
        Number of test steps to skip at the start of the test period.
    consern_var : str
        Target variable that is always retained. Default 'Nino34'.
    exclude_dims : int
        Number of non-target variables to simultaneously exclude (≥ 1).
    nmembers : int
        Number of ensemble members per hyperparameter set.
    steps : int
        Number of forecast lead-time steps.
    tp_omega : float
        Base angular frequency for TP features.
    tp_order : int
        Number of harmonics for TP features.
    noise_ini : float
        Noise amplitude for stochastic initial-condition perturbation.
    include_baseline : bool
        Whether to include the full-variable baseline. Default True.

    Returns
    -------
    results : dict
        Mapping from key → ensemble predictions (n_members, T_test, n_vars, steps).
        Key 'baseline' is the full model; other keys are '+'-joined names of
        excluded variables (e.g. 'WWV', 'NPMM+SPMM').
    results_mean : dict
        Ensemble-mean forecasts for each key, shape (T_test, n_vars, steps).
    """
    from itertools import combinations

    all_vars = list(ds.drop_vars(consern_var).data_vars)
    n_vars = len(all_vars)
    if exclude_dims < 1 or exclude_dims > n_vars:
        raise ValueError(f"exclude_dims must be between 1 and {n_vars}")

    def dataset_to_array(sub_ds):
        return np.stack([sub_ds[var].values for var in sub_ds.data_vars], axis=1)

    def train_and_forecast(TS):
        Ytest = TS[tl + wl:]
        if isinstance(hypers, list):
            all_trained_models = []
            for hyper in hypers:
                trained_models, TP = TPRC_Train_Ensemble(
                    TS, tl, dl=dl, hypers=hyper, nmember=nmembers,
                    tp_omega=tp_omega, tp_order=tp_order, tp_bias=0)
                all_trained_models.extend(trained_models)
            TP_test = get_RCTP(TS, steps=steps, omega=tp_omega,
                               order=tp_order, bias=0)[tl + wl:]
            Ypred_mean, ensemble_predictions = TPRC_Forecast_Ensemble(
                all_trained_models, Ytest, TP_test=TP_test, steps=steps,
                noise_ini=noise_ini, tp_order=tp_order, tp_omega=tp_omega)
        else:
            trained_models, TP = TPRC_Train_Ensemble(
                TS, tl, dl=dl, hypers=hypers, nmember=nmembers,
                tp_omega=tp_omega, tp_order=tp_order, tp_bias=0)
            TP_test = get_RCTP(TS, steps=steps, omega=tp_omega,
                               order=tp_order, bias=0)[tl + wl:]
            Ypred_mean, ensemble_predictions = TPRC_Forecast_Ensemble(
                trained_models, Ytest, TP_test=TP_test, steps=steps,
                noise_ini=noise_ini, tp_order=tp_order, tp_omega=tp_omega)
        return Ypred_mean, ensemble_predictions

    results = {}
    results_mean = {}

    if isinstance(hypers, list):
        total_members = len(hypers) * nmembers
        print(f"Configuration: {len(hypers)} hyperparameter sets × {nmembers} members = {total_members} total members")
    else:
        print(f"Configuration: 1 hyperparameter set × {nmembers} members = {nmembers} total members")

    if include_baseline:
        print("\nTraining baseline model (all variables)...")
        full_vars = [consern_var] + all_vars
        TS_full = dataset_to_array(ds[full_vars])
        Ypred_mean_baseline, ensemble_pred_baseline = train_and_forecast(TS_full)
        results['baseline'] = ensemble_pred_baseline
        results_mean['baseline'] = Ypred_mean_baseline
        print(f"✓ Baseline completed (dim={len(full_vars)}, total_members={ensemble_pred_baseline.shape[0]})")

    exclude_combinations = list(combinations(all_vars, exclude_dims))
    print(f"\nTesting {len(exclude_combinations)} combinations (excluding {exclude_dims} variable(s))...")

    for exclude_vars in tqdm(exclude_combinations, desc=f"Exclude {exclude_dims} var(s)"):
        kept_vars = [consern_var] + [v for v in all_vars if v not in exclude_vars]
        TS = dataset_to_array(ds[kept_vars])
        key = '+'.join(exclude_vars)
        Ypred_mean, ensemble_predictions = train_and_forecast(TS)
        results[key] = ensemble_predictions
        results_mean[key] = Ypred_mean

    return results, results_mean


def dimension_addition_xro_forecast(ds, tl, xro_model,
                                    retain_var=('Nino34', 'WWV'),
                                    maskb=('Nino34', 'IOD'), n_month=19):
    """XRO mode-addition experiment: measure skill gain from adding each variable.

    Mirrors dimension_addition_ensemble_forecast for the XRO benchmark.
    Trains one XRO model per variable combination and runs reforecasts
    over the test period.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing all climate-mode variables.
    tl : slice or array-like
        Time slice passed to ds.sel(time=tl) to select the training period.
    xro_model : XRO
        Initialised XRO model instance (e.g. XRO(ncycle=12, ac_order=2)).
    retain_var : list or tuple of str
        Variables always included (baseline set).
    maskb : list or tuple of str
        Variables for which nonlinear (quadratic) terms are fitted.
    n_month : int
        Forecast horizon in months.

    Returns
    -------
    results : dict
        Mapping from key → forecast array (T_test, n_vars, steps). Key
        'baseline' is the retain_var-only forecast; other keys are the
        names of individually added variables.
    """
    all_vars = list(ds.data_vars)
    for var in retain_var:
        if var not in all_vars:
            raise ValueError(f"{var} not found in dataset")
    other_vars = [var for var in all_vars if var not in retain_var]

    print(f"\nXRO Model - Baseline: {'+'.join(retain_var)}")
    print(f"Testing: {'+'.join(retain_var)} + {other_vars}\n")

    def run_forecast_xro(combo_vars):
        combo_ds = ds[combo_vars]
        train_ds = combo_ds.sel(time=tl)
        fit_result = xro_model.fit_matrix(train_ds, maskb=maskb)
        test_ds = combo_ds.isel(time=slice(len(train_ds.time), None))
        forecast_ds = xro_model.reforecast(
            fit_ds=fit_result, init_ds=test_ds,
            n_month=n_month, ncopy=1, noise_type='zero')
        return forecast_ds.to_array().values.transpose(1, 0, 2)

    results = {}
    print(f"Running baseline ({'+'.join(retain_var)})...")
    results['baseline'] = run_forecast_xro(list(retain_var))
    print(f"✓ Baseline completed (dim={len(retain_var)})")

    print(f"\nTesting {len(other_vars)} combinations...")
    for x_var in tqdm(other_vars, desc='Testing combinations'):
        results[x_var] = run_forecast_xro(list(retain_var) + [x_var])

    return results


def dimension_decoupling_xro_forecast(ds, tl, xro_model,
                                      base_vars=('Nino34', 'WWV'),
                                      exclude_dims=1,
                                      maskb=('Nino34', 'IOD'), n_month=19,
                                      include_baseline=True):
    """XRO mode-decoupling experiment: measure skill drop from removing each variable.

    Mirrors dimension_reduction_ensemble_forecast for the XRO benchmark.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing all climate-mode variables.
    tl : slice or array-like
        Time slice for the training period (passed to ds.sel(time=tl)).
    xro_model : XRO
        Initialised XRO model instance.
    base_vars : list or tuple of str
        Variables always retained (never removed).
    exclude_dims : int
        Number of non-base variables to simultaneously exclude.
    maskb : list or tuple of str
        Variables for which nonlinear terms are fitted.
    n_month : int
        Forecast horizon in months.
    include_baseline : bool
        Whether to include the full-variable baseline. Default True.

    Returns
    -------
    results : dict
        Mapping from key → forecast array (T_test, n_vars, steps). Key
        'baseline' uses all variables; other keys are '+'-joined names of
        excluded variables.
    """
    from itertools import combinations

    all_vars = list(ds.data_vars)
    for var in base_vars:
        if var not in all_vars:
            raise ValueError(f"{var} not found in dataset")
    decouple_vars = [v for v in all_vars if v not in base_vars]

    if exclude_dims < 1 or exclude_dims > len(decouple_vars):
        raise ValueError(f"exclude_dims must be between 1 and {len(decouple_vars)}")

    print(f"\nXRO Model - Mode Decoupling Experiment")
    print(f"Base variables (always retained): {'+'.join(base_vars)}")
    print(f"Decoupling pool: {decouple_vars}")
    print(f"Excluding {exclude_dims} variable(s) at a time\n")

    def run_forecast_xro(combo_vars):
        combo_ds = ds[combo_vars]
        train_ds = combo_ds.sel(time=tl)
        fit_result = xro_model.fit_matrix(train_ds, maskb=maskb)
        test_ds = combo_ds.isel(time=slice(len(train_ds.time), None))
        forecast_ds = xro_model.reforecast(
            fit_ds=fit_result, init_ds=test_ds,
            n_month=n_month, ncopy=1, noise_type='zero')
        return forecast_ds.to_array().values.transpose(1, 0, 2)

    results = {}

    if include_baseline:
        print(f"Running baseline ({'+'.join(all_vars)})...")
        results['baseline'] = run_forecast_xro(all_vars)
        print(f"✓ Baseline completed (dim={len(all_vars)})")

    exclude_combinations = list(combinations(decouple_vars, exclude_dims))
    print(f"\nTesting {len(exclude_combinations)} combinations (excluding {exclude_dims} variable(s))...")

    for exclude_vars in tqdm(exclude_combinations, desc=f'Exclude {exclude_dims} var(s)'):
        retained_vars = list(base_vars) + [v for v in decouple_vars if v not in exclude_vars]
        key = '+'.join(exclude_vars)
        results[key] = run_forecast_xro(retained_vars)

    print(f"\n✓ Completed {len(results)} experiments")
    if include_baseline:
        print(f"  - 1 baseline (dim={len(all_vars)})")
    print(f"  - {len(exclude_combinations)} decoupling tests (dim={len(all_vars) - exclude_dims} each)")

    return results


# =============================================================================
# Section 3 — Analysis
# =============================================================================

def ndforecast_skill(Ypred, Ytest, showdim=0, ismv3=True, wl=12, plot=False):
    """Compute Pearson correlation and RMSE of a multi-step forecast.

    For each lead time from 1 to steps-1, the forecast and observation
    arrays are optionally smoothed with a 3-month rolling mean before
    computing the skill scores.

    Parameters
    ----------
    Ypred : np.ndarray, shape (T, n_vars, steps)
        Forecast array. Axis 2 indexes lead time.
    Ytest : np.ndarray, shape (T, n_vars)
        Observed values aligned with the test period.
    showdim : int
        Variable index (column of Ytest) to evaluate. Default 0 (Niño3.4).
    ismv3 : bool
        If True, apply a 3-month centred rolling mean before computing
        correlation and RMSE. Default True.
    wl : int
        Number of initial time steps to exclude from skill calculation
        (additional warm-up buffer). Default 12.
    plot : bool
        If True, display correlation and RMSE vs. lead-time plots.

    Returns
    -------
    R : np.ndarray, shape (steps,)
        Pearson correlation coefficient at each lead time. R[0] = 1.0.
    rmse : np.ndarray, shape (steps,)
        Root-mean-square error at each lead time. rmse[0] = 0.0.
    """
    steps = Ypred.shape[2]
    R = np.zeros(steps)
    rmse = np.zeros(steps)
    R[0] = 1
    rmse[0] = 0

    for lead_time in range(1, steps):
        if ismv3:
            observation = (pd.Series(Ytest[lead_time + wl:, showdim])
                           .rolling(window=3, min_periods=1).mean().values)
            prediction = (pd.Series(Ypred[wl:-lead_time, showdim, lead_time])
                          .rolling(window=3, min_periods=1).mean().values)
        else:
            observation = Ytest[lead_time + wl:, showdim]
            prediction = Ypred[wl:-lead_time, showdim, lead_time]

        R[lead_time] = np.corrcoef(observation, prediction)[0, 1]
        rmse[lead_time] = np.sqrt(np.mean((observation - prediction) ** 2))

    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        axes[0].plot(range(steps), R, marker='o', color='orangered')
        axes[0].set_title('Correlation Coefficient (R) vs Lead Time')
        axes[0].set_xlabel('Lead Time')
        axes[0].set_ylabel('R')
        axes[1].plot(range(steps), rmse, marker='o', color='orangered')
        axes[1].set_title('RMSE vs Lead Time')
        axes[1].set_xlabel('Lead Time')
        axes[1].set_ylabel('RMSE')
        plt.tight_layout()
        plt.show()

    return R, rmse


def calculate_ensemble_skill(results, Ytest, results_mean=None, wl=0,
                             lower_percentile=2.5, upper_percentile=97.5,
                             showdim=0, ismv3=True, return_members=False,
                             bootstrap_size=None, random_state=None):
    """Compute ensemble forecast skill with uncertainty bounds.

    For each key in results, evaluates the ensemble-mean skill (R and RMSE)
    and estimates uncertainty bounds using percentiles across individual
    members or bootstrap sub-ensembles.

    Parameters
    ----------
    results : dict
        Mapping from key → ensemble predictions array
        (n_members, T_test, n_vars, steps).
    Ytest : np.ndarray, shape (T_test, n_vars)
        Observed values.
    results_mean : dict or None
        Pre-computed ensemble means. If None, computed from results.
    wl : int
        Warm-up buffer excluded from skill calculation.
    lower_percentile : float
        Lower bound percentile for uncertainty. Default 2.5.
    upper_percentile : float
        Upper bound percentile for uncertainty. Default 97.5.
    showdim : int
        Variable index to evaluate. Default 0 (Niño3.4).
    ismv3 : bool
        Apply 3-month rolling mean before skill calculation. Default True.
    return_members : bool
        If True, include per-member skill arrays in the output dict.
    bootstrap_size : int or None
        If not None, compute bounds using all C(n_members, bootstrap_size)
        sub-ensemble combinations instead of individual members.
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    R : dict
        Per-key correlation skill dicts with keys 'avg', 'lower', 'upper'
        (and optionally 'members', 'bootstrap_samples', 'n_bootstrap').
    rmse : dict
        Per-key RMSE dicts with the same structure as R.
    """
    from itertools import combinations

    R = {}
    rmse = {}

    if random_state is not None:
        np.random.seed(random_state)

    for mode in results.keys():
        ensemble_predictions = results[mode]  # (n_members, T_test, n_vars, steps)
        n_members = ensemble_predictions.shape[0]

        if results_mean is not None and mode in results_mean:
            ensemble_mean = results_mean[mode]
        else:
            ensemble_mean = np.mean(ensemble_predictions, axis=0)

        R_avg, rmse_avg = ndforecast_skill(
            ensemble_mean, Ytest, wl=wl, showdim=showdim,
            ismv3=ismv3, plot=False)

        R_members = []
        rmse_members = []
        for member_idx in range(n_members):
            member_pred = ensemble_predictions[member_idx]
            R_member, rmse_member = ndforecast_skill(
                member_pred, Ytest, wl=wl, showdim=showdim,
                ismv3=ismv3, plot=False)
            R_members.append(R_member)
            rmse_members.append(rmse_member)

        R_members = np.array(R_members)
        rmse_members = np.array(rmse_members)

        if bootstrap_size is not None and bootstrap_size < n_members:
            all_combinations = list(combinations(range(n_members), bootstrap_size))
            n_bootstrap = len(all_combinations)
            print(f"Mode {mode}: Computing {n_bootstrap} bootstrap combinations "
                  f"(C({n_members}, {bootstrap_size}))")

            R_bootstrap_samples = []
            rmse_bootstrap_samples = []
            for combo_indices in all_combinations:
                combo_mean = np.mean(ensemble_predictions[list(combo_indices)], axis=0)
                R_combo, rmse_combo = ndforecast_skill(
                    combo_mean, Ytest, wl=wl, showdim=showdim,
                    ismv3=ismv3, plot=False)
                R_bootstrap_samples.append(R_combo)
                rmse_bootstrap_samples.append(rmse_combo)

            R_bootstrap_samples = np.array(R_bootstrap_samples)
            rmse_bootstrap_samples = np.array(rmse_bootstrap_samples)
            R_lower = np.percentile(R_bootstrap_samples, lower_percentile, axis=0)
            R_upper = np.percentile(R_bootstrap_samples, upper_percentile, axis=0)
            rmse_lower = np.percentile(rmse_bootstrap_samples, lower_percentile, axis=0)
            rmse_upper = np.percentile(rmse_bootstrap_samples, upper_percentile, axis=0)
        else:
            R_lower = np.percentile(R_members, lower_percentile, axis=0)
            R_upper = np.percentile(R_members, upper_percentile, axis=0)
            rmse_lower = np.percentile(rmse_members, lower_percentile, axis=0)
            rmse_upper = np.percentile(rmse_members, upper_percentile, axis=0)

        R[mode] = {'avg': R_avg, 'lower': R_lower, 'upper': R_upper}
        rmse[mode] = {'avg': rmse_avg, 'lower': rmse_lower, 'upper': rmse_upper}

        if return_members:
            R[mode]['members'] = R_members
            rmse[mode]['members'] = rmse_members
            if bootstrap_size is not None and bootstrap_size < n_members:
                R[mode]['bootstrap_samples'] = R_bootstrap_samples
                rmse[mode]['bootstrap_samples'] = rmse_bootstrap_samples
                R[mode]['n_bootstrap'] = n_bootstrap
                rmse[mode]['n_bootstrap'] = n_bootstrap

    return R, rmse

def cal_rmse(obs, pred):
    """Compute root mean squared error between obs and pred arrays."""
    return np.sqrt(np.mean((pred - obs) ** 2))


def fast_stochastic_ESN_error_growth(
    TS, hypers, steps, tl, perturb_dim=1, nmembers=100, init_perturb=0.02
):
    """
    Estimate absolute error growth by projecting fixed-amplitude initial
    perturbations forward through a trained ESN/DESN.

    Parameters
    ----------
    TS : ndarray, shape (T, D)
        Full multivariate time series (training + test).
    hypers : dict
        Hyperparameter dictionary accepted by ``get_esn_from_hypers``.
    steps : int
        Number of forecast steps to evaluate.
    tl : int
        Training length (number of time steps).
    perturb_dim : int, optional
        Number of leading dimensions to perturb.  Default 1.
    nmembers : int, optional
        Number of stochastic ensemble members.  Default 100.
    init_perturb : float, optional
        L2 norm of the initial perturbation vector.  Default 0.02.

    Returns
    -------
    dict
        Keys ``'mean'``, ``'upper'`` (97.5th pct), ``'lower'`` (2.5th pct),
        each an ndarray of length *steps* containing the error ratio
        :math:`\\delta_i / \\delta_0`.
    """
    Xtrain = TS[:tl - 1]
    Ytrain = TS[1:tl]
    Ytest = TS[tl:]

    esn = get_esn_from_hypers(hypers)
    esn = esn.fit(Xtrain, Ytrain)

    length = Ytest.shape[0]
    error_ratios = np.zeros((nmembers, steps))

    for i in tqdm(range(nmembers), desc='Stochastic Simulating'):
        # Random unit-vector perturbation of fixed amplitude
        direction = np.random.normal(size=(perturb_dim,))
        direction /= np.linalg.norm(direction)
        perturbation = np.tile(direction * init_perturb, (length, 1))

        X_test = np.column_stack([Ytest[:, :perturb_dim] + perturbation,
                                   Ytest[:, perturb_dim:]])
        init_error = cal_rmse(X_test[:, :perturb_dim], Ytest[:, :perturb_dim])

        x_perturb = X_test
        x_ref = Ytest
        esn.noise_rc = 0
        esn.noise_in = 0

        for j in range(steps):
            if j == 0:
                pred_ref = x_ref[:, :perturb_dim]
                pred_perturb = x_perturb[:, :perturb_dim]
            else:
                x_ref = esn.run(x_ref)
                x_perturb = esn.run(x_perturb)
                pred_ref = x_ref[:, :perturb_dim]
                pred_perturb = x_perturb[:, :perturb_dim]

            current_error = cal_rmse(pred_perturb, pred_ref)
            error_ratios[i, j] = current_error / init_error

    return {
        'mean': np.mean(error_ratios, axis=0),
        'upper': np.percentile(error_ratios, 97.5, axis=0),
        'lower': np.percentile(error_ratios, 2.5, axis=0),
    }

# =============================================================================
# Section 4 — Visualization
# =============================================================================

def plot_main_skills_with_legend(skill_dict: dict, styles: dict = None,
                                 skill_name='Skill', title=None,
                                 figsize=(12, 5), grid_on=True, legend=True,
                                 xticks=None, yticks=None, xlim=None,
                                 ylim=(0, 1), add_hline=True,
                                 legend_loc="upper center",
                                 legend_in_main=False, n_cols=1):
    """Plot forecast skill curves for multiple models with a styled legend.

    Supports both simple skill arrays and dicts with 'avg'/'lower'/'upper'
    keys (shaded confidence bands). When legend_in_main is False, the legend
    is placed in a separate panel to the right.

    Parameters
    ----------
    skill_dict : dict
        Mapping from model name → skill values. Values can be:
        - np.ndarray of shape (steps,): simple skill curve.
        - dict with keys 'avg', 'lower', 'upper': mean ± bounds.
    styles : dict or None
        Per-model style dicts. Supported keys: 'color', 'linestyle',
        'marker', 'linewidth', 'markersize', 'alpha', 'hollowmarker'.
    skill_name : str
        Y-axis label. Default 'Skill'.
    title : str or None
        Axes title. Omitted if None.
    figsize : tuple
        Figure size (width, height). Default (12, 5).
    grid_on : bool
        Whether to show a background grid. Default True.
    legend : bool
        Whether to draw a legend. Default True.
    xticks : array-like or None
        Custom x-axis tick positions. Auto-computed if None.
    yticks : array-like or None
        Custom y-axis tick positions. Defaults to 0.0–1.0 in steps of 0.1.
    xlim : tuple or None
        X-axis limits (min, max). If None, the full range is shown.
    ylim : tuple or None
        Y-axis limits. Default (0, 1).
    add_hline : bool
        Draw a dashed reference line at y = 0.5. Default True.
    legend_loc : str
        Legend anchor location string. Default 'upper center'.
    legend_in_main : bool
        If True, embed the legend inside the main axes; otherwise place it
        in a separate right-hand panel. Default False.
    n_cols : int
        Number of legend columns. Default 1.
    """
    def get_length(val):
        return len(val['avg']) if isinstance(val, dict) else len(val)

    steps = max(get_length(values) for values in skill_dict.values())
    x_axis = np.arange(0, steps)
    default_colors = cycle(plt.cm.tab10.colors)

    if legend_in_main:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)
        ax_legend = None
    else:
        fig, (ax, ax_legend) = plt.subplots(
            1, 2, figsize=figsize, dpi=300,
            gridspec_kw={"width_ratios": [4, 1]})

    plotted_keys = []

    for key, values in skill_dict.items():
        style = styles.get(key, {}) if styles else {}
        color = style.get('color', next(default_colors))
        linestyle = style.get('linestyle', '-')
        marker = style.get('marker', '')
        linewidth = style.get('linewidth', 1.5)
        markersize = style.get('markersize', 7)
        alpha = style.get('alpha', 1.0)
        hollowmarker = style.get('hollowmarker', False)

        if isinstance(values, dict):
            avg = np.full(steps, np.nan)
            lower = np.full(steps, np.nan)
            upper = np.full(steps, np.nan)
            avg[:len(values['avg'])] = values['avg']
            lower[:len(values['lower'])] = values['lower']
            upper[:len(values['upper'])] = values['upper']

            if xlim is not None:
                start, end = xlim
                mask = (x_axis >= start) & (x_axis < end)
                x_t, avg_t, lower_t, upper_t = x_axis[mask], avg[mask], lower[mask], upper[mask]
            else:
                x_t, avg_t, lower_t, upper_t = x_axis, avg, lower, upper

            ax.plot(x_t, avg_t, color=color, linestyle=linestyle, marker=marker,
                    linewidth=linewidth, markersize=markersize,
                    markerfacecolor='none' if hollowmarker else color,
                    alpha=alpha, label=key)
            ax.fill_between(x_t, lower_t, upper_t, color=color, alpha=0.5)
        else:
            y_values = np.full(steps, np.nan)
            y_values[:len(values)] = values

            if xlim is not None:
                start, end = xlim
                mask = (x_axis >= start) & (x_axis < end)
                x_t, y_t = x_axis[mask], y_values[mask]
            else:
                x_t, y_t = x_axis, y_values

            ax.plot(x_t, y_t, color=color, linestyle=linestyle, marker=marker,
                    linewidth=linewidth, markersize=markersize,
                    markerfacecolor='none' if hollowmarker else color,
                    alpha=alpha, label=key)

        plotted_keys.append(key)

    if add_hline:
        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1, label='y=0.5')
    if grid_on:
        ax.grid(True)
    if title:
        ax.set_title(title, fontsize=16)
    ax.set_xlabel('Lead time (months)', fontsize=14)
    ax.set_ylabel(skill_name, fontsize=14)

    if yticks is None:
        yticks = np.arange(0, 1.01, 0.1)
    ax.set_yticks(yticks)

    if xticks is None:
        xticks = np.arange(1, steps + 1, 2)
    ax.set_xticks(xticks)
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_elements = []
    if styles:
        for key in plotted_keys:
            if key in styles:
                s = styles[key]
                legend_elements.append(Line2D(
                    [0], [0],
                    color=s.get('color', next(default_colors)),
                    linestyle=s.get('linestyle', '-'),
                    marker=s.get('marker', ''),
                    linewidth=s.get('linewidth', 1.5),
                    markersize=s.get('markersize', 7),
                    markerfacecolor='none' if s.get('hollowmarker', False) else s.get('color', 'black'),
                    alpha=s.get('alpha', 1.0),
                    label=key))

    if legend:
        if legend_in_main:
            ax.legend(handles=legend_elements, loc=legend_loc, fontsize=12, ncols=n_cols)
        else:
            ax_legend.axis("off")
            ax_legend.legend(handles=legend_elements, loc=legend_loc,
                             fontsize=12, ncols=n_cols)

    plt.tight_layout()
    plt.show()


def visualize_skill_vs_baseline(skill_dict, n_cols=3, figsize=None,
                                focus_modes=None, ylabel='Correlation',
                                ylim=(0.2, 1.0), xlim=(1, 19),
                                ref_line=None, title=None,
                                baseline_color='deepskyblue',
                                mode_color='orangered',
                                legend_loc='upper right'):
    """Plot each mode's skill curve alongside the baseline in a subplot grid.

    Each subplot shows the baseline (blue) and one mode-removed/mode-added
    experiment (red), making it easy to visually assess per-mode contributions.

    Parameters
    ----------
    skill_dict : dict
        Mapping from mode name → skill values (array or dict with 'avg').
        Must contain a 'baseline' key.
    n_cols : int
        Number of subplot columns. Default 3.
    figsize : tuple or None
        Figure size. Auto-computed if None.
    focus_modes : list or None
        Subset of modes to plot. All non-baseline modes are used if None.
    ylabel : str
        Y-axis label. Default 'Correlation'.
    ylim : tuple
        Y-axis limits. Default (0.2, 1.0).
    xlim : tuple
        X-axis limits (lead-time range). Default (1, 19).
    ref_line : float or None
        Y-value for a dashed reference line (e.g. 0.5). Omitted if None.
    title : str or None
        Overall figure title. Omitted if None.
    baseline_color : str
        Colour for the baseline curve. Default 'deepskyblue'.
    mode_color : str
        Colour for the per-mode curve. Default 'orangered'.
    legend_loc : str
        Legend location string. Default 'upper right'.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if 'baseline' not in skill_dict:
        raise ValueError("skill_dict must contain a 'baseline' key")

    baseline_data = skill_dict['baseline']
    baseline_values = np.array(
        baseline_data['avg'] if isinstance(baseline_data, dict) and 'avg' in baseline_data
        else baseline_data)

    if focus_modes is None:
        focus_modes = [k for k in skill_dict if k != 'baseline']

    n_modes = len(focus_modes)
    n_rows = int(np.ceil(n_modes / n_cols))
    if figsize is None:
        figsize = (n_cols * 3.5, n_rows * 3)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    lead_times = np.arange(len(baseline_values))

    for idx, mode in enumerate(focus_modes):
        ax = axes[idx]
        if mode not in skill_dict:
            ax.axis('off')
            continue

        mode_data = skill_dict[mode]
        if isinstance(mode_data, dict):
            if 'avg' not in mode_data:
                ax.axis('off')
                continue
            mode_values = mode_data['avg']
        else:
            mode_values = np.array(mode_data)

        ax.plot(lead_times, baseline_values, linewidth=2.5, markersize=5,
                color=baseline_color, label='DESN', alpha=0.85, marker='o')
        ax.plot(lead_times, mode_values, linewidth=2.5, markersize=7,
                color=mode_color, label=f'$D_{{{mode}}}$', alpha=0.85, marker='x')

        if ref_line is not None:
            ax.axhline(y=ref_line, color='black', linestyle='--', linewidth=1.5)

        ax.set_title(f'Effect of {mode} Decoupling', fontsize=14, pad=10)
        if idx % n_cols == 0:
            ax.set_ylabel(ylabel, fontsize=12)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Lead Time (months)', fontsize=11)

        if xlim is not None:
            ax.set_xlim(xlim)
            ax.set_xticks(np.arange(xlim[0], xlim[1] + 1, 2))
        else:
            ax.set_xlim(1, len(lead_times) - 1)
        if ylim is not None:
            ax.set_ylim(ylim)
            ax.set_yticks(np.arange(ylim[0], ylim[1] + 0.01, 0.1))

        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.legend(loc=legend_loc, fontsize=12, framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=10)
        ax.tick_params(which='minor', length=3)
        ax.tick_params(which='major', length=5)

    for idx in range(n_modes, len(axes)):
        axes[idx].axis('off')

    if title is not None:
        plt.suptitle(title, fontsize=16, y=0.995)

    plt.tight_layout()
    return fig


def visualize_skill_comparison_vs_baseline(
    comparison_skill, n_cols=3, figsize=None, ylabel='Correlation',
    ylim=(0.2, 1.0), xlim=(1, 19), ref_line=None, title=None,
    subtitle_template='$D_{{{mode}}}$', colors=None, markers=None,
    show_difference=False, diff_reverse=False,
    show_errorbar=False, show_baseline=True,
):
    """Compare mode-attribution skill across multiple models in a subplot grid.

    Plots one subplot per climate mode (plus an optional baseline panel).
    Each subplot overlays skill curves from all models in comparison_skill,
    and optionally shades the difference between models.

    Parameters
    ----------
    comparison_skill : dict
        Nested dict of {model_name: skill_dict}, where skill_dict maps
        mode names to arrays or dicts with 'avg' (and optional 'lower'/
        'upper' confidence bounds). Must contain a 'baseline' key.
    n_cols : int
        Number of subplot columns. Default 3.
    figsize : tuple or None
        Figure size. Auto-computed if None.
    ylabel : str
        Y-axis label. Default 'Correlation'.
    ylim : tuple
        Y-axis limits. Default (0.2, 1.0).
    xlim : tuple
        X-axis limits in lead-time months. Default (1, 19).
    ref_line : float or None
        Y-value of a dashed reference line. Omitted if None.
    title : str or None
        Overall figure title. Omitted if None.
    subtitle_template : str
        Format string for subplot titles; '{mode}' is replaced by the mode
        name. Default '$D_{{{mode}}}$'. Use '$A_{{{mode}}}$' for addition
        experiments.
    colors : list or None
        Per-model line colours.
    markers : list or None
        Per-model marker styles.
    show_difference : bool
        Draw a filled region between the first two models' curves. Default False.
    diff_reverse : bool
        If False, difference = model[0] - model[1]; if True, reversed.
    show_errorbar : bool
        Draw confidence-bound error bars when bounds are available. Default False.
    show_baseline : bool
        Include a dedicated baseline subplot as the first panel. Default True.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    def _extract(data):
        if isinstance(data, dict):
            vals  = np.array(data['avg'])
            lower = np.array(data['lower']) if 'lower' in data else None
            upper = np.array(data['upper']) if 'upper' in data else None
        else:
            vals, lower, upper = np.array(data), None, None
        return vals, lower, upper

    def _plot_with_errbar(ax, x, vals, lower, upper, show_eb=False, **kwargs):
        line, = ax.plot(x, vals, **kwargs)
        if show_eb and lower is not None and upper is not None:
            ax.errorbar(x, vals,
                        yerr=[np.clip(vals - lower, 0, None),
                              np.clip(upper - vals, 0, None)],
                        fmt='none', ecolor=kwargs['color'],
                        elinewidth=1.5, capsize=2, capthick=1.5,
                        alpha=kwargs.get('alpha', 0.85),
                        zorder=line.get_zorder() - 1)

    def _decorate_ax(ax, idx, n_rows, n_cols, mode_label):
        if ref_line is not None:
            ax.axhline(ref_line, color='black', linestyle='--',
                       linewidth=1.5, zorder=0)
        ax.set_title(mode_label, fontsize=14, pad=10)
        if idx % n_cols == 0:
            ax.set_ylabel(ylabel, fontsize=12)
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Lead Time (months)', fontsize=11)
        if xlim:
            ax.set_xlim(xlim)
            ax.set_xticks(np.arange(xlim[0], xlim[1] + 1, 2))
        if ylim:
            ax.set_ylim(ylim)
            ax.set_yticks(np.arange(ylim[0], ylim[1] + 0.01, 0.1))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(labelsize=10)
        ax.tick_params(which='minor', length=3)
        ax.tick_params(which='major', length=5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    model_names = list(comparison_skill.keys())
    focus_modes = [m for m in comparison_skill[model_names[0]] if m != 'baseline']

    n_plots = len(focus_modes) + (1 if show_baseline else 0)
    n_rows  = int(np.ceil(n_plots / n_cols))
    if figsize is None:
        figsize = (n_cols * 3.5, n_rows * 3)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()

    if colors is None:
        colors  = ['deepskyblue', 'orangered', 'forestgreen',
                   'purple', 'orange', 'brown', 'pink', 'gray']
    if markers is None:
        markers = ['o', 'x', '^', 'D', 'v', '<', '>', 'p']

    ax_offset = 0

    # Baseline panel
    if show_baseline:
        ax = axes[0]
        ax_offset = 1
        for m_idx, m_name in enumerate(model_names):
            bl_data = comparison_skill[m_name].get('baseline')
            if bl_data is None:
                continue
            vals, lower, upper = _extract(bl_data)
            _plot_with_errbar(ax, np.arange(len(vals)), vals, lower, upper,
                              show_eb=show_errorbar, label=m_name,
                              marker=markers[m_idx % len(markers)],
                              linewidth=2.5, markersize=5,
                              color=colors[m_idx % len(colors)],
                              alpha=0.85, zorder=2)
        _decorate_ax(ax, 0, n_rows, n_cols, 'Baseline')

    # Mode subplots
    for idx, mode in enumerate(focus_modes):
        ax     = axes[idx + ax_offset]
        ax_idx = idx + ax_offset

        for m_idx, m_name in enumerate(model_names):
            mode_data = comparison_skill[m_name].get(mode)
            if mode_data is None:
                continue
            vals, lower, upper = _extract(mode_data)

            min_len = min(
                len(_extract(comparison_skill[mn][mode])[0])
                for mn in model_names if mode in comparison_skill[mn])
            vals = vals[:min_len]

            _plot_with_errbar(ax, np.arange(min_len), vals,
                              lower[:min_len] if lower is not None else None,
                              upper[:min_len] if upper is not None else None,
                              show_eb=show_errorbar,
                              label=m_name,
                              marker=markers[m_idx % len(markers)],
                              linewidth=2.5, markersize=5,
                              color=colors[m_idx % len(colors)],
                              alpha=0.85, zorder=2)

        # Difference shading between first two models
        if show_difference and len(model_names) >= 2:
            d0 = comparison_skill[model_names[0]].get(mode)
            d1 = comparison_skill[model_names[1]].get(mode)
            if d0 is not None and d1 is not None:
                v0 = _extract(d0)[0]
                v1 = _extract(d1)[0]
                min_len = min(len(v0), len(v1))
                x = np.arange(min_len)
                diff = (v1[:min_len] - v0[:min_len]) if diff_reverse else (v0[:min_len] - v1[:min_len])
                ax.fill_between(x, 0, diff,
                                where=diff > 0,
                                color=colors[0], alpha=0.15, zorder=1)
                ax.fill_between(x, 0, diff,
                                where=diff < 0,
                                color=colors[1], alpha=0.15, zorder=1)

        mode_label = subtitle_template.format(mode=mode)
        _decorate_ax(ax, ax_idx, n_rows, n_cols, mode_label)

    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    if title is not None:
        plt.suptitle(title, fontsize=16, y=0.995)

    plt.tight_layout()
    return fig

# =============================================================================
# Section 5 — Helper Func
# =============================================================================

def reorder_and_rename_results(results, key_order=None, remove_prefix='-'):
    """Reorder and optionally rename keys in a results dictionary.

    Useful for standardising the output of dimension_reduction experiments
    (whose keys carry a '-' prefix) before plotting.

    Parameters
    ----------
    results : dict
        Input dictionary, e.g. {'-NPMM': array, 'baseline': array, ...}.
    key_order : list or None
        Desired key order. Keys not in key_order are appended at the end.
        If None, only renaming is performed (order preserved).
    remove_prefix : str or None
        Prefix to strip from all keys. Default '-'. Pass None to skip.

    Returns
    -------
    reordered_results : dict
        Dictionary with renamed keys in the requested order.
    """
    from collections import OrderedDict

    renamed_results = {}
    for key, value in results.items():
        if remove_prefix and key.startswith(remove_prefix):
            new_key = key[len(remove_prefix):]
        else:
            new_key = key
        renamed_results[new_key] = value

    if key_order is not None:
        reordered_results = OrderedDict()
        for key in key_order:
            if key in renamed_results:
                reordered_results[key] = renamed_results[key]
        for key in renamed_results:
            if key not in reordered_results:
                reordered_results[key] = renamed_results[key]
        return dict(reordered_results)
    else:
        return renamed_results


def convert_to_standard_calendar(time):
    """Convert a mixed-type time array to a list of standard datetime objects.

    Handles cftime non-standard calendars (e.g. 360-day, noleap), NumPy
    datetime64, and native Python datetime.

    Parameters
    ----------
    time : array-like
        Array of time values of any supported type: cftime.datetime,
        numpy.datetime64, numpy.float32, or datetime.datetime.

    Returns
    -------
    standard_time : list of datetime.datetime
        Corresponding list of Python datetime objects.

    Raises
    ------
    ValueError
        If an unrecognised time type is encountered.
    """
    standard_time = []
    for t in time:
        if isinstance(t, cftime.datetime):
            standard_time.append(datetime(t.year, t.month, t.day))
        elif isinstance(t, np.datetime64):
            standard_time.append(pd.to_datetime(str(t)).to_pydatetime())
        elif isinstance(t, np.float32):
            standard_time.append(pd.to_datetime(str(t)).to_pydatetime())
        elif isinstance(t, datetime):
            standard_time.append(t)
        else:
            raise ValueError(f"Unrecognised time type: {type(t)}")
    return standard_time


def standardize_time_to_month_start(ds, time_dim='time'):
    """Snap all time coordinates in a dataset to the first day of each month.

    Converts any non-standard calendar times (cftime, etc.) to Python
    datetime objects and then replaces each timestamp with the first day of
    its month, ensuring consistent monthly indexing.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset whose time coordinate may use a non-standard calendar.
    time_dim : str
        Name of the time dimension. Default 'time'.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with the time coordinate replaced by month-start datetimes.
    """
    time = ds[time_dim].values
    standard_time = convert_to_standard_calendar(time)
    standardized_time = [datetime(t.year, t.month, 1) for t in standard_time]
    ds = ds.assign_coords({time_dim: standardized_time})
    return ds