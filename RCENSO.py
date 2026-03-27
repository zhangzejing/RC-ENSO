import reservoirpy as rpy
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import cycle
from tqdm import tqdm
from datetime import datetime
import cftime
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, FixedLocator
from matplotlib.colors import LinearSegmentedColormap
from copy import deepcopy
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
from scipy.stats import ortho_group
from scipy.linalg import orth,qr
from scipy.interpolate import interp1d
import itertools

# 全局设置 Matplotlib 样式
plt.rcParams['font.family'] = 'sans-serif'  # 无衬线字体
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica'] + plt.rcParams['font.sans-serif'] 
plt.rcParams['axes.linewidth'] = 1.5  # 坐标轴线宽
plt.rcParams['legend.frameon'] = False  # 图例无边框
plt.rcParams['xtick.major.width'] = 1.5  # X轴主刻度线宽
plt.rcParams['ytick.major.width'] = 1.5  # Y轴主刻度线宽
plt.rcParams['xtick.minor.width'] = 1.  # X轴次刻度线宽
plt.rcParams['ytick.minor.width'] = 1.  # Y轴次刻度线宽
plt.rcParams['xtick.labelsize'] = 12  # X轴刻度标签字体大小
plt.rcParams['ytick.labelsize'] = 12  # Y轴刻度标签字体大小
plt.rcParams['font.size'] = 14  # 全局字体大小
plt.rcParams['legend.fontsize'] = 12  # 图例字体大小

def Create_New_ESN(units=4000,
                           lr=1,
                           sr=0.95,
                           rc_connectivity=0.2,
                           noise_rc=0.02,
                           noise_in=0.0,
                           output_dim=1,
                           input_scaling = 1,
                           W=None,
                           ridge=5e-6,
                           input_connectivity=0.1,
                           use_raw_input=False,
                           seed=None):
    """
    Create a new Echo State Network (ESN).

    Parameters:
    units (int): Number of neurons in the reservoir.
    lr (float): Leak rate.
    sr (float): Spectral radius.
    rc_connectivity (float): Reservoir connectivity.
    noise_rc (float): Reservoir noise.
    noise_in (float): Input noise.
    output_dim (int): Output dimension.
    ridge (float): Ridge regression parameter.
    seed (int): Random number seed.

    Returns:
    esn0 (object): The created Echo State Network object.
    """
    if W is None:    
        res0 = rpy.nodes.Reservoir(units=units, 
                                lr=lr, 
                                sr=sr,
                                rc_connectivity=rc_connectivity,
                                input_connectivity=input_connectivity,
                                input_scaling = input_scaling,
                                noise_rc=noise_rc,
                                noise_in=noise_in,seed=seed)
    else:
        res0 = rpy.nodes.Reservoir(units=units,
                                lr=lr,
                                sr=sr,
                                noise_rc=noise_rc,
                                noise_in=noise_in,
                                W=W,
                                input_connectivity=input_connectivity,
                                input_scaling = input_scaling,
                                seed=seed)
    readout0 = rpy.nodes.Ridge(output_dim=output_dim, ridge=ridge)
    esn0 = res0 >> readout0
    esn0.use_raw_input = use_raw_input
    return esn0

def Create_Online_ESN(units=4000,
                           lr=1,
                           sr=0.95,
                           rc_connectivity=0.2,
                           noise_rc=0.02,
                           noise_in=0.02,
                           output_dim=1,
                           W=None,
                           alpha=5e-6,
                           input_connectivity=0.1,
                           input_scaling = 1,
                           use_raw_input=False,
                           seed=None):
    """
    Create a new Online Echo State Network (ESN).

    Parameters:
    units (int): Number of neurons in the reservoir.
    lr (float): Leak rate.
    sr (float): Spectral radius.
    rc_connectivity (float): Reservoir connectivity.
    noise_rc (float): Reservoir noise.
    noise_in (float): Input noise.
    output_dim (int): Output dimension.
    ridge (float): Ridge regression parameter.
    seed (int): Random number seed.

    Returns:
    esn0 (object): The created Echo State Network object.
    """
    if W is None:    
        res0 = rpy.nodes.Reservoir(units=units, 
                                lr=lr, 
                                sr=sr,
                                rc_connectivity=rc_connectivity,
                                input_connectivity=input_connectivity,
                                input_scaling = input_scaling,
                                noise_rc=noise_rc,
                                noise_in=noise_in,seed=seed)
    else:
        res0 = rpy.nodes.Reservoir(units=units,
                                lr=lr,
                                sr=sr,
                                noise_rc=noise_rc,
                                noise_in=noise_in,
                                W=W,
                                input_connectivity=input_connectivity,
                                input_scaling = input_scaling,
                                seed=seed)
    force = rpy.nodes.FORCE(output_dim=output_dim, alpha=alpha)
    esn0 = res0 >> force
    esn0.use_raw_input = use_raw_input
    return esn0

def Create_New_IPESN(units=4000,
                           lr=1,
                           sr=0.95,
                           rc_connectivity=0.2,
                           noise_rc=0.0,
                           noise_in=0.0,
                           output_dim=1,
                           W=None,
                           ridge=5e-6,
                           use_raw_input=False,
                           input_connectivity=0.1,
                           input_scaling = 1,
                           seed=None):
    """
    Create a new IP Echo State Network (IPESN).

    Parameters:
    units (int): Number of neurons in the reservoir.
    lr (float): Leak rate.
    sr (float): Spectral radius.
    rc_connectivity (float): Reservoir connectivity.
    noise_rc (float): Reservoir noise.
    noise_in (float): Input noise.
    output_dim (int): Output dimension.
    ridge (float): Ridge regression parameter.
    seed (int): Random number seed.

    Returns:
    esn0 (object): The created Echo State Network object.
    """
    if W is None:    
        res0 = rpy.nodes.IPReservoir(units=units, 
                                lr=lr, 
                                sr=sr,
                                rc_connectivity=rc_connectivity,
                                input_connectivity=input_connectivity,
                                noise_rc=noise_rc,
                                input_scaling = input_scaling,
                                noise_in=noise_in,seed=seed)
    else:
        res0 = rpy.nodes.IPReservoir(units=units,
                                lr=lr,
                                sr=sr,
                                noise_rc=noise_rc,
                                noise_in=noise_in,
                                W=W,
                                input_connectivity=input_connectivity,
                                input_scaling = input_scaling,
                                seed=seed)
    readout0 = rpy.nodes.Ridge(output_dim=output_dim, ridge=ridge)
    esn0 = res0>>readout0
    esn0.use_raw_input = use_raw_input
    return esn0

def Create_New_Reservoir(units=4000,
                           lr=1,
                           sr=0.95,
                           rc_connectivity=0.2,
                           noise_rc=0.0,
                           noise_in=0.0,
                           output_dim=1,
                           W=None,
                           ridge=5e-6,
                           use_raw_input=False,
                           input_connectivity=0.1,
                           seed=None,ip_reservoir=False):
    """
    Create a new IP Echo State Network (IPESN).
    Parameters:
    units (int): Number of neurons in the reservoir.
    lr (float): Leak rate.
    sr (float): Spectral radius.
    rc_connectivity (float): Reservoir connectivity.
    noise_rc (float): Reservoir noise.
    noise_in (float): Input noise.
    output_dim (int): Output dimension.
    ridge (float): Ridge regression parameter.
    seed (int): Random number seed.

    Returns:
    esn0 (object): The created Echo State Network object.
    """
    if ip_reservoir:
        if W is None:    
            res0 = rpy.nodes.IPReservoir(units=units, 
                                        lr=lr, 
                                        sr=sr,
                                        rc_connectivity=rc_connectivity,
                                        input_connectivity=input_connectivity,
                                        noise_rc=noise_rc,
                                        noise_in=noise_in,seed=seed)
        else:
            res0 = rpy.nodes.IPReservoir(units=units,
                                        lr=lr,
                                        sr=sr,
                                        noise_rc=noise_rc,
                                        noise_in=noise_in,
                                        W=W,
                                        input_connectivity=input_connectivity,
                                        seed=seed)
    else:
        if W is None:    
            res0 = rpy.nodes.Reservoir(units=units, 
                                        lr=lr, 
                                        sr=sr,
                                        rc_connectivity=rc_connectivity,
                                        input_connectivity=input_connectivity,
                                        noise_rc=noise_rc,
                                        noise_in=noise_in,seed=seed)
        else:
            res0 = rpy.nodes.Reservoir(units=units,
                                        lr=lr,
                                        sr=sr,
                                        noise_rc=noise_rc,
                                        noise_in=noise_in,
                                        W=W,
                                        input_connectivity=input_connectivity,
                                        seed=seed)
    esn0 = res0
    esn0.use_raw_input = use_raw_input
    return esn0

def Create_Deep_ESN(units: list, lr: list, sr: list, rc_connectivity: list = None,
                    input_scaling: list = None,
                    W: list = None, Win: list = None,
                    noise_rc: list = None, noise_in: list = None, 
                    input_connectivity: list = None,
                    output_dim=1, ridge=5e-6, seed=None,
                    deep_ip_list: list = None):
    """
    创建一个深度 ESN 模型。

    参数:
        units (list): 每层 reservoir 的神经元数量。
        lr (list): 每层的泄露率 (leaking rate)。
        sr (list): 每层的光谱半径 (spectral radius)。
        rc_connectivity (list): 每层的稀疏连接度（可选，只有 W 为 None 时传递）。
        W (list): 每层的内部权重矩阵（可选，传递后 rc_connectivity 不传递）。
        Win (list): 每层的输入权重矩阵（可选，传递后 input_connectivity 不传递）。
        noise_rc (list): 每层 reservoir 的噪声（默认 0）。
        noise_in (list): 每层输入的噪声（默认 0）。
        input_connectivity (list): 每层的输入连接密度（可选，只有 Win 为 None 时传递）。
        output_dim (int): 输出层的维度（默认 1）。
        ridge (float): Ridge 回归的正则化系数（默认 5e-6）。
        seed (int): 随机种子（可选）。
        deep_ip_list (list): 每层是否为 IP Reservoir 的布尔值列表（默认 False）。
    
    返回:
        object: 构建的深度 ESN 模型。
    """
    # 参数检查
    if not (len(units) == len(lr) == len(sr)):
        raise ValueError("Parameters 'units', 'lr', and 'sr' must have the same length.")
    if rc_connectivity is not None and len(rc_connectivity) != len(units):
        raise ValueError("Parameter 'rc_connectivity' must have the same length as 'units' or be None.")
    if W is not None and len(W) != len(units):
        raise ValueError("Parameter 'W' must have the same length as 'units' or be None.")
    if Win is not None and len(Win) != len(units):
        raise ValueError("Parameter 'Win' must have the same length as 'units' or be None.")
    if input_connectivity is not None and len(input_connectivity) != len(units):
        raise ValueError("Parameter 'input_connectivity' must have the same length as 'units' or be None.")
    if deep_ip_list is not None and len(deep_ip_list) != len(units):
        raise ValueError("Parameter 'deep_ip_list' must have the same length as 'layers' or be None.")
    
    # 设置默认值
    if noise_rc is None:
        noise_rc = [0.0] * len(units)
    if noise_in is None:
        noise_in = [0.0] * len(units)
    if input_connectivity is None:
        # 第一层为 0.1，其余层为 1.0
        input_connectivity = [0.1] + [1.0] * (len(units) - 1)
    if rc_connectivity is None:
        # 默认稀疏连接度为 0.1
        rc_connectivity = [0.1] * len(units)
    if deep_ip_list is None:
        # 默认所有层都为 False
        deep_ip_list = [False] * len(units)
    
    # 初始化输入和输出节点
    input = rpy.nodes.Input()
    readout = rpy.nodes.Ridge(output_dim=output_dim, ridge=ridge)
    
    # 初始化路径容器
    path1 = []  # 输入到每层 reservoir 的路径
    path2 = []  # reservoir 到输出的路径

    # 构建每一层的 reservoir
    for i in range(len(units)):
        print(f"Layer {i+1}: units={units[i]}, sr={sr[i]}, lr={lr[i]}, ip_reservoir={deep_ip_list[i]}")
        
        # 构建 reservoir 时，动态传递参数
        reservoir_params = {
            "units": units[i],
            "sr": sr[i],
            "lr": lr[i],
            "noise_rc": noise_rc[i],
            "noise_in": noise_in[i],
            "seed": seed,
            "input_scaling": input_scaling[i],
        }
        
        # 如果 W[i] 为空，则传递 rc_connectivity
        if W is None or W[i] is None:
            reservoir_params["rc_connectivity"] = rc_connectivity[i]
        else:
            reservoir_params["W"] = W[i]  # 如果 W[i] 不为空，传递 W
        
        # 如果 Win[i] 为空，则传递 input_connectivity
        if Win is None or Win[i] is None:
            reservoir_params["input_connectivity"] = input_connectivity[i]
        else:
            reservoir_params["Win"] = Win[i]  # 如果 Win[i] 不为空，传递 Win
        
        # 根据 ip_reservoir 决定创建普通 Reservoir 或 IPReservoir
        if deep_ip_list[i]:
            reservoir = rpy.nodes.IPReservoir(**reservoir_params)
        else:
            reservoir = rpy.nodes.Reservoir(**reservoir_params)

        # 构建路径
        path1.append(input >> reservoir)
        path2.append(reservoir >> readout)
        input = reservoir  # 将当前 reservoir 作为下一个输入

    # 合并路径
    esn = rpy.merge(*path1, *path2)  # 合并路径列表
    
    return esn

def Create_Deep_ESN_with_Reservoirs(units: list, lr: list, sr: list, rc_connectivity: list = None,
                    W: list = None, Win: list = None,
                    noise_rc: list = None, noise_in: list = None, 
                    input_connectivity: list = None,
                    input_scaling: list = None,
                    output_dim=1, ridge=5e-6, seed=None,
                    deep_ip_list: list = None):
    """
    创建一个深度 ESN 模型。

    参数:
        units (list): 每层 reservoir 的神经元数量。
        lr (list): 每层的泄露率 (leaking rate)。
        sr (list): 每层的光谱半径 (spectral radius)。
        rc_connectivity (list): 每层的稀疏连接度（可选，只有 W 为 None 时传递）。
        W (list): 每层的内部权重矩阵（可选，传递后 rc_connectivity 不传递）。
        Win (list): 每层的输入权重矩阵（可选，传递后 input_connectivity 不传递）。
        noise_rc (list): 每层 reservoir 的噪声（默认 0）。
        noise_in (list): 每层输入的噪声（默认 0）。
        input_connectivity (list): 每层的输入连接密度（可选，只有 Win 为 None 时传递）。
        output_dim (int): 输出层的维度（默认 1）。
        ridge (float): Ridge 回归的正则化系数（默认 5e-6）。
        seed (int): 随机种子（可选）。
        deep_ip_list (list): 每层是否为 IP Reservoir 的布尔值列表（默认 False）。
    
    返回:
        object: 构建的深度 ESN 模型。
    """
    # 参数检查
    if not (len(units) == len(lr) == len(sr)):
        raise ValueError("Parameters 'units', 'lr', and 'sr' must have the same length.")
    if rc_connectivity is not None and len(rc_connectivity) != len(units):
        raise ValueError("Parameter 'rc_connectivity' must have the same length as 'units' or be None.")
    if W is not None and len(W) != len(units):
        raise ValueError("Parameter 'W' must have the same length as 'units' or be None.")
    if Win is not None and len(Win) != len(units):
        raise ValueError("Parameter 'Win' must have the same length as 'units' or be None.")
    if input_connectivity is not None and len(input_connectivity) != len(units):
        raise ValueError("Parameter 'input_connectivity' must have the same length as 'units' or be None.")
    if deep_ip_list is not None and len(deep_ip_list) != len(units):
        raise ValueError("Parameter 'deep_ip_list' must have the same length as 'units' or be None.")
    
    # 设置默认值
    if noise_rc is None:
        noise_rc = [0.0] * len(units)
    if noise_in is None:
        noise_in = [0.0] * len(units)
    if input_connectivity is None:
        # 第一层为 0.1，其余层为 1.0
        input_connectivity = [0.1] + [1.0] * (len(units) - 1)
    if rc_connectivity is None:
        # 默认稀疏连接度为 0.1
        rc_connectivity = [0.1] * len(units)
    if deep_ip_list is None:
        # 默认所有层都为 False
        deep_ip_list = [False] * len(units)
    if seed is None:
        seed = [None] * len(units)
    
    # 初始化输入和输出节点
    input = rpy.nodes.Input()
    readout = rpy.nodes.Ridge(output_dim=output_dim, ridge=ridge)
    concat = rpy.nodes.Concat()
    # 初始化路径容器
    path1 = []  # 输入到每层 reservoir 的路径
    path2 = []  # reservoir 到输出的路径
    path3 = []

    # 构建每一层的 reservoir
    for i in range(len(units)):
        print(f"Layer {i+1}: units={units[i]}, sr={sr[i]}, lr={lr[i]}, ip_reservoir={deep_ip_list[i]}")
        
        # 构建 reservoir 时，动态传递参数
        reservoir_params = {
            "units": units[i],
            "sr": sr[i],
            "lr": lr[i],
            "noise_rc": noise_rc[i],
            "noise_in": noise_in[i],
            "seed": seed[i],
            "input_scaling": input_scaling[i],
        }
        
        # 如果 W[i] 为空，则传递 rc_connectivity
        if W is None or W[i] is None:
            reservoir_params["rc_connectivity"] = rc_connectivity[i]
        else:
            reservoir_params["W"] = W[i]  # 如果 W[i] 不为空，传递 W
        
        # 如果 Win[i] 为空，则传递 input_connectivity
        if Win is None or Win[i] is None:
            reservoir_params["input_connectivity"] = input_connectivity[i]
        else:
            reservoir_params["Win"] = Win[i]  # 如果 Win[i] 不为空，传递 Win
        
        # 根据 ip_reservoir 决定创建普通 Reservoir 或 IPReservoir
        if deep_ip_list[i]:
            reservoir = rpy.nodes.IPReservoir(**reservoir_params)
        else:
            reservoir = rpy.nodes.Reservoir(**reservoir_params)

        # 构建路径
        path1.append(input >> reservoir)
        path2.append(reservoir >> readout)
        path3.append(reservoir >> concat)
        input = reservoir  # 将当前 reservoir 作为下一个输入

    # 合并路径
    esn = rpy.merge(*path1, *path2)  # 合并路径列表
    reservoirs = rpy.merge(*path1, *path3)  # 合并路径列表    
    return esn,reservoirs

def get_esns_from_hyperpara_dict(hypers):
    """
    从超参数字典中创建多个 ESN，包括普通 ESN、在线 ESN、IP ESN 和 Deep ESN。

    参数:
        hypers (dict): 包含多个模型超参数的字典。

    返回:
        dict: 包含模型名称和对应 ESN 的字典。
    """
    esn_dict = {}
    
    for model_name in hypers.keys():
        # 提取超参数
        params = get_hyperparameters_for_model(model_name, hypers)
        
        # 根据超参数创建 ESN
        if params["deep"]:  # 如果是 Deep ESN
            esn_dict[model_name] = Create_Deep_ESN(
                units=params["units"],
                sr=params["sr"],
                lr=params["lr"],
                rc_connectivity=params["rc_connectivity"],
                noise_rc=params["noise_rc"],
                noise_in=params["noise_in"],
                ridge=params["ridge"],
                input_connectivity=params["input_connectivity"],
                input_scaling=params["input_scaling"],
                output_dim=params["output_dim"],
                seed=params["seed"],
                deep_ip_list=params["deep_ip_list"],
            )
        elif params["online"]:  # 如果是在线 ESN
            esn_dict[model_name] = Create_Online_ESN(
                units=params["units"],
                lr=params["lr"],
                sr=params["sr"],
                rc_connectivity=params["rc_connectivity"],
                noise_rc=params["noise_rc"],
                noise_in=params["noise_in"],
                output_dim=params["output_dim"],
                input_connectivity=params["input_connectivity"],
                input_scaling=params["input_scaling"],
                alpha=params["alpha"],
                seed=params["seed"],
                use_raw_input=params["use_raw_input"]
            )
        elif params["ip_reservoir"]:  # 如果是 IP ESN
            esn_dict[model_name] = Create_New_IPESN(
                units=params["units"],
                lr=params["lr"],
                sr=params["sr"],
                rc_connectivity=params["rc_connectivity"],
                noise_rc=params["noise_rc"],
                noise_in=params["noise_in"],
                output_dim=params["output_dim"],
                input_connectivity=params["input_connectivity"],
                input_scaling=params["input_scaling"],
                ridge=params["ridge"],
                seed=params["seed"],
                use_raw_input=params["use_raw_input"]
            )
        else:  # 普通 ESN
            esn_dict[model_name] = Create_New_ESN(
                units=params["units"],
                lr=params["lr"],
                sr=params["sr"],
                rc_connectivity=params["rc_connectivity"],
                noise_rc=params["noise_rc"],
                noise_in=params["noise_in"],
                output_dim=params["output_dim"],
                input_connectivity=params["input_connectivity"],
                input_scaling=params["input_scaling"],
                ridge=params["ridge"],
                seed=params["seed"],
                use_raw_input=params["use_raw_input"]
            )
    
    return esn_dict


def get_hyperparameters_for_model(model_name, hypers):
    """
    从 hypers 中提取指定模型的超参数。
    
    参数:
        model_name (str): 模型名称（对应 hypers 的键）。
        hypers (dict): 包含超参数的字典。

    返回:
        dict: 提取的超参数。
    """
    hyper = hypers[model_name]
    
    # 提取基本超参数
    params = {
        "units": hyper['units'],
        "lr": hyper['lr'],
        "sr": hyper['sr'],
        "rc_connectivity": hyper['rc_connectivity'],
        "noise_rc": hyper['noise_rc'],
        "noise_in": hyper['noise_in'],
        "seed": hyper['seed'],
        "input_connectivity": hyper['input_connectivity'],
        "input_scaling": hyper['input_scaling'],
        "use_raw_input": hyper.get('use_raw_input', False),  # 默认值 False
        "output_dim": hyper.get('output_dim', None),  # 默认值 None
        "deep": hyper.get('deep', False),  # 是否为 Deep ESN
        "deep_ip_list":hyper.get('deep_ip_list', None),
    }
    
    # 针对不同类型的模型添加额外参数
    if hyper.get('online', False):  # 如果是在线 ESN
        params["online"] = True
        params["alpha"] = hyper.get('alpha', 1e-6)  # 默认值 1e-6
    else:  # 如果不是在线 ESN
        params["online"] = False
        params["ridge"] = hyper['ridge']
        params["ip_reservoir"] = hyper.get('ip_reservoir', False)  # 默认值 False
    
    return params


def get_hyperparameters(hypers):
    """
    从 hypers 字典中提取超参数。

    参数:
        hypers (dict): 包含单个模型超参数的字典。

    返回:
        dict: 提取的超参数。
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
        "output_dim": hypers.get('output_dim', None),  # 默认值 None
        "use_raw_input": hypers.get('use_raw_input', False),  # 默认值 False
        "online": hypers.get('online', False),  # 默认值 False
        "alpha": hypers.get('alpha', 1e-6) if hypers.get('online', False) else None,  # 在线模式需要 alpha
        "ip_reservoir": hypers.get('ip_reservoir', False),  # 默认值 False
        "ridge": hypers.get('ridge', None) if not hypers.get('online', False) else None,  # 非在线模式需要 ridge
        "deep": hypers.get('deep', False),  # 是否为 Deep ESN，默认值 False
        "deep_ip_list":hypers.get('deep_ip_list', None),
    }
    return params


def get_esn_from_hypers(hypers):
    """
    根据超参数字典创建一个 ESN 模型，包括普通 ESN、在线 ESN、IP ESN 和 Deep ESN。

    参数:
        hypers (dict): 包含单个模型超参数的字典。

    返回:
        object: 创建的 ESN 模型。
    """
    # 提取超参数
    params = get_hyperparameters(hypers)
    
    # 根据 online、ip_reservoir 或 deep 决定创建哪种模型
    if params["deep"]:  # Deep ESN
        esn = Create_Deep_ESN(
            units=params["units"],
            lr=params["lr"],
            sr=params["sr"],
            rc_connectivity=params["rc_connectivity"],
            noise_rc=params["noise_rc"],
            noise_in=params["noise_in"],
            ridge=params["ridge"],
            input_connectivity=params["input_connectivity"],
            input_scaling=params["input_scaling"],
            output_dim=params["output_dim"],
            seed=params["seed"],
            deep_ip_list=params["deep_ip_list"],
        )
    elif params["online"]:  # 在线 ESN
        esn = Create_Online_ESN(
            units=params["units"],
            lr=params["lr"],
            sr=params["sr"],
            rc_connectivity=params["rc_connectivity"],
            noise_rc=params["noise_rc"],
            noise_in=params["noise_in"],
            input_connectivity=params["input_connectivity"],
            input_scaling=params["input_scaling"],
            output_dim=params["output_dim"],
            alpha=params["alpha"],  # 在线模式需要 alpha
            seed=params["seed"],
            use_raw_input=params["use_raw_input"]
        )
    elif params["ip_reservoir"]:  # IP ESN
        esn = Create_New_IPESN(
            units=params["units"],
            lr=params["lr"],
            sr=params["sr"],
            rc_connectivity=params["rc_connectivity"],
            noise_rc=params["noise_rc"],
            noise_in=params["noise_in"],
            input_connectivity=params["input_connectivity"],
            input_scaling=params["input_scaling"],
            output_dim=params["output_dim"],
            ridge=params["ridge"],  # IP 模式需要 ridge
            seed=params["seed"],
            use_raw_input=params["use_raw_input"],
        )
    else:  # 普通 ESN
        esn = Create_New_ESN(
            units=params["units"],
            lr=params["lr"],
            sr=params["sr"],
            rc_connectivity=params["rc_connectivity"],
            noise_rc=params["noise_rc"],
            noise_in=params["noise_in"],
            input_connectivity=params["input_connectivity"],
            input_scaling=params["input_scaling"],
            output_dim=params["output_dim"],
            ridge=params["ridge"],  # 普通模式需要 ridge
            seed=params["seed"],
            use_raw_input=params["use_raw_input"],
        )
    
    return esn

def RC_Forecast_Train_Test(TS, tl, steps=22, units=4000,sr=0.95,
                           rc_connectivity=0.14,
                           noise_rc=0.01,
                           noise_in=0.01,
                           ridge=6e-06,
                           use_raw_input=False,
                           esn=None,
                           seed=None):
    if esn is None:
        esn = Create_New_ESN(output_dim=TS.shape[1], 
                            units=units,
                            sr=sr, 
                            rc_connectivity=rc_connectivity,
                            noise_rc=noise_rc,
                            noise_in=noise_in,
                            ridge=ridge,
                            use_raw_input=use_raw_input,
                            seed=seed) 
    if not esn.fitted:
        Xtrain = TS[:tl-1,]
        Ytrain = TS[1:tl,]
        esn = esn.fit(Xtrain,Ytrain)
    Ytest = TS[tl:,]
    x = Ytest
    Ypred = np.zeros((x.shape[0],x.shape[1],steps))
    for j in range(steps):
        if j == 0:
            Ypred[:,:,j] = x
        else:
            x = esn.run(x)
            Ypred[:,:,j] = x
    return Ypred,Ytest,esn

def TPRC_Forecast_Train_Test(TS,tl,steps=22,dl=0,units=4000,sr=0.95,
                           rc_connectivity=0.14,
                           noise_rc=0.01,
                           noise_in=0.01,
                           ridge=6e-06,seed=None,
                           esn=None,
                           use_raw_input=False,
                           tp_omega=2*np.pi/12,
                           tp_order=2,
                           tp_bias=0):
    if esn is None:
        esn = Create_New_ESN(output_dim=TS.shape[1], 
                                    units=units,
                                    sr=sr, 
                                    rc_connectivity=rc_connectivity,
                                    noise_rc=noise_rc,
                                    noise_in=noise_in,
                                    ridge=ridge,
                                    use_raw_input=use_raw_input,
                                    seed=seed)
    TP = get_RCTP(TS,steps=steps,omega=tp_omega,order=tp_order,bias=tp_bias)    
    TS_TP = np.hstack((TS,TP[:-steps]))
    Xtrain = TS_TP[:tl-1,]
    Ytrain = TS[1:tl,]
    esn = esn.fit(Xtrain,Ytrain,warmup=dl)
    Ytest = TS_TP[tl:,]
    x = Ytest
    Ypred = np.zeros((x.shape[0],x.shape[1],steps))
    for j in range(steps):
        if j == 0:
            Ypred[:,:,j] = x
        else:
            ts_prediction = esn.run(x)
            x = np.hstack((ts_prediction,TP[tl+j:-steps+j]))
            Ypred[:,:,j] = x
    return Ypred,Ytest,esn

def TPRC_ReForecast_Train_Test(TS,tl=None, steps=22,dl=0, units=4000,sr=0.95,
                           rc_connectivity=0.14,
                           noise_rc=0.01,
                           noise_in=0.01,
                           ridge=6e-06,seed=None,
                           esn=None,
                           use_raw_input=False,
                           tp_omega=2*np.pi/12,
                           tp_order=3,
                           tp_bias=0):
    if tl is None:
        tl = TS.shape[0]
    if esn is None:
        esn = Create_New_ESN(output_dim=TS.shape[1], 
                                    units=units,
                                    sr=sr, 
                                    rc_connectivity=rc_connectivity,
                                    noise_rc=noise_rc,
                                    noise_in=noise_in,
                                    ridge=ridge,
                                    use_raw_input=use_raw_input,
                                    seed=seed)
    TP = get_RCTP(TS,steps=steps,omega=tp_omega,order=tp_order,bias=tp_bias)    
    TS_TP = np.hstack((TS,TP[:-steps]))
    Xtrain = TS_TP[:tl-1,]
    Ytrain = TS[1:tl,]
    esn = esn.fit(Xtrain,Ytrain)
    Ytest = TS_TP[:tl,]
    x = Ytest
    Ypred = np.zeros((x.shape[0],x.shape[1],steps))
    for j in range(steps):
        if j == 0:
            Ypred[:,:,j] = x
        else:
            ts_prediction = esn.run(x)
            x = np.hstack((ts_prediction,TP[j:-steps+j]))
            Ypred[:,:,j] = x
    return Ypred,Ytest,esn

def TPRC_In_Out_Forecast_Train_Test(TS,tl=None, steps=22,dl=0, units=4000,sr=0.95,
                           rc_connectivity=0.14,
                           noise_rc=0.01,
                           noise_in=0.01,
                           ridge=6e-06,seed=None,
                           esn=None,
                           use_raw_input=False,
                           tp_omega=2*np.pi/12,
                           tp_order=3,
                           tp_bias=0):
    if tl is None:
        tl = TS.shape[0]
    if esn is None:
        esn = Create_New_ESN(output_dim=TS.shape[1], 
                                    units=units,
                                    sr=sr, 
                                    rc_connectivity=rc_connectivity,
                                    noise_rc=noise_rc,
                                    noise_in=noise_in,
                                    ridge=ridge,
                                    use_raw_input=use_raw_input,
                                    seed=seed)
    TP = get_RCTP(TS,steps=steps,omega=tp_omega,order=tp_order,bias=tp_bias)    
    TS_TP = np.hstack((TS,TP[:-steps]))
    Xtrain = TS_TP[:tl-1,]
    Ytrain = TS[1:tl,]
    esn = esn.fit(Xtrain,Ytrain)
    Ytest = TS_TP[:,]
    x = Ytest
    Ypred = np.zeros((x.shape[0],x.shape[1],steps))
    for j in range(steps):
        if j == 0:
            Ypred[:,:,j] = x
        else:
            ts_prediction = esn.run(x)
            x = np.hstack((ts_prediction,TP[j:-steps+j]))
            Ypred[:,:,j] = x
    return Ypred,Ytest,esn


def RC_ReForecast_Train_Test(TS, tl=None, steps=22, units=4000,sr=0.95,
                           rc_connectivity=0.14,
                           noise_rc=0.01,
                           noise_in=0.01,
                           ridge=6e-06,esn=None,esnTrained=False,seed=None):
    if tl is None:
        tl = TS.shape[0]
    if esn is None:
        esn = Create_New_ESN(output_dim=TS.shape[1], 
                                    units=units,
                                    sr=sr, 
                                    rc_connectivity=rc_connectivity,
                                    noise_rc=noise_rc,
                                    noise_in=noise_in,
                                    ridge=ridge,seed=seed) 
    if not esnTrained:
        Xtrain = TS[:tl-1,]
        Ytrain = TS[1:tl,]
        esn = esn.fit(Xtrain,Ytrain)
    Ytest = TS[:tl,]
    x = Ytest
    Ypred = np.zeros((x.shape[0],x.shape[1],steps))
    for j in range(steps):
        if j == 0:
            Ypred[:,:,j] = x
        else:
            x = esn.run(x)
            Ypred[:,:,j] = x
    return Ypred,Ytest,esn

def ESN_Generate(Ytest,esn,steps=22):
    x = Ytest
    Ypred = np.zeros((x.shape[0],x.shape[1],steps))
    for j in range(steps):
        if j == 0:
            Ypred[:,:,j] = x
        else:
            x = esn.run(x)
            Ypred[:,:,j] = x
    return Ypred

def ndforecast_leadskill(Ypred,Ytest,lead_time=12,showdim=0,ismv3=True,wl=12):
    if ismv3:
        observation = pd.Series(Ytest[lead_time+wl:,showdim]).rolling(window=3,min_periods=1).mean().values
        prediction = pd.Series(Ypred[wl:-lead_time,showdim,lead_time]).rolling(window=3,min_periods=1).mean().values
    else:
        observation = Ytest[lead_time+wl:,showdim]
        prediction = Ypred[wl:-lead_time,showdim,lead_time]
    R = np.corrcoef(observation,prediction)[0,1]
    rmse = np.sqrt(np.mean((observation-prediction)**2))
    return R,rmse

def ndforecast_skill(Ypred,Ytest,showdim=0,ismv3=True,wl=12,plot=False):
    """
    Calculate the multi-dimensional forecast skill, including the correlation coefficient (R) and the root mean square error (RMSE).

    Parameters:
    Ypred (ndarray): The array of predicted values.
    Ytest (ndarray): The array of test values.
    showdim (int, optional): The dimension to display. Default is 0.
    ismv3 (bool, optional): Whether to use the mv3 mode. Default is True.
    wl (int, optional): The window length. Default is 12.

    Returns:
    tuple: A tuple containing the correlation coefficient and the root mean square error.
    """
    # Get the number of time steps in the prediction
    steps = Ypred.shape[2]
    # Initialize the arrays for the correlation coefficient and the root mean square error
    R = np.zeros(steps)
    rmse = np.zeros(steps)
    # Set the first correlation coefficient to 1 and the first root mean square error to 0
    R[0] = 1
    rmse[0] = 0
    # Iterate over each time step in the prediction
    for lead_time in range(1,steps):
        # If in mv3 mode
        if ismv3:
            # Calculate the rolling window average as the observed and predicted values
            observation = pd.Series(Ytest[lead_time+wl:,showdim]).rolling(window=3,min_periods=1).mean().values
            prediction = pd.Series(Ypred[wl:-lead_time,showdim,lead_time]).rolling(window=3,min_periods=1).mean().values
        # If not in mv3 mode
        else:
            # Use the raw observed and predicted values
            observation = Ytest[lead_time+wl:,showdim]
            prediction = Ypred[wl:-lead_time,showdim,lead_time]
        # Calculate the correlation coefficient and the root mean square error
        R[lead_time] = np.corrcoef(observation,prediction)[0,1]
        rmse[lead_time] = np.sqrt(np.mean((observation-prediction)**2))

    # If plot is True, plot the correlation coefficient and the root mean square error
    if plot:
        plt.figure(figsize=(10, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(range(steps), R, marker='o', color='orangered')
        plt.title('Correlation Coefficient (R) vs Lead Time')
        plt.xlabel('Lead Time')
        plt.ylabel('R')
        
        plt.subplot(2, 1, 2)
        plt.plot(range(steps), rmse, marker='o', color='orangered')
        plt.title('Root Mean Square Error (RMSE) vs Lead Time')
        plt.xlabel('Lead Time')
        plt.ylabel('RMSE')
        
        plt.tight_layout()
        plt.show()
    # Return the correlation coefficient and the root mean square error
    return R,rmse
    
def simforecast_skill(Ypred_sim_mean,Ytest,showdim=0,ismv3=True,wl=12):
    steps = Ypred_sim_mean.shape[1]
    R = np.zeros(steps)
    rmse = np.zeros(steps)
    R[0] = 1
    rmse[0] = 0
    for lead_time in range(1,steps):
        if ismv3:
            observation = pd.Series(Ytest[lead_time+wl:,showdim]).rolling(window=3,min_periods=1).mean().values
            prediction = pd.Series(Ypred_sim_mean[wl:-lead_time,lead_time]).rolling(window=3,min_periods=1).mean().values
        else:
            observation = Ytest[lead_time+wl:,showdim]
            prediction = Ypred_sim_mean[wl:-lead_time,lead_time]
        R[lead_time] = np.corrcoef(observation,prediction)[0,1]
        rmse[lead_time] = np.sqrt(np.mean((observation-prediction)**2))
    return R,rmse

def plot_ndforecast_result(Ypred,Ytest,time,lead_time=12,tl=0,showdim=0,ismv3=True,wl=12):
    if ismv3:
        observation = pd.Series(Ytest[lead_time+wl:,showdim]).rolling(window=3,min_periods=1).mean().values
        prediction = pd.Series(Ypred[wl:-lead_time,showdim,lead_time]).rolling(window=3,min_periods=1).mean().values
    else:
        observation = Ytest[lead_time+wl:,showdim]
        prediction = Ypred[wl:-lead_time,showdim,lead_time]
    R1 = np.corrcoef(observation,prediction)[0,1]
    rmse1 = np.sqrt(np.mean((observation-prediction)**2))
    plt.figure(figsize=(12, 4))
    plt.plot(time[tl+lead_time+wl:],observation,label = 'Observation', color='black')
    plt.plot(time[tl+lead_time+wl:],prediction,label = 
            f'RC predict,R={R1:.2},rmse={rmse1:.2f}',color='r',linestyle = '-')
    plt.ylabel('SST')
    plt.xlabel('Year')
    plt.title(f'Nino3.4 Index Prediction using RC by lead time = {lead_time}')
    plt.legend()
    plt.show()

def pack_TS_anualTP(TS,omega=2*np.pi/12,order=3,bias=0):
    t = np.linspace(0, TS.shape[0]-1, TS.shape[0], dtype=int)
    for i in range(1,order+1):
        sint = (bias+np.sin(i*omega*t)).reshape(-1,1)
        cost = (bias+np.cos(i*omega*t)).reshape(-1,1)
        if i == 1:
            TP = np.hstack((sint, cost))
        else:
            TP = np.hstack((TP, sint, cost))
    TS_TP = np.hstack((TS, TP))
    return TS_TP

def get_RCTP(TS,steps=22,omega=2*np.pi/12,order=3,bias=0):
    t = np.linspace(0, TS.shape[0]-1+steps, TS.shape[0]+steps, dtype=int)
    for i in range(1,order+1):
        sint = (bias+np.sin(i*omega*t)).reshape(-1,1)
        cost = (bias+np.cos(i*omega*t)).reshape(-1,1)
        if i == 1:
            TP = np.hstack((sint, cost))
        else:
            TP = np.hstack((TP, sint, cost))
    return TP

def top_n_frequencies(data, max_freq_count=10, min_interval=0, fs=1,isNormalize=True):
    # Perform FFT
    n = len(data)
    freqs = fftfreq(n, d=1/fs)
    power_spectrum = np.abs(fft(data))**2
    
    # Take only positive frequencies
    positive_freqs = freqs > 0
    freqs = freqs[positive_freqs]
    power_spectrum = power_spectrum[positive_freqs]
    
    # Sort frequencies by power
    sorted_indices = np.argsort(power_spectrum)[::-1]
    sorted_freqs = freqs[sorted_indices]
    sorted_amp = np.sqrt(power_spectrum[sorted_indices])
    
    # Filter to get top frequencies with specified interval
    selected_freqs = []
    selected_amplitudes = []
    for f, amp in zip(sorted_freqs, sorted_amp):
        if len(selected_freqs) >= max_freq_count:
            break
        if selected_freqs == [] or all(abs(f - sf) > min_interval for sf in selected_freqs):
            selected_freqs.append(f)
            selected_amplitudes.append(amp)
    if isNormalize:
        selected_amplitudes = np.array(selected_amplitudes)
        selected_amplitudes = selected_amplitudes/np.sqrt(np.sum(selected_amplitudes**2))
    return np.array(selected_freqs), selected_amplitudes

def pack_TS_givenTP(TS,ref_data=None,f_number=10,order=1):
    if ref_data is None:
        ref_data = TS[:,0]
    t = np.linspace(0, TS.shape[0]-1, TS.shape[0], dtype=int)
    selected_freqs, selected_amplitudes = top_n_frequencies(ref_data,f_number,min_interval=0,fs=1)
    for i,f in enumerate(selected_freqs):
        omega = 2*np.pi*f
        for j in range(1,order+1):
            sint = selected_amplitudes[i]*(np.sin(j*omega*t)+1).reshape(-1,1)
            cost = selected_amplitudes[i]*(np.cos(j*omega*t)+1).reshape(-1,1)
            if i==0 and j == 1:
                TP = np.hstack((sint, cost))
            else:
                TP = np.hstack((TP,sint, cost))        
    TS_TP = np.hstack((TS, TP))
    return TS_TP

def generate_scaled_brownian_noise(length, target_std_dev):
    steps = np.random.normal(loc=0, scale=1, size=length)
    brownian_noise = np.cumsum(steps)
    # 缩放噪声使其标准差接近目标值
    brownian_noise *= (target_std_dev / np.std(brownian_noise))
    return brownian_noise

def generate_scaled_white_noise(length, target_std_dev):
    return np.random.normal(loc=0, scale=target_std_dev, size=length)

def stochastic_perturb_ESN(data,steps,esn,perturb_dim=10,nmembers=100,isRedNoise=False):
    perturbation = np.zeros((data.shape[0],perturb_dim))
    red_noise = np.zeros((data.shape[0],perturb_dim))
    white_noise = np.zeros((data.shape[0],perturb_dim))
    Y_pred_sim = np.zeros((nmembers,data.shape[0],steps))
    for i in tqdm(range(0,nmembers)):
        # 参数
        length = data.shape[0]  # 噪声序列的长度
        target_std_dev = 0.02  # 目标标准偏差

        # 生成并缩放布朗噪声
        for j in range(0,perturb_dim ):
            red_noise[:,j] = generate_scaled_brownian_noise(length, target_std_dev)

        # 生成高斯白噪声
        for j in range(0,perturb_dim ):
            white_noise[:,j] = generate_scaled_white_noise(length, target_std_dev)
            
        if isRedNoise:
            perturbation = red_noise
        else:
            perturbation = white_noise
            
        X_test = np.column_stack([data[:,:perturb_dim]+perturbation, data[:,perturb_dim:]])
        x = X_test
        for k in range(0,steps):
            x = esn.run(x)
            Y_pred_sim[i,:,k] = x[:,0]
    return Y_pred_sim

def stochastic_noise_ESN(TS,steps,noise_esn,nmembers=100,concern_dim=0):
    """
    Perform stochastic simulations using a noise-driven Echo State Network (ESN).

    Parameters:
    - TS (ndarray): Input time series data.
    - steps (int): Number of time steps to predict.
    - noise_esn (object): Noise-driven ESN model.
    - nmembers (int, optional): Number of simulation members. Default is 100.
    - concern_dim (int, optional): Dimension of concern. Default is 0.

    Returns:
    - Ypred_sim (ndarray): Array containing the results of the stochastic simulations, with shape (nmembers, TS.shape[0], steps).
    """
    Ypred_sim = np.zeros((nmembers,TS.shape[0],steps))
    for i in tqdm(range(0,nmembers)):
        x  = TS
        for k in range(0,steps):
            x = noise_esn.run(x)
            Ypred_sim[i,:,k] = x[:,concern_dim]
    return Ypred_sim

def stochastic_synthesize_ESN_from_hypers(TS,hypers,steps,tl,perturb_dim=1,nmembers=100,concern_dim=0,target_std=0.02,isReforecast=False,noiseType='zero'):
    """
    使用一系列相同超参数的回声状态网络(系综ESN,synthesize_ESN)进行多次重新训练和预测。考虑随机数种子的影响。
    并在预测阶段进行初值扰动。
    参数：
    - TS (ndarray): 输入时间序列数据。包含训练阶段。
    - steps (int): 预测的时间步数。
    - hypers (dict): 超参数字典。
    - tl (int): 训练时间长度。
    - perturb_dim (int, 可选): 扰动前perturb_dim维。默认为1。
    - nmembers (int, 可选): 模拟的成员数量。默认为100。
    - concern_dim (int, 可选): 关注的维度。默认为0。
    - isReforecast (bool, 可选): 是否进行重新预测。默认为False。
    - noiseType (str, 可选): 噪声类型，可选值为 'red' 或 'white'。默认为 'zero'。

    返回：
    - Ypred_sim (ndarray): 包含随机合成预测结果的数组，形状为 (nmembers, TS[tl:].shape[0], steps)。
    """
    Ypred_sim = np.zeros((nmembers,TS[tl:].shape[0],steps))
    Xtrain = TS[:tl-1,]
    Ytrain = TS[1:tl,]
    if isReforecast:
        Ytest = TS[:tl,]
    else:
        Ytest = TS[tl:,]
    noise_length = Ytest.shape[0]  # 噪声序列的长度
    red_noise = np.zeros((noise_length,perturb_dim))
    white_noise = np.zeros((noise_length,perturb_dim))
    zero_noise = np.zeros((noise_length,perturb_dim))

    for i in tqdm(range(0,nmembers),desc='Stochastic Simulating'):
        
        target_std = target_std  # 目标标准偏差

        # 生成并缩放布朗噪声
        for j in range(0,perturb_dim):
            red_noise[:,j] = generate_scaled_brownian_noise(noise_length, target_std)

        # 生成高斯白噪声
        for j in range(0,perturb_dim ):
            white_noise[:,j] = generate_scaled_white_noise(noise_length, target_std)
            
        if noiseType == 'red':
            perturbation = red_noise
        elif noiseType == 'white':
            perturbation = white_noise
        elif noiseType == 'zero':
            perturbation = zero_noise
        else:
            perturbation = zero_noise
            
        X_test = np.column_stack([Ytest[:,:perturb_dim]+perturbation, Ytest[:,perturb_dim:]])
        x = X_test
        esn = get_esn_from_hypers(hypers)
        esn = esn.reset()
        esn = esn.fit(Xtrain,Ytrain)
        for j in range(steps):
            if j == 0:
                Ypred_sim[i,:,j] = x[:,concern_dim]
            else:
                x = esn.run(x)
                Ypred_sim[i,:,j] = x[:,concern_dim]
    return Ypred_sim,Ytest


def TPRC_Forecast_Train_Test_Ensemble(
    TS,
    tl,
    wl=0,
    steps=22,
    dl=0,
    hypers=None,
    nmember=10,
    isReforecast=False,
    noise_ini=0.0  
):
    """
    使用集合预报的方式进行时间序列预测，每个集合成员使用一个 ESN。
    """
    # 获取时间周期特征 (TP)
    tp_omega=2 * np.pi / 12
    tp_order=2
    tp_bias=0
    TP = get_RCTP(TS, steps=steps, omega=tp_omega, order=tp_order, bias=tp_bias)
    TS_TP = np.hstack((TS, TP[:-steps]))  # 将时间序列与时间周期特征结合
    Xtrain = TS_TP[:tl - 1,]
    Ytrain = TS[1:tl,]

    # 记录所有成员的预测结果
    if isReforecast:
        ensemble_predictions = np.zeros((nmember, TS_TP[dl:tl].shape[0], TS_TP[dl:tl].shape[1], steps))
    else:
        ensemble_predictions = np.zeros((nmember, TS_TP[tl+wl:].shape[0], TS_TP[tl+wl:].shape[1], steps))

    # 遍历集合成员，训练多个 ESN 并生成预测
    for m in tqdm(range(nmember)):
        # 初始化一个新的 ESN
        member_esn = get_esn_from_hypers(hypers)

        # 训练 ESN
        member_esn = member_esn.fit(Xtrain, Ytrain, warmup=dl)

        # 获取测试数据
        if isReforecast:
            Ytest = TS_TP[dl:tl,]
        else:
            Ytest = TS_TP[tl+wl:,]
        x = Ytest
        Ypred = np.zeros((x.shape[0], x.shape[1], steps))

        # 生成多步预测
        for j in range(steps):
            if j == 0:
                Ypred[:, :, j] = x  
            else:
                # 添加噪声到初始状态
                noise = np.random.normal(0, noise_ini, size=x.shape)
                x += noise 
                ts_prediction = member_esn.run(x)  # 使用 ESN 进行预测
                if isReforecast:
                    x = np.hstack((ts_prediction, TP[dl + j : tl + dl + j]))  # 更新输入
                else:
                    x = np.hstack((ts_prediction, TP[tl + wl + j : -steps + j]))  # 更新输入
                Ypred[:, :, j] = x

        # 保存当前成员的预测结果
        ensemble_predictions[m] = Ypred

    # 计算集合成员的平均预测作为最终预测结果
    Ypred_mean = np.mean(ensemble_predictions, axis=0)

    # 返回集合平均预测结果、测试数据、所有成员的预测结果
    return Ypred_mean, Ytest, ensemble_predictions

def RC_Forecast_Train_Test_Ensemble(
    TS,
    tl,
    steps=22,
    dl=0,
    hypers=None,
    nmember=10,  # 集合成员数
    isReforecast=False,
    reSetNoise=False,
    noise_ini=0.0
):
    """
    使用集合预报的方式进行时间序列预测，每个集合成员使用一个 ESN。
    """
    
    Xtrain = TS[:tl - 1,]
    Ytrain = TS[1:tl,]
    ESNlist = []
    # 记录所有成员的预测结果
    if isReforecast:
        ensemble_predictions = np.zeros((nmember, TS[:tl].shape[0], TS[:tl].shape[1], steps))

    else:
        ensemble_predictions = np.zeros((nmember, TS[tl:].shape[0], TS[tl:].shape[1], steps))

    # 遍历集合成员，训练多个 ESN 并生成预测
    for m in range(nmember):
        # 初始化一个新的 ESN
        member_esn = get_esn_from_hypers(hypers)

        # 训练 ESN
        member_esn = member_esn.fit(Xtrain, Ytrain, warmup=dl)
        ESNlist.append(member_esn)
        
        # 获取测试数据
        if isReforecast:
            Ytest = TS[:tl,]
        else:
            Ytest = TS[tl:,]
        x = Ytest
        Ypred = np.zeros((x.shape[0], x.shape[1], steps))

        # 生成多步预测
        for j in range(steps):
            if j == 0:
                Ypred[:, :, j] = x  # 初始状态
            else:
                # 添加噪声到初始状态
                noise = np.random.normal(0, noise_ini, size=x.shape)
                x += noise 
                ts_prediction = member_esn.run(x)  # 使用 ESN 进行预测
                if reSetNoise:
                    member_esn.noise_rc = 0
                x = ts_prediction  # 更新输入
                Ypred[:, :, j] = x

        # 保存当前成员的预测结果
        ensemble_predictions[m] = Ypred

    # 计算集合成员的平均预测作为最终预测结果
    Ypred_mean = np.mean(ensemble_predictions, axis=0)

    # 返回集合平均预测结果、测试数据、所有成员的预测结果
    return Ypred_mean, ESNlist, ensemble_predictions

def stochastic_allreturn_synthesize_ESN_from_hypers(TS,hypers,steps,tl,perturb_dim=1,nmembers=100,target_std=0.02,isReforecast=False,noiseType='zero'):
    """
    使用一系列相同超参数的回声状态网络(系综ESN,synthesize_ESN)进行多次重新训练和预测。考虑随机数种子的影响。
    并在预测阶段进行初值扰动。
    参数：
    - TS (ndarray): 输入时间序列数据。包含训练阶段。
    - steps (int): 预测的时间步数。
    - hypers (dict): 超参数字典。
    - tl (int): 训练时间长度。
    - perturb_dim (int, 可选): 扰动前perturb_dim维。默认为1。
    - nmembers (int, 可选): 模拟的成员数量。默认为100。
    - isReforecast (bool, 可选): 是否进行重新预测。默认为False。
    - noiseType (str, 可选): 噪声类型，可选值为 'red' 或 'white'。默认为 'zero'。

    返回：
    - Ypred_sim (ndarray): 包含随机合成预测结果的数组，形状为 (nmembers, TS[tl:].shape[0], steps)。
    """
    Ypred_sim = np.zeros((nmembers,TS[tl:].shape[0],perturb_dim,steps))
    Xtrain = TS[:tl-1,]
    Ytrain = TS[1:tl,]
    if isReforecast:
        Ytest = TS[:tl,]
    else:
        Ytest = TS[tl:,]
    noise_length = Ytest.shape[0]  # 噪声序列的长度
    red_noise = np.zeros((noise_length,perturb_dim))
    white_noise = np.zeros((noise_length,perturb_dim))
    zero_noise = np.zeros((noise_length,perturb_dim))

    for i in tqdm(range(0,nmembers),desc='Stochastic Simulating'):
        
        target_std = target_std  # 目标标准偏差

        # 生成并缩放布朗噪声
        for j in range(0,perturb_dim):
            red_noise[:,j] = generate_scaled_brownian_noise(noise_length, target_std)

        # 生成高斯白噪声
        for j in range(0,perturb_dim ):
            white_noise[:,j] = generate_scaled_white_noise(noise_length, target_std)
            
        if noiseType == 'red':
            perturbation = red_noise
        elif noiseType == 'white':
            perturbation = white_noise
        elif noiseType == 'zero':
            perturbation = zero_noise
        else:
            perturbation = zero_noise
            
        X_test = np.column_stack([Ytest[:,:perturb_dim]+perturbation, Ytest[:,perturb_dim:]])
        x = X_test
        esn = get_esn_from_hypers(hypers)
        esn = esn.reset()
        esn = esn.fit(Xtrain,Ytrain)
        for j in range(steps):
            if j == 0:
                Ypred_sim[i,:,:,j] = x[:,:perturb_dim]
            else:
                x = esn.run(x)
                Ypred_sim[i,:,:,j] = x[:,:perturb_dim]
    return Ypred_sim,Ytest

def stochastic_synthesize_ESN(TS,steps,esn,tl,perturb_dim=1,nmembers=100,concern_dim=0,target_std=0.02,isReforecast=False,noiseType='zero'):
    """
    使用一系列相同超参数的回声状态网络(系综ESN,synthesize_ESN)进行多次重新训练和预测。
    并在预测阶段进行初值扰动。
    参数：
    - TS (ndarray): 输入时间序列数据。包含训练阶段。
    - steps (int): 预测的时间步数。
    - esn (object): 回声状态网络模型。
    - tl (int): 训练时间长度。
    - perturb_dim (int, 可选): 扰动前perturb_dim维。默认为1。
    - nmembers (int, 可选): 模拟的成员数量。默认为100。
    - concern_dim (int, 可选): 关注的维度。默认为0。
    - isReforecast (bool, 可选): 是否进行重新预测。默认为False。
    - noiseType (str, 可选): 噪声类型，可选值为 'red' 或 'white'。默认为 'zero'。

    返回：
    - Ypred_sim (ndarray): 包含随机合成预测结果的数组，形状为 (nmembers, TS[tl:].shape[0], steps)。
    """
    Ypred_sim = np.zeros((nmembers,TS[tl:].shape[0],steps))
    Xtrain = TS[:tl-1,]
    Ytrain = TS[1:tl,]
    if isReforecast:
        Ytest = TS[:tl,]
    else:
        Ytest = TS[tl:,]
    noise_length = Ytest.shape[0]  # 噪声序列的长度
    red_noise = np.zeros((noise_length,perturb_dim))
    white_noise = np.zeros((noise_length,perturb_dim))
    for i in tqdm(range(0,nmembers),desc='Stochastic Simulating'):
        
        target_std = target_std  # 目标标准偏差

        # 生成并缩放布朗噪声
        for j in range(0,perturb_dim):
            red_noise[:,j] = generate_scaled_brownian_noise(noise_length, target_std)

        # 生成高斯白噪声
        for j in range(0,perturb_dim ):
            white_noise[:,j] = generate_scaled_white_noise(noise_length, target_std)
            
        if noiseType == 'red':
            perturbation = red_noise
        elif noiseType == 'white':
            perturbation = white_noise
        elif noiseType == 'zero':
            perturbation = np.zeros((noise_length,perturb_dim))
        else:
            perturbation = np.zeros((noise_length,perturb_dim))
            
        X_test = np.column_stack([Ytest[:,:perturb_dim]+perturbation, Ytest[:,perturb_dim:]])
        x = X_test
        esn = esn.reset()
        esn = esn.fit(Xtrain,Ytrain)
        for j in range(steps):
            if j == 0:
                Ypred_sim[i,:,j] = x[:,concern_dim]
            else:
                x = esn.run(x)
                Ypred_sim[i,:,j] = x[:,concern_dim]
    return Ypred_sim

def get_sim_mean(Ypred_sim, n=10):
    """
    Compute the mean of a randomly selected subset of simulations along the first dimension.

    Parameters:
    - Y_pred_sim: np.ndarray
        The simulation results array with shape (num_simulations, ...).
    - n: int
        The number of random simulations to sample for computing the mean.

    Returns:
    - Y_pred_sim_mean: np.ndarray
        The mean of the randomly selected subset along the first dimension.
    """
    # Get the total number of simulations (size of the first dimension)
    num_simulations = Ypred_sim.shape[0]

    # Check if n is larger than the total number of simulations
    if n > num_simulations:
        raise ValueError(f"`n` ({n}) cannot be larger than the number of simulations ({num_simulations}).")

    # Randomly select n indices from the first dimension without replacement
    random_indices = np.random.choice(num_simulations, size=n, replace=False)

    # Select the corresponding simulations and compute the mean along the first dimension
    Y_pred_sim_subset = Ypred_sim[random_indices]
    Y_pred_sim_mean = np.mean(Y_pred_sim_subset, axis=0)

    return Y_pred_sim_mean

def SSA(sequence,windowlen):
    serieslen = len(np.array(sequence))
    if not isinstance(serieslen,int):
        raise ValueError(f"serieslen must be an integer: {serieslen}")
    
    #1.嵌入
    K = serieslen-windowlen+1
    X = np.zeros((windowlen,K))
    for i in range(windowlen):
        X[i,:] = sequence[i:i+K]
    #2.SVD分解
    U,sigma,VT = np.linalg.svd(X,full_matrices=False)
    ZT = np.zeros((VT.shape))
    for n in range(len(sigma)):
        ZT[n,:] = sigma[n]*VT[n,:]

    #3.对每个Ui*ZTi所得到的Xi进行对角平均获得重构子序列，一般我们假设L<K
    PC = np.zeros((windowlen,serieslen))
    L = windowlen
    for num in range(U.shape[1]):
        Xi = np.outer(U[:,num],ZT[num,:])
        for k in range(serieslen):
            if not isinstance(k,int):
                raise ValueError(f"k must be an integer: {k}")
            sum=0
            if(k<=L-1):
                for p in range(k+1):
                    sum+=Xi[p,k-p]
                PC[num,k] = sum/(k+1)
            elif(k<=K-1):
                for p in range(L):
                    sum+=Xi[p,k-p]
                PC[num,k] = sum/L
            elif(k<=serieslen-1):
                for p in range(k-K+1,serieslen-K+1):
                    sum+=Xi[p,k-p]
                PC[num,k] = sum/(serieslen-k)
    return PC

def butterworth_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def reconstruct_Ypred_skill(Ypred, Ytest, given_cutoff=None, concern_dim=0, cutofflist=np.linspace(0.06, 0.20, 80)):
    """
    重构预测数据并计算相关的技能指标。

    参数:
    - Ypred: 预测数据，形状为 (样本数, 维度, 时间步长)。
    - Ytest: 测试数据，形状为 (时间步长, 维度)。
    - given_cutoff: 给定的截止频率，如果为 None，则从 cutofflist 中选择。
    - concern_dim: 关注的维度，默认为 0。
    - cutofflist: 截止频率列表，默认为从 0.06 到 0.20 的 80 个值。

    返回:
    - reconstructed_Ypred: 重构后的预测数据。
    - Rf: 相关系数。
    - rmsef: 均方根误差。
    - sel_cutoff: 选择的截止频率。
    """
    steps = Ypred.shape[2]
    reconstructed_Ypred = np.zeros((Ypred.shape[0], steps))
    Rf = np.zeros((steps))
    rmsef = np.zeros((steps))
    sel_cutoff = np.zeros((steps))

    if len(cutofflist) == 0 and given_cutoff is None:
        raise ValueError("cutofflist cannot be empty when given_cutoff is None.")

    for lead_time in range(steps):
        if lead_time == 0:
            Rf[lead_time] = 1
            rmsef[lead_time] = 0
        else:
            if given_cutoff is None:
                Rf[lead_time] = -1
                rmsef[lead_time] = np.inf
                for cutoff in cutofflist:
                    if not (0 < cutoff < 0.5):  # Ensure cutoff is valid
                        continue
                    reconstructed_Ypred[:, lead_time] = butterworth_filter(
                        Ypred[:, concern_dim, lead_time], cutoff=cutoff, fs=1
                    )
                    y_test = butterworth_filter(Ytest[lead_time:, concern_dim], cutoff=cutoff, fs=1)
                    _Rf = np.corrcoef(y_test, reconstructed_Ypred[:-lead_time, lead_time])[0, 1]
                    if _Rf > Rf[lead_time]:
                        Rf[lead_time] = _Rf
                        sel_cutoff[lead_time] = cutoff
                    _rmsef = rpy.observables.rmse(y_test, reconstructed_Ypred[:-lead_time, lead_time])
                    if _rmsef < rmsef[lead_time]:
                        rmsef[lead_time] = _rmsef
                if sel_cutoff[lead_time] > 0:
                    reconstructed_Ypred[:, lead_time] = butterworth_filter(
                        Ypred[:, concern_dim, lead_time], cutoff=sel_cutoff[lead_time], fs=1
                    )
            else:
                reconstructed_Ypred[:, lead_time] = butterworth_filter(
                    Ypred[:, concern_dim, lead_time], cutoff=given_cutoff, fs=1
                )
                y_test = butterworth_filter(Ytest[lead_time:, concern_dim], cutoff=given_cutoff, fs=1)
                Rf[lead_time] = np.corrcoef(y_test, reconstructed_Ypred[:-lead_time, lead_time])[0, 1]
                rmsef[lead_time] = rpy.observables.rmse(y_test, reconstructed_Ypred[:-lead_time, lead_time])

    if given_cutoff is None and (np.any(sel_cutoff == 0) or np.any(Rf < -1)):
        raise ValueError("Failed to compute valid cutoff or correlation for one or more steps.")

    if given_cutoff is None:
        return reconstructed_Ypred, Rf, rmsef, sel_cutoff
    else:
        return reconstructed_Ypred, Rf, rmsef

# Generate a datetime array for every year and month
def generate_datetime_array(years):
    """
    Generate an array of datetime objects, where each element represents the first day of each month in the specified years.

    Args:
        years (list): A list of integers representing the years.

    Returns:
        list: An array of datetime objects.
    """
    return [datetime(int(year), month, 1) for year in years for month in range(1, 13)]

def plot_stochastic_result(Y_pred_sim,Ytest,time,lead_time=12,showdim=0,title='ensumble forecast time series with 12 months lead'):
    nmembers = Y_pred_sim.shape[0]
    Y_pred_sim_mean = np.mean(Y_pred_sim, axis=0)
    Y_pred_sim_mean_mv3 = pd.Series(Y_pred_sim_mean[:-lead_time,showdim,lead_time]).rolling(window=3,min_periods=1).mean().values
    y_test = pd.Series(Ytest[lead_time:,showdim]).rolling(window=3,min_periods=1).mean().values
    Y_Sim_std = np.std(Y_pred_sim, axis=0)

    plt.figure(figsize=(12, 4))  
    plt.plot(time[lead_time:],Y_pred_sim_mean_mv3,
            label = f'average,R={np.corrcoef(Y_pred_sim_mean_mv3,y_test)[0,1]:.2},rmse={rpy.observables.rmse(Y_pred_sim_mean_mv3,y_test):.2}',
            linestyle='--',color='orangered')

    plt.fill_between(time[lead_time:], Y_pred_sim_mean_mv3 - Y_Sim_std[:-lead_time,lead_time], Y_pred_sim_mean_mv3 + Y_Sim_std[:-lead_time,lead_time], alpha=0.5, label='simulations std' ,color='orangered')  # 绘制标准差区域  
    plt.plot(time[lead_time:],y_test,label = 'Observation', color='black')

    plt.title(title)
    # Customize legend
    ax = plt.gca()
    legend = ax.legend()
    for text in legend.get_texts():
        text.set_color("blue")  # Change the color
        text.set_fontweight("bold")  # Make it bold
    plt.show()

def cal_frequency_spectrum(signal, sampling_rate=1, title="Frequency Spectrum",norm_amp=True,plotFig=False,
                            isYearCoord=False,useSoothed=False,smooth_window=20,color='royalblue'):
    """
    计算信号的频谱，并将横坐标转换为周期（以年为单位）。

    参数：
    - signal: np.ndarray
        输入的时间序列信号。
    - sampling_rate: float, optional (default=1)
        采样率（单位：采样次数/月），默认每月采样一次。
    - title: str, optional (default="Frequency Spectrum")
        图表标题。
    - plotFig: bool, optional (default=False)
        是否绘制频谱图。
    - isYearCoord: bool, optional (default=False)
        是否将横坐标转换为以年为单位的周期。
    - useSoothed: bool, optional (default=False)
        是否使用平滑处理。
    - smooth_window: int, optional (default=50)
        平滑窗口大小。
    - color: str, optional (default='royalblue')
        绘图颜色。

    返回：
    - show_coord: np.ndarray
        频率(周期)坐标。
    - positive_amplitudes: np.ndarray
        频谱幅值（对应周期）。
    """
    
    # 计算 FFT 和频率
    n = len(signal)
    fft_result = np.fft.fft(signal,n)  # 快速傅里叶变换
    freqs = np.fft.fftfreq(len(signal), d=1/sampling_rate)  # 频率轴（单位：1/月）
    if norm_amp:
        amplitudes = 2 * np.abs(fft_result) / n  # 振幅归一化，乘以 2 以保留正负频率的完整能量
    else:
        amplitudes = np.abs(fft_result)

    # 只保留正频率部分
    positive_freqs = freqs[freqs > 0]  # 正频率
    positive_amplitudes = amplitudes[freqs > 0]
    if isYearCoord:
        # 转换为周期（以年为单位）
        show_coord = 1 / positive_freqs / 12  # 周期（单位：年）
        xlabel = r"Period ($year$)"
    else:
        show_coord = positive_freqs
        xlabel = r"freq ($month^{-1}$)"

    if useSoothed:
        from scipy.signal import savgol_filter
        positive_amplitudes = savgol_filter(positive_amplitudes, window_length=smooth_window, polyorder=3)
    else:
        positive_amplitudes = positive_amplitudes
    if plotFig:
        # 绘制频谱
        plt.figure(figsize=(10, 6))
        plt.plot(show_coord, positive_amplitudes, label="Frequency Spectrum",color=color)
        plt.xlabel(xlabel)
        plt.ylabel("Amplitude")
        plt.title(title)
        plt.grid(alpha=0.5, which="both", linestyle="--", color='gray')
        plt.legend()
        plt.show()
    # 返回频率(周期)和对应的幅值
    return show_coord, positive_amplitudes


def plot_xy_dict(x_dict, y_dict, colors=None,figsize=(10, 6), linestyles=None, title="Plot from Dictionaries", xlabel="X", ylabel="Y", grid_on=True):
    """
    从 x, y, colors, 和 linestyles 字典中获取值，并绘制在一张图中。

    Parameters:
    - x_dict: dict
        包含 x 数据的字典，键为曲线名，值为 x 数据。
    - y_dict: dict
        包含 y 数据的字典，键为曲线名，值为 y 数据。
    - colors: dict, optional
        包含颜色的字典，键为曲线名，值为颜色（默认值为 "blue"）。
    - linestyles: dict, optional
        包含线条样式的字典，键为曲线名，值为线条样式（默认值为 "-"）。
    - title: str, optional
        图的标题，默认值为 "Plot from Dictionaries"。
    - xlabel: str, optional
        x 轴标签，默认值为 "X"。
    - ylabel: str, optional
        y 轴标签，默认值为 "Y"。

    Returns:
    - None
    """
    plt.figure(figsize=figsize)  # 设置图表大小

    # 遍历曲线名（假设 x_dict 和 y_dict 的键一致）
    for label in x_dict.keys():
        # 获取 x 和 y 数据
        x = x_dict.get(label)
        y = y_dict.get(label)

        if x is None or y is None:
            raise ValueError(f"Missing 'x' or 'y' data for '{label}'")

        # 获取颜色和线条样式
        color = colors.get(label, "blue") if colors else "blue"  # 默认颜色为蓝色
        linestyle = linestyles.get(label, "-") if linestyles else "-"  # 默认线条样式为实线

        # 绘制曲线
        plt.plot(x, y, label=label, color=color, linestyle=linestyle)

    # 添加标题和标签
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 添加图例
    plt.legend()

    # 显示网格
    if grid_on:
        plt.grid(grid_on,alpha=0.5, which="both", linestyle="--", color='gray')

    # 显示图像
    plt.show()


def get_stochastic_mean(Y_pred_sim,axis=0):
    return np.mean(Y_pred_sim, axis=axis)

def plot_main_skills_with_legend(skill_dict: dict,
                                 styles: dict = None,
                                 skill_name='Skill',
                                 title=None,
                                 figsize=(12, 5),
                                 grid_on=True,
                                 legend=True,
                                 xticks=None,
                                 yticks=None,
                                 xlim=None,
                                 ylim=(0, 1),
                                 add_hline=True,
                                 legend_loc="upper center",
                                 legend_in_main=False,
                                 n_cols=1):  
    """
    在主图中绘制曲线，并根据 legend_in_main 参数决定图例位置。
    支持绘制上下界（当值为字典时）。
    只绘制实际绘制过的曲线的图例。
    
    参数:
    --------
    skill_dict : dict
        - 键：曲线名称
        - 值：可以是数组（简单曲线）或字典（包含 'avg', 'lower', 'upper' 的上下界）
    legend_in_main : bool, default=False
        - True: 在主图坐标系中绘制图例
        - False: 在右侧单独子图中绘制图例
    """
    # 最大步数，steps 为字典中值的最大长度
    def get_length(val):
        if isinstance(val, dict):
            return len(val['avg'])
        else:
            return len(val)
    
    steps = max(get_length(values) for values in skill_dict.values())
    x_axis = np.arange(0, steps)

    # 创建颜色迭代器（默认使用 Matplotlib 的 tab10 调色板）
    default_colors = cycle(plt.cm.tab10.colors)

    # 根据 legend_in_main 参数决定图形布局
    if legend_in_main:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)
        ax_legend = None
    else:
        fig, (ax, ax_legend) = plt.subplots(1, 2, figsize=figsize, dpi=300, 
                                            gridspec_kw={"width_ratios": [4, 1]})

    # 记录实际绘制的曲线
    plotted_keys = []
    
    # 绘制每条曲线（按照 skill_dict 的顺序）
    for key, values in skill_dict.items():
        # 获取曲线样式
        style = styles.get(key, {}) if styles else {}
        color = style.get('color', next(default_colors))
        linestyle = style.get('linestyle', '-')
        marker = style.get('marker', '')
        linewidth = style.get('linewidth', 1.5)
        markersize = style.get('markersize', 7)
        alpha = style.get('alpha', 1.0)
        hollowmarker = style.get('hollowmarker', False)
        
        # 如果值是字典，绘制上下界
        if isinstance(values, dict):
            avg = np.full(steps, np.nan)
            lower = np.full(steps, np.nan)
            upper = np.full(steps, np.nan)

            # 填充有效数据
            avg[:len(values['avg'])] = values['avg']
            lower[:len(values['lower'])] = values['lower']
            upper[:len(values['upper'])] = values['upper']
            
            # 根据 xlim 截取数据
            if xlim is not None:
                start, end = xlim
                x_indices = (x_axis >= start) & (x_axis < end)
                x_axis_trimmed = x_axis[x_indices]
                avg_trimmed = avg[x_indices]
                lower_trimmed = lower[x_indices]
                upper_trimmed = upper[x_indices]
            else:
                x_axis_trimmed = x_axis
                avg_trimmed = avg
                lower_trimmed = lower
                upper_trimmed = upper
            
            # 绘制平均值曲线
            ax.plot(x_axis_trimmed, avg_trimmed,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=linewidth,
                    markersize=markersize,
                    markerfacecolor='none' if hollowmarker else color,
                    alpha=alpha,
                    label=key)
            
            # 绘制上下界阴影
            ax.fill_between(x_axis_trimmed, lower_trimmed, upper_trimmed,
                           color=color, alpha=0.5)
        
        else:
            # 简单曲线：处理 NaN 值
            y_values = np.full(steps, np.nan)
            y_values[:len(values)] = values
            
            # 根据 xlim 截取数据
            if xlim is not None:
                start, end = xlim
                x_indices = (x_axis >= start) & (x_axis < end)
                x_axis_trimmed = x_axis[x_indices]
                y_values_trimmed = y_values[x_indices]
            else:
                x_axis_trimmed = x_axis
                y_values_trimmed = y_values

            # 绘制曲线
            ax.plot(x_axis_trimmed, y_values_trimmed,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=linewidth,
                    markersize=markersize,
                    markerfacecolor='none' if hollowmarker else color,
                    alpha=alpha,
                    label=key)
        
        # 记录已绘制的key
        plotted_keys.append(key)
        
    # 主图设置
    # 添加 y=0.5 的水平虚线
    if add_hline:
        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1, label='y=0.5')

    # 添加网格、标题和标签
    if grid_on:
        ax.grid(True)
    if title:
        ax.set_title(title, fontsize=16)
    ax.set_xlabel('Lead time (months)', fontsize=14)
    ax.set_ylabel(skill_name, fontsize=14)

    # 设置 xy 轴主刻度和副刻度
    if yticks is None:
        yticks = np.arange(0, 1.01, 0.1)
    ax.set_yticks(yticks)  
    
    # 设置 x 轴主刻度
    if xticks is None:
        xticks = np.arange(1, steps + 1, 2)
    ax.set_xticks(xticks)
    
    # 添加 x 轴次刻度（每个月一个）
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    # 设置 y 轴范围
    if ylim is not None:
        ax.set_ylim(ylim)
    # 设置 x 轴范围
    if xlim is not None:
        ax.set_xlim(xlim)

    # 设置右开口样式：隐藏顶部和右侧轴线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 只为实际绘制的曲线创建图例元素
    legend_elements = []
    if styles:
        for key in plotted_keys:
            if key in styles:
                style = styles[key]
                color = style.get('color', next(default_colors))
                linestyle = style.get('linestyle', '-')
                marker = style.get('marker', '')
                linewidth = style.get('linewidth', 1.5)
                markersize = style.get('markersize', 7)
                alpha = style.get('alpha', 1.0)
                hollowmarker = style.get('hollowmarker', False)

                legend_elements.append(Line2D([0], [0],
                                              color=color,
                                              linestyle=linestyle,
                                              marker=marker,
                                              linewidth=linewidth,
                                              markersize=markersize,
                                              markerfacecolor='none' if hollowmarker else color,
                                              alpha=alpha,
                                              label=key))

    # 根据 legend_in_main 参数绘制图例
    if legend:
        if legend_in_main:
            ax.legend(handles=legend_elements, loc=legend_loc, fontsize=12, ncols=n_cols)
        else:
            ax_legend.axis("off")
            ax_legend.legend(handles=legend_elements, loc=legend_loc, fontsize=12, ncols=n_cols)

    # 显示图表
    plt.tight_layout()
    plt.show()

def plot_several_skills(skill_dict: dict, 
                        colors: dict = None, 
                        linestyles: dict = None,
                        markers: dict = None, 
                        alphas: dict = None,
                        markersizes: dict = None,
                        skill_name='Correlation',
                        title=None, 
                        steps=22,
                        start_step=1, 
                        isSmoothed=True,
                        window=3,
                        interpolation_points=10,
                        xticks=None,
                        yticks=None,
                        xlim=None,
                        ylim=(0, 1),         
                        smooth_method='cubic',figsize=(10, 4),grid_on=False,legend=True):
    """
    绘制多条技能曲线，支持平滑曲线（插值法）。

    Parameters:
        skill_dict (dict): 技能数据字典，每个键对应一条曲线。
        colors (dict): 每条曲线的颜色字典。
        linestyles (dict): 每条曲线的线型字典。
        markers (dict): 每条曲线的标记字典。
        skill_name (str): 技能名称，用于标题。
        steps (int): 原始数据的最大步数。
        start_step (int): 开始绘制的步数。
        isSmoothed (bool): 是否平滑曲线。
        window (int): 移动平均窗口大小。
        interpolation_points (int): 每两个原始点之间插值的点数。
        smooth_method (str): 插值方法，默认 'cubic'（三次样条插值）。
                                   支持 'linear', 'quadratic', 'cubic'。
    """
    max_value = 0
    plt.figure(figsize=figsize)

    # 如果没有传入 colors, linestyles, markers，则使用默认样式
    if colors is None:
        colors = {key: plt.cm.tab10(i) for i, key in enumerate(skill_dict.keys())}
    if linestyles is None:
        linestyles = {key: '-' for key in skill_dict.keys()}
    if markers is None:
        markers = {key: '' for key in skill_dict.keys()}  # 默认不使用标记
    if markersizes is None:
        markersizes = {key: 10 for key in skill_dict.keys()}
    if alphas is None:
        alphas = {key: 1.0 for key in skill_dict.keys()}  

    # 保存初始的 steps 值
    initial_steps = steps

    # 用于存储图例项
    legend_elements = []

    for keys, values in skill_dict.items():
        # 去除 NaN 值
        values = values[~np.isnan(values)]
        
        # 动态调整 steps 以适应 values 的长度
        if len(values) < steps:
            steps = len(values)
        
        # 提取前 `steps` 个点的数据
        x_original = np.arange(start_step, steps)  # 原始 x 轴
        y_original = values[start_step:steps]   # 原始 y 数据
        
        # 更新最大值
        if max_value < max(y_original):
            max_value = max(y_original)
        
        # 插值计算
        if isSmoothed:
            if smooth_method == 'moving avg':
                # 使用移动平均平滑
                y_interp = pd.Series(y_original).rolling(window=window, center=True, min_periods=1).mean()
                x_interp = x_original
            elif smooth_method == 'linear' or 'quadratic' or 'cubic':
                # 在每两个原始点之间插值 `interpolation_points` 个点
                x_interp = np.linspace(x_original[0], x_original[-1], num=(steps - 1) * interpolation_points + steps)
                interp_func = interp1d(x_original, y_original, kind=smooth_method, fill_value="extrapolate")
                y_interp = interp_func(x_interp)
            else:
                # 使用移动平均平滑
                y_interp = pd.Series(y_original).rolling(window=window, center=True, min_periods=1).mean()
                x_interp = x_original
        else:
            # 如果不平滑，直接使用原始数据
            x_interp = x_original
            y_interp = y_original
        
        # 绘制平滑曲线
        plt.plot(x_interp, y_interp, 
                 color=colors[keys], 
                 linestyle=linestyles[keys],
                 alpha=alphas[keys],
                 marker='')  # 不使用标记
        
        # 绘制原始数据点
        plt.plot(x_original, y_original, 
                 color=colors[keys], 
                 marker=markers[keys],
                 markersize=markersizes[keys], 
                 alpha=alphas[keys],
                 linestyle='')  # 不使用线型
        
        # 创建图例项
        legend_elements.append(Line2D([0], [0], color=colors[keys], 
                                      linestyle=linestyles[keys], 
                                      marker=markers[keys],
                                      markersize=markersizes[keys], 
                                      alpha=alphas[keys], label=keys))
        
        # 重置 steps 为初始值
        steps = initial_steps
    
    # 添加参考线（适用于 correlation）
    if skill_name == 'Correlation':
        plt.axhline(y=0.5, color='black', linestyle='--', lw=1)

    # 添加网格、标题和标签
    if grid_on:
        plt.grid(alpha=0.3, which="both", linestyle="--", color='gray')
        # 添加网格线（可选）
        ax.grid(which='major', color='black', linestyle='--', linewidth=0.5)  # 主刻度网格线
        ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)  # 副刻度网格线
    if title:
        plt.title(title, fontsize=16)
    plt.xlabel('Lead time (months)', fontsize=14)
    plt.ylabel(skill_name, fontsize=14)

    # 设置右开口样式：隐藏顶部和右侧轴线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 隐藏顶部轴线
    ax.spines['right'].set_visible(False)  # 隐藏右侧轴线

    # 设置 X 轴和 Y 轴的主刻度和副刻度
    ax.xaxis.set_major_locator(MultipleLocator(2))  # 主刻度间隔 2
    if xticks is not None:
        ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_minor_locator(MultipleLocator(1))  # 副刻度间隔 1

    # 设置 Y 轴主刻度为间隔 0.5，副刻度为间隔 0.1
    ax.yaxis.set_major_locator(MultipleLocator(0.2))  # 主刻度间隔 0.2
    if yticks is not None:
        ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))  # 副刻度间隔 0.1

    

    # 设置 y 轴范围
    if ylim is not None:
        plt.ylim(ylim)
    # 设置 x 轴范围
    if xlim is not None:
        plt.xlim(xlim)


    # 添加图例
    if legend:
        plt.legend(handles=legend_elements,fontsize=12)
    
    plt.show()

def compute_jacobian(W_out, W_res, r_t, x_t=None, W_in=None, alpha=1.0):
    m = W_out.T.shape[0]
    n = W_res.shape[0]
    if x_t is None:
        x_t = np.zeros((1, m))
    if W_in is None:
        W_in = np.zeros((n, m))
    # Calculate the diagonal terms from the activation function derivative
    tanh_derivative = 1 - np.tanh((W_in @ x_t.T).reshape(n,) + (W_res @ r_t).reshape(n,))**2
    diag_terms = alpha * np.diag(tanh_derivative)
    # Calculate Jacobian matrix
    J_rr = diag_terms @ W_res
    J_rx = diag_terms @ W_in
    J_xr = W_out.T @ diag_terms @ W_res
    J_xx = W_out.T @ diag_terms @ W_in
    # Combine into full Jacobian matrix
    J = np.block([[J_rr.reshape(n,n), J_rx.reshape(n,m)],
                  [J_xr.reshape(m,n), J_xx.reshape(m,m)]])
    return J

def compute_jacobian_deep_esn(W_out, W_res1, W_res2, W_in, W_21, r1, r2, x_t):
    """
    计算双层 ESN 的雅可比矩阵 (Jacobian Matrix)。

    参数:
        W_res1: 第一层的递归权重矩阵 (n1 x n1)
        W_res2: 第二层的递归权重矩阵 (n2 x n2)
        W_in: 输入到第一层的权重矩阵 (n1 x m)
        W_21: 第一层到第二层的连接权重矩阵 (n2 x n1)
        W_out: 输出层的权重矩阵 ((n1 + n2) x m)
        r1: 第一层的当前状态向量 (n1, 1)
        r2: 第二层的当前状态向量 (n2, 1)
        x_t: 当前输入向量 (1, m)

    返回:
        雅可比矩阵 J ((n1 + n2 + 2m) x (n1 + n2 + m))
    """
    # 获取各维度
    n1 = len(r1)  # 第一层神经元数量
    n2 = len(r2)  # 第二层神经元数量
    m = len(x_t.T)  # 输入维度

    # 第一层的激活函数导数矩阵 D1
    D1 = np.diag(1 - np.tanh((W_in @ x_t.T).reshape(n1,) + (W_res1 @ r1).reshape(n1,)) ** 2)

    # 第二层的激活函数导数矩阵 D2
    D2 = np.diag(1 - np.tanh((W_21 @ r1).reshape(n2,) + (W_res2 @ r2).reshape(n2,)) ** 2)

    # 构建雅可比矩阵的分块
    # 1. J_r1_r1
    J_r1_r1 = D1 @ W_res1
    
    # 2. J_r1_r2
    J_r1_r2 = np.zeros((n1, n2))
    
    # 3. J_r1_x
    J_r1_x = D1 @ W_in

    # 4. J_r2_r1
    J_r2_r1 = D2 @ W_21
    
    # 5. J_r2_r2
    J_r2_r2 = D2 @ W_res2
    
    # 6. J_r2_x
    J_r2_x = np.zeros((n2, m))

    # 7. J_x_r1
    J_x_r1 = W_out[:n1,: ].T @ (D1 @ W_res1) + W_out[n1:,: ].T @ (D2 @ W_21)    
   
    
    # 8. J_x_r2
    J_x_r2 = np.zeros((m, n2)) + W_out[n1:, :].T @ (D2 @ W_res2)  

    # 9. J_x_x
    J_x_x = W_out[:n1, :].T @ (D1 @ W_in) + np.zeros((m, m))              

    # 拼接雅可比矩阵
    # 第一部分 (r1 的导数)
    J_top = np.hstack([J_r1_r1, J_r1_r2, J_r1_x])
    
    # 第二部分 (r2 的导数)
    J_middle = np.hstack([J_r2_r1, J_r2_r2, J_r2_x])
    
    # 第三部分 (x 的导数)
    J_bottom = np.hstack([J_x_r1, J_x_r2, J_x_x])
    
    # 合并所有部分
    J = np.vstack([J_top, J_middle, J_bottom])

    return J

def DESN_Lyaps_with_Record(TS, tl=None, num_lyaps=40, iterations=1000, norm_time=10, desn=None):
    """
    计算双层 DESN 的李雅普诺夫指数并记录中间结果。

    参数:
        TS: 输入时间序列 (total_time x m)
        tl: 时间序列长度（默认取 TS 的行数）
        num_lyaps: 要计算的李雅普诺夫指数数量
        iterations: 总迭代步数
        norm_time: 每隔 norm_time 时间步进行正交化和归一化
        desn: 已训练的深度 ESN 模型

    返回:
        LE: 最终的李雅普诺夫指数 (num_lyaps,)
        le_record: 每隔 norm_time 步记录的李雅普诺夫指数 (num_lyaps x (iterations // norm_time))
        R_ii_record: 每隔 norm_time 步记录的对数扩展因子 (num_lyaps x (iterations // norm_time) x norm_time)
    """
    if tl is None:
        tl = TS.shape[0]
    if desn is None:
        raise ValueError("You must provide a trained Deep ESN instance (desn).")

    Xtrain = TS[:tl-1,]
    Ytrain = TS[1:tl,]
    desn = desn.fit(Xtrain, Ytrain)

    W_res1 = desn.nodes[1].W      # 第一层递归权重
    W_res2 = desn.nodes[2].W      # 第二层递归权重
    W_in = desn.nodes[1].Win      # 输入到第一层的权重
    W_21 = desn.nodes[2].Win      # 第一层到第二层的权重
    W_out = desn.nodes[4].Wout    # 输出层权重

    n1 = W_res1.shape[0]
    n2 = W_res2.shape[0]
    m = TS.shape[1]

    delta = orth(np.random.rand(n1 + n2 + m, num_lyaps))  # 初始正交基
    x_t = TS[0, :].reshape(1, m)
    Sum = np.zeros(num_lyaps)
    le_record = np.zeros((num_lyaps, iterations // norm_time))
    R_ii_record = np.zeros((num_lyaps, iterations // norm_time, norm_time))

    r1 = np.zeros((n1,1))
    r2 = np.zeros((n2,1))

    for t in tqdm(range(iterations), desc="LE Iterations"):
        # 确保输入的形状
        if x_t.ndim == 1:
            x_t = x_t.reshape(1, -1)  # 保证 x_t 为二维向量 (1, m)

        # 更新第一层和第二层的状态
        r1 = np.tanh(W_in @ x_t.T + W_res1 @ r1)  # 第一层状态更新 (n1,1)
        r2 = np.tanh(W_21 @ r1 + W_res2 @ r2)     # 第二层状态更新 (n2,1)


        # 更新输出
        x_t = desn.run(x_t)  # 更新输出，x_t 应为形状 (1, m)

        # 计算雅可比矩阵
        J = compute_jacobian_deep_esn(W_out, W_res1, W_res2, W_in, W_21, r1, r2, x_t)
        
        # 更新微扰向量
        delta = J @ delta
        # 截断 delta 的行数，使其与输入维度匹配
        delta = delta[:n1 + n2 + m, :]

        if (t + 1) % norm_time == 0:
            # 使用 QR 分解进行正交化
            Q, R = qr(delta, mode='economic')
            delta = Q[:, :num_lyaps]
            R_ii = np.log(np.abs(np.diag(R[:num_lyaps, :num_lyaps])))
            Sum += R_ii
            R_ii_record[:, (t + 1) // norm_time - 1, t % norm_time] = np.real(R_ii)
            le_record[:, (t + 1) // norm_time - 1] = np.real(Sum) / (t + 1)
        else:
            _, R = qr(delta, mode='economic')
            R_ii = np.log(np.abs(np.diag(R[:num_lyaps, :num_lyaps])))
            R_ii_record[:, (t + 1) // norm_time - 1, t % norm_time] = np.real(R_ii)

    LE = np.real(Sum) / iterations
    return LE, le_record, R_ii_record


def return_states(reservoir, TS, tl):
    states = []
    states = reservoir.run(TS[:tl,])
    return states

def plot_eigenvalues(eigenvalues, threshold=0.05, title='Eigenvalues in the Complex Plane',gridon=True,legend=True, return_fig=False, small_fig=False):
    """
    绘制特征值分布，并在单位圆附近用颜色填充表示阈值区域。
    
    Parameters:
        eigenvalues (array-like): 特征值序列（复数）。
        threshold (float): 判断单位圆附近的距离阈值。
        title (str): 图标题。
        return_fig (bool): 是否返回 Matplotlib 的 Figure 对象而不直接绘制。
    
    Returns:
        fig (matplotlib.figure.Figure) or None: 如果 return_fig=True, 则返回 Figure 对象；否则无返回值。
    """
    if small_fig:
        fig, ax = plt.subplots(figsize=(3, 3))
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
    # 分类特征值
    inside = []  # 单位圆内的特征值
    near = []    # 单位圆附近的特征值
    outside = [] # 单位圆外的特征值
    
    for ev in eigenvalues:
        distance = abs(ev)  # 距离原点的模
        if distance < 1 - threshold:
            inside.append(ev)  # 单位圆内
        elif 1 - threshold <= distance <= 1 + threshold:
            near.append(ev)    # 单位圆附近
        else:
            outside.append(ev) # 单位圆外

    # 转换为 NumPy 数组便于处理
    inside = np.array(inside)
    near = np.array(near)
    outside = np.array(outside)
    
    # 绘制阈值区域（单位圆附近）
    outer_circle = plt.Circle((0, 0), 1 + threshold, color='deepskyblue', alpha=0.2, label='Threshold Region', zorder=0)
    inner_circle = plt.Circle((0, 0), 1 - threshold, color='white', alpha=1, zorder=1)
    ax.add_artist(outer_circle)  # 外圆
    ax.add_artist(inner_circle)  # 内圆（覆盖外圆中间部分）
    
    # 绘制单位圆
    unit_circle = plt.Circle((0, 0), 1, color='deepskyblue', fill=False, linestyle='-', linewidth=1.5,label='Unit Circle', zorder=2)
    ax.add_artist(unit_circle)
    
    # 绘制分类后的特征值
    if small_fig:
        if inside.size > 0:
            ax.scatter(inside.real, inside.imag, color='royalblue', s=1, label='Inside Unit Circle', zorder=3)
        if near.size > 0:
            ax.scatter(near.real, near.imag, color='orangered', s=10, label='Near Unit Circle', zorder=3)
        if outside.size > 0:
            ax.scatter(outside.real, outside.imag, color='mediumvioletred', s=10, label='Outside Unit Circle', zorder=3)
    else:
        if inside.size > 0:
            ax.scatter(inside.real, inside.imag, color='royalblue', s=5, label='Inside Unit Circle', zorder=3)
        if near.size > 0:
            ax.scatter(near.real, near.imag, color='orangered', s=20, label='Near Unit Circle', zorder=3)
        if outside.size > 0:
            ax.scatter(outside.real, outside.imag, color='mediumvioletred', s=20, label='Outside Unit Circle', zorder=3)
    
    # 设置轴范围和样式
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title(title,loc='left')
    if gridon:
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.grid(True, linestyle='--', alpha=0.7)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    if legend:
        ax.legend(loc='upper left')
    
    # 根据 return_fig 参数决定是否返回 Figure 对象
    if return_fig:
        return fig  # 返回 Figure 对象
    else:
        plt.show()  # 直接绘制图像

def top_n_frequencies(data, max_freq_count=10, min_interval=0.001, fs=1, plot=False, 
                      useSoothed=True, 
                      smooth_window=5):
    """
    获取信号的前 N 个主要频率及其对应的振幅和相位，并可选择绘制频谱图 (使用振幅归一化)。

    Parameters:
        data (array-like): 输入信号数据。
        max_freq_count (int): 返回的频率数量上限。
        min_interval (float): 选取频率间的最小间隔。
        fs (float): 采样频率。
        plot (bool): 是否绘制频谱图。

    Returns:
        selected_freqs (list): 选取的主要频率。
        amplitudes (list): 对应的振幅。
        phases (list): 对应的相位。
    """
    # Perform FFT
    n = len(data)
    freqs = fftfreq(n, d=1/fs)
    fft_result = fft(data,n)
    amplitudes = 2 * np.abs(fft_result) / n  # 振幅归一化，乘以 2 以保留正负频率的完整能量

    # Take only positive frequencies
    positive_freqs = freqs > 0
    freqs = freqs[positive_freqs]
    amplitudes = amplitudes[positive_freqs]
    fft_result = fft_result[positive_freqs]  # 只保留正频率部分的 FFT 结果
    if useSoothed:
        from scipy.signal import savgol_filter
        amplitudes = savgol_filter(amplitudes, window_length=smooth_window, polyorder=3)
    else:
        amplitudes = amplitudes
    # Sort frequencies by amplitude
    sorted_indices = np.argsort(amplitudes)[::-1]
    sorted_freqs = freqs[sorted_indices]
    sorted_amplitudes = amplitudes[sorted_indices]
    sorted_fft_result = fft_result[sorted_indices]

    # Filter to get top frequencies with specified interval
    selected_freqs = []
    selected_amplitudes = []
    selected_phases = []

    for i, f in enumerate(sorted_freqs):
        if len(selected_freqs) >= max_freq_count:
            break
        if len(selected_freqs) == 0 or all(abs(f - sf) > min_interval for sf in selected_freqs):
            selected_freqs.append(f)
            selected_amplitudes.append(sorted_amplitudes[i])  # 振幅
            selected_phases.append(np.angle(sorted_fft_result[i]))  # 相位

    # Plot frequency spectrum if required
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(freqs, amplitudes, label="Amplitude Spectrum", color="blue")
        plt.scatter(selected_freqs, selected_amplitudes, color="red", label="Selected Frequencies",marker='*', zorder=5)
        plt.title("Frequency Spectrum with Selected Frequencies (Amplitude)")
        plt.xlabel(r"Frequency (month$^{-1}$)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.show()

    return selected_freqs, selected_amplitudes, selected_phases

def generate_multidimensional_waves(selected_freqs, amplitudes, phases, t):
    """
    根据选取的频率、振幅和相位生成多维基波时间序列。

    Parameters:
        selected_freqs (list): 选取的频率 (Hz)。
        amplitudes (list): 对应的振幅。
        phases (list): 对应的相位 (弧度)。
        t (array-like): 时间序列 (秒)。

    Returns:
        waves (2D numpy array): 每一列是一个基波的时间序列，形状为 (len(selected_freqs), len(t))。
    """
    num_waves = len(selected_freqs)  # 基波数量
    waves = np.zeros((len(t),num_waves))  # 初始化二维数组，形状为 ( 时间点数量,基波数量)

    # 遍历每个频率、振幅和相位，生成对应的基波
    for i in range(num_waves):
        f = selected_freqs[i]      # 当前频率 (Hz)
        A = amplitudes[i]          # 当前振幅
        phi = phases[i]            # 当前相位 (弧度)
        waves[:, i] = A * np.cos(2 * np.pi * f * t + phi)  # 基波公式

    return waves

def generate_cycles(selected_freqs, t):
    """
    根据选取的频率、振幅和相位生成sin/cos序列。

    Parameters:
        selected_freqs (list): 选取的频率 (Hz)。
        t (array-like): 时间序列 (秒)。

    Returns:
        waves (2D numpy array): 每一列是一个不含相位的基波时间序列，形状为 (len(selected_freqs), len(t))。
    """
    num_waves = len(selected_freqs)  # 基波数量
    waves = np.zeros((len(t),2*num_waves))  # 初始化二维数组，形状为 ( 时间点数量,基波数量)

    # 遍历每个频率、振幅和相位，生成对应的基波
    for i in range(num_waves):
        f = selected_freqs[i]      # 当前频率 (Hz)
        waves[:, 2*i] = np.cos(2 * np.pi * f * t)
        waves[:, 2*i+1] = np.sin(2 * np.pi * f * t)

    return waves

def generate_anual_cycles(t):
    return generate_cycles([2*np.pi/12], t)

def generate_semi_anual_cycles(t):
    anual_cycles = generate_cycles([2*np.pi/12], t)
    semi_anual_cycles = generate_cycles([2*np.pi/6], t)
    waves = np.hstack((anual_cycles,semi_anual_cycles))
    return waves

def get_sub_dict(original_dict, keys):
    """
    根据指定的键列表，从原始字典中提取子字典。

    Parameters:
        original_dict (dict): 原始字典。
        keys (list): 要提取的键列表。

    Returns:
        sub_dict (dict): 提取后的子字典，仅包含指定的键。
    """
    return {key: original_dict[key] for key in keys if key in original_dict}

def generate_signals_from_fourier(selected_freqs, amplitudes, t, sampling_rate=1,bandwidth=0.001):
    """
    根据选定的频带，通过傅里叶变换生成多维信号。

    Parameters:
        selected_freqs (list): 中心频率列表 (Hz)。
        bandwidth (float): 中心频率的频带宽度 (Hz)。
        amplitudes (list): 每个频带的振幅因子。
        t (array-like): 时间序列 (秒)。
        sampling_rate (float): 采样率 (Hz)。

    Returns:
        signals (2D numpy array): 每列是一个频带生成的时间信号 (shape: len(t) x len(selected_freqs))。
    """
    num_signals = len(selected_freqs)  # 信号数量
    num_points = len(t)  # 时间点数量
    signals = np.zeros((num_points, num_signals))  # 初始化二维数组

    # 对每个频带生成信号
    for i in range(num_signals):
        f_center = selected_freqs[i]  # 中心频率
        amplitude = amplitudes[i]  # 振幅因子

        # 创建频域信号
        freq = np.fft.fftfreq(num_points, d=1/sampling_rate)  # 频率轴
        spectrum = np.zeros(num_points, dtype=complex)  # 初始化频域信号

        # 定义频带范围
        band_mask = (np.abs(freq) >= f_center - bandwidth / 2) & (np.abs(freq) <= f_center + bandwidth / 2)

        # 在频带范围内随机赋值（振幅和相位）
        spectrum[band_mask] = amplitude * (np.random.rand(np.sum(band_mask)) + 
                                           1j * np.random.rand(np.sum(band_mask)))

        # 保证共轭对称性（时域信号为实数）
        spectrum = np.fft.ifftshift(np.fft.fftshift(spectrum).conj())

        # 逆傅里叶变换得到时域信号
        signal = np.fft.ifft(spectrum).real  # 取实部
        signals[:, i] = signal  # 保存到结果数组中

    return signals
def get_TS_TP(TS,tl,concern_dim=0,sel_freqs=4,min_interval=0.001,smooth_window=20,useBandwidth=True,plot=True):
    t = np.arange(TS.shape[0])
    if concern_dim==None:
        selected_freqs, selected_amplitudes, selected_phases = top_n_frequencies(TS[:tl,:].T.flatten(), 
                                                                            max_freq_count=sel_freqs, 
                                                                            min_interval=min_interval, 
                                                                            fs=1,plot=plot,
                                                                            useSoothed=True,
                                                                            smooth_window=smooth_window)
    else:
        selected_freqs, selected_amplitudes, selected_phases = top_n_frequencies(TS[:tl,concern_dim], 
                                                                                max_freq_count=sel_freqs, 
                                                                                min_interval=min_interval, 
                                                                                fs=1,plot=plot,
                                                                                useSoothed=True,
                                                                                smooth_window=smooth_window)
    if useBandwidth:
        TP = generate_signals_from_fourier(selected_freqs, selected_amplitudes,t,bandwidth=min_interval)
    else:
        TP = generate_multidimensional_waves(selected_freqs, selected_amplitudes,selected_phases,t)
    TS_TP = np.hstack((TS,TP))
    return TS_TP

def euclidean_distance(ts1, ts2):
    """
    计算两个多维时间序列的平均欧几里得距离。

    Parameters:
        ts1 (2D array): 时间序列 1,形状为 (时间点数, 维度数)。
        ts2 (2D array): 时间序列 2,形状为 (时间点数, 维度数)。

    Returns:
        float: 平均欧几里得距离。
    """
    if ts1.shape != ts2.shape:
        raise ValueError("两个时间序列的形状必须相同。")
    return np.mean(np.sqrt((ts1 - ts2)**2))

def compute_dist_skill(Ypred, Ytest ,wl=12):
    steps = Ypred.shape[2]
    dist_skill = np.zeros(steps)
    for lead_time in range(1,steps):
        observation = Ytest[lead_time+wl:,:]
        prediction = Ypred[wl:-lead_time,:,lead_time]
        dist_skill[lead_time] = euclidean_distance(observation, prediction)
    return dist_skill

def TPRC_Train_Ensemble(
    TS,
    tl,
    dl=0,
    hypers=None,
    nmember=10,
    tp_omega=2 * np.pi / 12,
    tp_order=2,
    tp_bias=0
):
    """
    训练集合预报模型，返回训练好的 ESN 模型列表。
    Returns:
        trained_models: 训练好的 ESN 模型列表
        TP: 时间周期特征（用于后续预测）
    """
    # 获取时间周期特征 (TP)
    TP = get_RCTP(TS, steps=0, omega=tp_omega, order=tp_order, bias=tp_bias)
    TS_TP = np.hstack((TS, TP))
    
    Xtrain = TS_TP[:tl - 1]
    Ytrain = TS[1:tl]
    
    # 存储所有训练好的模型
    trained_models = []
    
    # 训练每个集合成员
    for m in tqdm(range(nmember), desc="Training ensemble members"):
        # 初始化并训练 ESN
        member_esn = get_esn_from_hypers(hypers)
        member_esn = member_esn.fit(Xtrain, Ytrain, warmup=dl)
        trained_models.append(member_esn)
    
    return trained_models, TP


def TPRC_Forecast_Ensemble(
    trained_models,
    Ytest,
    TP_test=None,
    steps=22,
    noise_ini=0.0,
    tp_order=2,
    tp_omega=2*np.pi/12
):
    """
    使用训练好的集合模型进行预测。
    
    Args:
        trained_models: 训练好的 ESN 模型列表
        Ytest: 测试数据 (包含初始条件和时间周期特征)
        TP_test: 测试期的时间周期特征，shape=(测试样本数+steps, TP维度)
        steps: 预测步数
        noise_ini: 初始噪声标准差
    
    Returns:
        Ypred_mean: 集合平均预测
        ensemble_predictions: 所有成员的预测结果
    """
    nmember = len(trained_models)
    if TP_test is None:
        TP_test = get_RCTP(Ytest, steps=steps, omega=tp_omega, order=tp_order, bias=0)
    # 记录所有成员的预测结果
    Ytest_TP = np.hstack((Ytest, TP_test[:-steps]))
    ensemble_predictions = np.zeros((nmember, Ytest.shape[0], Ytest.shape[1], steps))
    
    # 遍历每个集合成员进行预测
    for m in tqdm(range(nmember), desc="Forecasting with ensemble"):
        member_esn = trained_models[m]
        
        x = Ytest_TP.copy()
        Ypred = np.zeros((Ytest.shape[0], Ytest.shape[1], steps))
        
        # 生成多步预测
        for j in range(steps):
            if j == 0:
                Ypred[:, :, j] = Ytest
                # 添加噪声到初始状态
                noise = np.random.normal(0, noise_ini, size=x.shape)
                x += noise
            else:
                ts_prediction = member_esn.run(x)
                
                # 更新输入：预测值 + 对应时间步的 TP 特征
                x = np.hstack((ts_prediction, TP_test[j : Ytest.shape[0] + j]))
                
                Ypred[:, :, j] = ts_prediction
        
        ensemble_predictions[m] = Ypred
    
    # 计算集合平均
    Ypred_mean = np.mean(ensemble_predictions, axis=0)
    
    return Ypred_mean, ensemble_predictions


def dimension_addition_ensemble_forecast(ds, tl, hypers, wl=0, dl=0, retain_var=['Nino34', 'WWV'], 
                                         nmembers=10, steps=22, tp_omega=2*np.pi/12, 
                                         tp_order=2, noise_ini=0.0):
    """
    维度增加集合预报实验：retain_var + X（包含基线 retain_var）
    
    参数:
    - ds: 数据集
    - tl: 训练时长
    - hypers: DESN超参数 (dict 或 list)
    - wl: 预测drop时长（从测试集开始丢弃的步数）
    - dl: warmup时长（训练时丢弃的初始步数）
    - retain_var: 保持的初始变量列表（基线），默认 ['Nino34', 'WWV']
    - nmembers: 每组超参数的集合成员数
    - steps: 预测步数
    - tp_omega: 时间周期频率
    - tp_order: 时间周期阶数
    - noise_ini: 初始噪声标准差
    
    返回:
    - results: dict of arrays, 各组合的所有成员预测
        键: 'baseline', 'NPMM', 'SPMM', ...
    - results_mean: dict of arrays, 各组合的集合平均预测
        键: 'baseline', 'NPMM', 'SPMM', ...
    """
    # 获取所有变量
    all_vars = list(ds.data_vars)
    
    # 检查 retain_var 是否都在数据集中
    for var in retain_var:
        if var not in all_vars:
            raise ValueError(f"{var} not found in dataset")
    
    # 获取其他变量（除了 retain_var 中的变量）
    other_vars = [var for var in all_vars if var not in retain_var]
    
    # 打印配置信息
    if isinstance(hypers, list):
        total_members = len(hypers) * nmembers
        print(f"Configuration: {len(hypers)} hyperparameter sets × {nmembers} members = {total_members} total members")
    else:
        total_members = nmembers
        print(f"Configuration: 1 hyperparameter set × {nmembers} members = {total_members} total members")
    
    print(f"\nDimension Addition Ensemble Forecast - Baseline: {'+'.join(retain_var)}")
    print(f"Testing: {'+'.join(retain_var)} + {other_vars}\n")
    
    # 数据集转换函数
    def dataset_to_array(sub_ds):
        arrays = [sub_ds[var].values for var in sub_ds.data_vars]
        return np.stack(arrays, axis=1)
    
    def run_forecast_ensemble(combo_vars):
        """运行单个组合的集合预测"""
        # 构建组合数据集
        combo_ds = ds[combo_vars]
        TS = dataset_to_array(combo_ds)
        
        # 获取测试数据（从 tl+wl 开始）
        Ytest = TS[tl+wl:]
        
        # 训练集合模型
        if isinstance(hypers, list):
            # 如果hypers是列表，每组超参数训练nmembers个成员
            all_trained_models = []
            
            for hyper in hypers:
                trained_models, TP = TPRC_Train_Ensemble(
                    TS, tl, dl=dl, hypers=hyper, nmember=nmembers,
                    tp_omega=tp_omega, tp_order=tp_order, tp_bias=0
                )
                all_trained_models.extend(trained_models)
            
            trained_models = all_trained_models
        else:
            # 单组超参数
            trained_models, TP = TPRC_Train_Ensemble(
                TS, tl, dl=dl, hypers=hypers, nmember=nmembers,
                tp_omega=tp_omega, tp_order=tp_order, tp_bias=0
            )
        
        # 生成测试期的TP特征（从 tl+wl 开始）
        TP_test = get_RCTP(TS, steps=steps, omega=tp_omega, order=tp_order, bias=0)[tl+wl:]
        
        # 集合预测
        Ypred_mean, ensemble_predictions = TPRC_Forecast_Ensemble(
            trained_models, Ytest, TP_test=TP_test, steps=steps,
            noise_ini=noise_ini, tp_order=tp_order, tp_omega=tp_omega
        )
        
        return Ypred_mean, ensemble_predictions
    
    # 初始化结果字典
    results = {}
    results_mean = {}
    
    # 运行基线
    print(f"Running baseline ({'+'.join(retain_var)})...")
    Ypred_mean, ensemble_predictions = run_forecast_ensemble(retain_var)
    results['baseline'] = ensemble_predictions
    results_mean['baseline'] = Ypred_mean
    print(f"✓ Baseline completed (dim={len(retain_var)}, members={ensemble_predictions.shape[0]})")
    
    # 运行各组合
    print(f"\nTesting {len(other_vars)} combinations...")
    for x_var in tqdm(other_vars, desc='Testing combinations'):
        Ypred_mean, ensemble_predictions = run_forecast_ensemble(retain_var + [x_var])
        results[x_var] = ensemble_predictions
        results_mean[x_var] = Ypred_mean
    
    return results, results_mean

def dimension_reduction_ensemble_forecast(ds, tl, hypers,dl=0, wl=0, consern_var='Nino34', 
                                 exclude_dims=1, nmembers=10, steps=22,
                                 tp_omega=2*np.pi/12, tp_order=2, noise_ini=0.0,
                                 include_baseline=True):
    """
    从完整维度的模型中减去指定维度数，返回模型集合预测结果
    
    Parameters:
    -----------
    ds : xr.Dataset
        完整的数据集（包含所有变量）
    tl : int
        训练时长
    wl : int
        warmup时长，从tl+wl开始预测
    hypers : dict or list
        模型超参数
        - dict: 所有成员使用相同超参数，训练nmembers个成员
        - list: 每组超参数训练nmembers个成员，总成员数=len(hypers)*nmembers
    consern_var : str
        关注的变量，默认'Nino34'
    exclude_dims : int
        要排除的变量数量(1-9)
    nmembers : int
        每组超参数的集合预报成员数
    steps : int
        预测步数
    tp_omega : float
        时间周期频率
    tp_order : int
        时间周期阶数
    noise_ini : float
        初始噪声标准差
    include_baseline : bool
        是否包含baseline（不排除任何变量），默认True
        
    Returns:
    --------
    tuple: (results, results_mean)
        results: dict of arrays, 各组合的所有成员预测
        results_mean: dict of arrays, 各组合的集合平均预测
        
        baseline的键为'baseline'
        减去1维时，键为'-变量名'（如'-NPMM'）
        减去2维时，键为'-变量名1-变量名2'（如'-NPMM-SPMM'）
        ...
    """
    from itertools import combinations
    
    # 获取所有变量名（排除consern_var）
    all_vars = list(ds.drop_vars(consern_var).data_vars)
    n_vars = len(all_vars)
    
    # 检查排除维度数的合理性
    if exclude_dims < 1 or exclude_dims > n_vars:
        raise ValueError(f"exclude_dims must be between 1 and {n_vars}")
    
    # 数据集转换函数
    def dataset_to_array(sub_ds):
        arrays = [sub_ds[var].values for var in sub_ds.data_vars]
        return np.stack(arrays, axis=1)
    
    # 训练和预测的辅助函数
    def train_and_forecast(TS, key_name):
        """训练和预测的统一流程"""
        # 获取测试数据
        Ytest = TS[tl+wl:]
        
        # 训练集合模型
        if isinstance(hypers, list):
            # 如果hypers是列表，每组超参数训练nmembers个成员
            all_trained_models = []
            
            for hyper_idx, hyper in enumerate(hypers):
                # 为每组超参数训练nmembers个成员
                trained_models, TP = TPRC_Train_Ensemble(
                    TS, tl, dl=dl, hypers=hyper, nmember=nmembers,
                    tp_omega=tp_omega, tp_order=tp_order, tp_bias=0
                )
                all_trained_models.extend(trained_models)
            
            # 总成员数 = len(hypers) * nmembers
            total_members = len(all_trained_models)
            
            # 生成测试期的TP特征
            TP_test = get_RCTP(TS, steps=steps, omega=tp_omega, order=tp_order, bias=0)[tl+wl:]
            
            # 集合预测（使用所有成员）
            Ypred_mean, ensemble_predictions = TPRC_Forecast_Ensemble(
                all_trained_models, Ytest, TP_test=TP_test, steps=steps,
                noise_ini=noise_ini, tp_order=tp_order, tp_omega=tp_omega
            )
            
        else:
            # 如果hypers是单个字典，所有成员使用相同超参数
            trained_models, TP = TPRC_Train_Ensemble(
                TS, tl, dl=dl, hypers=hypers, nmember=nmembers,
                tp_omega=tp_omega, tp_order=tp_order, tp_bias=0
            )
            
            # 生成测试期的TP特征
            TP_test = get_RCTP(TS, steps=steps, omega=tp_omega, order=tp_order, bias=0)[tl+wl:]
            
            # 集合预测
            Ypred_mean, ensemble_predictions = TPRC_Forecast_Ensemble(
                trained_models, Ytest, TP_test=TP_test, steps=steps,
                noise_ini=noise_ini, tp_order=tp_order, tp_omega=tp_omega
            )
        
        return Ypred_mean, ensemble_predictions
    
    results = {}
    results_mean = {}
    
    # 打印配置信息
    if isinstance(hypers, list):
        total_members = len(hypers) * nmembers
        print(f"Configuration: {len(hypers)} hyperparameter sets × {nmembers} members = {total_members} total members")
    else:
        total_members = nmembers
        print(f"Configuration: 1 hyperparameter set × {nmembers} members = {total_members} total members")
    
    # 1. 首先计算baseline（如果需要）
    if include_baseline:
        print("\nTraining baseline model (all variables)...")
        full_vars = [consern_var] + all_vars
        full_ds = ds[full_vars]
        TS_full = dataset_to_array(full_ds)
        
        Ypred_mean_baseline, ensemble_pred_baseline = train_and_forecast(TS_full, 'baseline')
        
        results['baseline'] = ensemble_pred_baseline
        results_mean['baseline'] = Ypred_mean_baseline
        print(f"✓ Baseline completed (dim={len(full_vars)}, total_members={ensemble_pred_baseline.shape[0]})")
    
    # 2. 生成所有可能的排除组合
    exclude_combinations = list(combinations(all_vars, exclude_dims))
    
    print(f"\nTesting {len(exclude_combinations)} combinations (excluding {exclude_dims} variable(s))...")
    
    # 3. 遍历每个排除组合
    for exclude_vars in tqdm(exclude_combinations, desc=f"Exclude {exclude_dims} var(s)"):
        # 保留的变量（除了被排除的）
        kept_vars = [consern_var] + [v for v in all_vars if v not in exclude_vars]
        
        # 构建子数据集
        sub_ds = ds[kept_vars]
        TS = dataset_to_array(sub_ds)
        
        # 构建键名
        key = '+'.join(exclude_vars)
        
        # 训练和预测
        Ypred_mean, ensemble_predictions = train_and_forecast(TS, key)
        
        # 存储结果
        results[key] = ensemble_predictions
        results_mean[key] = Ypred_mean
    
    return results, results_mean

def dimension_addition_xro_forecast(ds, tl, xro_model, retain_var=['Nino34', 'WWV'],
                                    maskb=['Nino34', 'IOD'], n_month=19):
    """
    XRO维度增加预测实验：retain_var + X（包含基线 retain_var）
    
    参数:
    - ds: 数据集
    - tl: 训练时长（时间片段）
    - xro_model: XRO模型实例
    - retain_var: 保持的初始变量列表（基线），默认 ['Nino34', 'WWV']
    - maskb: 掩码列表
    - n_month: 预测月数
    
    返回:
    - results: dict of arrays, 各组合的预测结果 (n_samples, n_vars, steps)
        键: 'baseline', 'NPMM', 'SPMM', ...
    """
    # 获取所有变量
    all_vars = list(ds.data_vars)
    
    # 检查 retain_var 是否都在数据集中
    for var in retain_var:
        if var not in all_vars:
            raise ValueError(f"{var} not found in dataset")
    
    # 获取其他变量（除了 retain_var 中的变量）
    other_vars = [var for var in all_vars if var not in retain_var]
    
    print(f"\nXRO Model - Baseline: {'+'.join(retain_var)}")
    print(f"Testing: {'+'.join(retain_var)} + {other_vars}\n")
    
    def run_forecast_xro(combo_vars):
        """运行单个组合的XRO预测"""
        combo_ds = ds[combo_vars]
        train_ds = combo_ds.sel(time=tl)
        fit_result = xro_model.fit_matrix(train_ds, maskb=maskb)
        
        test_ds = combo_ds.isel(time=slice(len(train_ds.time), None))
        forecast_ds = xro_model.reforecast(fit_ds=fit_result, init_ds=test_ds, 
                                         n_month=n_month, ncopy=1, noise_type='zero')
        
        Ypred = forecast_ds.to_array().values.transpose(1, 0, 2)
        
        return Ypred
    
    # 初始化结果字典
    results = {}
    
    # 运行基线
    print(f"Running baseline ({'+'.join(retain_var)})...")
    results['baseline'] = run_forecast_xro(retain_var)
    print(f"✓ Baseline completed (dim={len(retain_var)})")
    
    # 运行各组合
    print(f"\nTesting {len(other_vars)} combinations...")
    for x_var in tqdm(other_vars, desc='Testing combinations'):
        results[x_var] = run_forecast_xro(retain_var + [x_var])
    
    return results

def dimension_decoupling_xro_forecast(ds, tl, xro_model, base_vars=['Nino34', 'WWV'],
                                     exclude_dims=1, maskb=['Nino34', 'IOD'], n_month=19,
                                     include_baseline=True):
    """
    XRO模式解耦预测实验：移除指定数量的变量
    
    类似DESN的decoupling实验：
    - baseline: 所有变量
    - exclude 1 var: 逐一移除单个变量
    - exclude 2 vars: 移除所有可能的2变量组合
    - ...
    
    参数:
    - ds: 数据集
    - tl: 训练时长（时间片段）
    - xro_model: XRO模型实例
    - base_vars: 基础变量（始终保留，不参与移除）
    - exclude_dims: 要移除的变量数量（1到可移除变量总数）
    - maskb: 掩码列表
    - n_month: 预测月数
    - include_baseline: 是否包含baseline（不移除任何变量）
    
    返回:
    - results: dict of arrays, 各组合的预测结果
        键: 'baseline', 'NPMM', 'NPMM+SPMM', ...
        移除1维时，键为变量名（如'NPMM'）
        移除2维时，键为'变量名1+变量名2'（如'NPMM+SPMM'）
    """
    from itertools import combinations
    
    # 获取所有变量
    all_vars = list(ds.data_vars)
    
    # 检查base_vars是否都在数据集中
    for var in base_vars:
        if var not in all_vars:
            raise ValueError(f"{var} not found in dataset")
    
    # 获取可移除的变量（除了base_vars）
    decouple_vars = [v for v in all_vars if v not in base_vars]
    
    # 检查排除维度数的合理性
    if exclude_dims < 1 or exclude_dims > len(decouple_vars):
        raise ValueError(f"exclude_dims must be between 1 and {len(decouple_vars)}")
    
    print(f"\nXRO Model - Mode Decoupling Experiment")
    print(f"Base variables (always retained): {'+'.join(base_vars)}")
    print(f"Decoupling pool: {decouple_vars}")
    print(f"Excluding {exclude_dims} variable(s) at a time\n")
    
    def run_forecast_xro(combo_vars):
        """运行单个组合的XRO预测"""
        combo_ds = ds[combo_vars]
        train_ds = combo_ds.sel(time=tl)
        fit_result = xro_model.fit_matrix(train_ds, maskb=maskb)
        
        test_ds = combo_ds.isel(time=slice(len(train_ds.time), None))
        forecast_ds = xro_model.reforecast(fit_ds=fit_result, init_ds=test_ds, 
                                         n_month=n_month, ncopy=1, noise_type='zero')
        
        Ypred = forecast_ds.to_array().values.transpose(1, 0, 2)
        
        return Ypred
    
    # 初始化结果字典
    results = {}
    
    # 运行基线（如果需要）
    if include_baseline:
        print(f"Running baseline ({'+'.join(all_vars)})...")
        results['baseline'] = run_forecast_xro(all_vars)
        print(f"✓ Baseline completed (dim={len(all_vars)})")
    
    # 生成所有可能的移除组合
    exclude_combinations = list(combinations(decouple_vars, exclude_dims))
    
    print(f"\nTesting {len(exclude_combinations)} combinations (excluding {exclude_dims} variable(s))...")
    
    # 遍历每个移除组合
    for exclude_vars in tqdm(exclude_combinations, desc=f'Exclude {exclude_dims} var(s)'):
        # 保留的变量（base_vars + 未被移除的decouple_vars）
        retained_vars = base_vars + [v for v in decouple_vars if v not in exclude_vars]
        
        # 构建键名：移除的变量名用'+'连接
        key = '+'.join(exclude_vars)
        
        # 运行预测
        results[key] = run_forecast_xro(retained_vars)
    
    print(f"\n✓ Completed {len(results)} experiments")
    if include_baseline:
        print(f"  - 1 baseline (dim={len(all_vars)})")
    print(f"  - {len(exclude_combinations)} decoupling tests (dim={len(all_vars)-exclude_dims} each)")
    
    return results

def calculate_ensemble_skill(results, Ytest, results_mean=None, wl=0, 
                             lower_percentile=2.5, upper_percentile=97.5,
                             showdim=0, ismv3=True, return_members=False):
    """
    计算集合预测的技巧指标（相关系数R和RMSE）
    
    参数:
    - results: dict of arrays, 各组合的所有成员预测 (n_members, n_samples, n_vars, steps)
    - Ytest: 真实值数组 (n_samples, n_vars, ...)
    - results_mean: dict of arrays, 各组合的集合平均预测 (可选)
                    如果为None，则从results计算集合平均
    - wl: warmup length for skill calculation
    - lower_percentile: 下界百分位数，默认2.5
    - upper_percentile: 上界百分位数，默认97.5
    - showdim: 用于计算skill的维度
    - ismv3: 是否使用mv3格式
    - return_members: 是否返回所有成员的技巧指标，默认False
    
    返回:
    - R: dict, 每个模式的相关系数 
        {'mode': {'avg': ..., 'lower': ..., 'upper': ..., ['members': ...]}}
    - rmse: dict, 每个模式的RMSE 
        {'mode': {'avg': ..., 'lower': ..., 'upper': ..., ['members': ...]}}
    """
    R = {}
    rmse = {}
    
    for mode in results.keys():
        ensemble_predictions = results[mode]  # (n_members, n_samples, n_vars, steps)
        n_members = ensemble_predictions.shape[0]
        
        # 1. 计算或获取集合平均
        if results_mean is not None and mode in results_mean:
            # 使用提供的集合平均
            ensemble_mean = results_mean[mode]
        else:
            # 从results计算集合平均
            ensemble_mean = np.mean(ensemble_predictions, axis=0)  # (n_samples, n_vars, steps)
        
        # 2. 计算集合平均的技巧
        R_avg, rmse_avg = ndforecast_skill(
            ensemble_mean, Ytest, 
            wl=wl, showdim=showdim, ismv3=ismv3, plot=False
        )
        
        # 3. 计算每个成员的技巧
        R_members = []
        rmse_members = []
        
        for member_idx in range(n_members):
            member_pred = ensemble_predictions[member_idx]  # (n_samples, n_vars, steps)
            R_member, rmse_member = ndforecast_skill(
                member_pred, Ytest,
                wl=wl, showdim=showdim, ismv3=ismv3, plot=False
            )
            R_members.append(R_member)
            rmse_members.append(rmse_member)
        
        # 转换为数组 (n_members, steps)
        R_members = np.array(R_members)
        rmse_members = np.array(rmse_members)
        
        # 4. 计算百分位数（上下界）
        R_lower = np.percentile(R_members, lower_percentile, axis=0)
        R_upper = np.percentile(R_members, upper_percentile, axis=0)
        
        rmse_lower = np.percentile(rmse_members, lower_percentile, axis=0)
        rmse_upper = np.percentile(rmse_members, upper_percentile, axis=0)
        
        # 5. 存储结果
        R[mode] = {
            'avg': R_avg,
            'lower': R_lower,
            'upper': R_upper
        }
        
        rmse[mode] = {
            'avg': rmse_avg,
            'lower': rmse_lower,
            'upper': rmse_upper
        }
        
        # 6. 可选：保存所有成员的结果
        if return_members:
            R[mode]['members'] = R_members
            rmse[mode]['members'] = rmse_members
    
    return R, rmse

def visualize_ensemble_skill(skill_dict, n_cols=3, figsize=None, 
                             focus_modes=None, show_uncertainty=True,
                             ylabel='Correlation', ylim=(0.2, 1.0), xlim=(1, 19),
                             ref_line=None, title=None):
    """
    可视化集合预测技巧结果（通用版本，支持任意数量的模式）
    
    参数:
    - skill_dict: dict, 技巧字典，支持两种格式：
        1. {'mode': {'avg': ..., 'lower': ..., 'upper': ...}}  # 带不确定性
        2. {'mode': array}  # 简单数组
    - n_cols: int, 每行显示的子图数量，默认3
    - figsize: tuple, 图形大小，None则自动计算
    - focus_modes: list, 重点关注的模式列表，None则显示全部
    - show_uncertainty: bool, 是否显示不确定性区间，默认True
    - ylabel: str, y轴标签，默认'Correlation'
    - ylim: tuple, y轴范围，默认(0.2, 1.0)
    - xlim: tuple, x轴范围，默认(1, 19)
    - ref_line: float, 参考线的y值，如0.5，None则不显示
    - title: str, 总标题，None则不显示
    
    返回:
    - fig: matplotlib figure对象
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MultipleLocator, AutoMinorLocator
    
    # 确定要显示的模式
    if focus_modes is None:
        all_modes = list(skill_dict.keys())
        # 将baseline放在最前面
        if 'baseline' in all_modes:
            all_modes.remove('baseline')
            all_modes = ['baseline'] + sorted(all_modes)
        focus_modes = all_modes
    else:
        # 确保baseline在最前面（如果存在）
        if 'baseline' not in focus_modes and 'baseline' in skill_dict:
            focus_modes = ['baseline'] + focus_modes
    
    n_modes = len(focus_modes)
    
    # 计算子图布局
    n_rows = int(np.ceil(n_modes / n_cols))
    
    # 自动计算图形大小
    if figsize is None:
        figsize = (n_cols * 3.5, n_rows * 3)
    
    # 创建子图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # 定义颜色
    color_avg = 'orangered'
    color_fill = 'orangered'
    
    for idx, mode in enumerate(focus_modes):
        ax = axes[idx]
        
        if mode not in skill_dict:
            ax.axis('off')
            continue
        
        mode_data = skill_dict[mode]
        
        # 检测数据格式
        if isinstance(mode_data, dict):
            # 格式1: {'avg': ..., 'lower': ..., 'upper': ...}
            if 'avg' in mode_data:
                skill_values = mode_data['avg']
                has_uncertainty = ('lower' in mode_data and 'upper' in mode_data)
            else:
                # 可能是其他字典格式，跳过
                ax.axis('off')
                continue
        else:
            # 格式2: 直接是数组
            skill_values = np.array(mode_data)
            has_uncertainty = False
        
        # 获取lead times
        lead_times = np.arange(len(skill_values))
        
        # 绘制主曲线
        ax.plot(lead_times, skill_values, 
               marker='o', linewidth=2.5, markersize=4,
               color=color_avg, alpha=0.85)
        
        # 绘制不确定性区间（如果有且需要显示）
        if show_uncertainty and has_uncertainty:
            ax.fill_between(
                lead_times, 
                mode_data['lower'], 
                mode_data['upper'], 
                color=color_fill, alpha=0.2
            )
        
        # 添加参考线
        if ref_line is not None:
            ax.axhline(y=ref_line, color='black', linestyle='--', linewidth=1.5)
        
        # 设置标题
        if mode == 'baseline':
            title_text = 'Baseline'
        else:
            title_text = mode
        ax.set_title(title_text, fontsize=14, pad=10)
        
        # 设置y轴标签（只在第一列显示）
        if idx % n_cols == 0:
            ax.set_ylabel(ylabel, fontsize=12)
        else:
            ax.set_ylabel('')
        
        # 设置x轴标签（只在最后一行显示）
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Lead Time (months)', fontsize=11)
        
        # 设置坐标轴范围
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(1, len(lead_times) - 1)
        
        if ylim is not None:
            ax.set_ylim(ylim)
        
        # 设置主刻度
        if xlim is not None:
            ax.set_xticks(np.arange(xlim[0], xlim[1] + 1, 2))
        else:
            ax.set_xticks(np.arange(1, len(lead_times), 2))
        
        if ylim is not None:
            ax.set_yticks(np.arange(ylim[0], ylim[1] + 0.01, 0.1))
        
        # 设置次刻度
        ax.xaxis.set_minor_locator(MultipleLocator(1))  # x轴次刻度间隔为1
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))  # y轴次刻度间隔为0.1
        
        # 隐藏顶部和右侧轴线
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 设置刻度大小
        ax.tick_params(labelsize=10)
        ax.tick_params(which='minor', length=3)  # 次刻度长度
        ax.tick_params(which='major', length=5)  # 主刻度长度
    
    # 隐藏多余的子图
    for idx in range(n_modes, len(axes)):
        axes[idx].axis('off')
    
    # 设置总标题
    if title is not None:
        plt.suptitle(title, fontsize=16, y=0.995)
    
    plt.tight_layout()
    
    return fig

def visualize_skill_vs_baseline(skill_dict, n_cols=3, figsize=None,
                                focus_modes=None, ylabel='Correlation',
                                ylim=(0.2, 1.0), xlim=(1, 19), ref_line=None,
                                title=None, baseline_color='deepskyblue',
                                mode_color='orangered',legend_loc='upper right'):
    """
    可视化各模式与baseline的对比（每个子图显示一个模式vs baseline）
    
    参数:
    - skill_dict: dict, 技巧字典，支持两种格式：
        1. {'mode': {'avg': ..., 'lower': ..., 'upper': ...}}  # 带不确定性
        2. {'mode': array}  # 简单数组
        必须包含 'baseline' 键
    - n_cols: int, 每行显示的子图数量，默认3
    - figsize: tuple, 图形大小，None则自动计算
    - focus_modes: list, 重点关注的模式列表（不包括baseline），None则显示全部
    - ylabel: str, y轴标签，默认'Correlation'
    - ylim: tuple, y轴范围，默认(0.2, 1.0)
    - xlim: tuple, x轴范围，默认(1, 19)
    - ref_line: float, 参考线的y值，如0.5，None则不显示
    - title: str, 总标题，None则不显示
    - baseline_color: str, baseline曲线颜色，默认'gray'
    - mode_color: str, 模式曲线颜色，默认'orangered'
    - baseline_style: str, baseline线型，默认'--'
    - mode_style: str, 模式线型，默认'-'
    
    返回:
    - fig: matplotlib figure对象
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MultipleLocator
    
    # 检查是否有baseline
    if 'baseline' not in skill_dict:
        raise ValueError("skill_dict must contain 'baseline' key")
    
    # 获取baseline数据
    baseline_data = skill_dict['baseline']
    if isinstance(baseline_data, dict) and 'avg' in baseline_data:
        baseline_values = baseline_data['avg']
    else:
        baseline_values = np.array(baseline_data)
    
    # 确定要显示的模式（排除baseline）
    if focus_modes is None:
        all_modes = [key for key in skill_dict.keys() if key != 'baseline']
        focus_modes = all_modes
    
    n_modes = len(focus_modes)
    
    # 计算子图布局
    n_rows = int(np.ceil(n_modes / n_cols))
    
    # 自动计算图形大小
    if figsize is None:
        figsize = (n_cols * 3.5, n_rows * 3)
    
    # 创建子图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # 获取lead times
    lead_times = np.arange(len(baseline_values))
    
    for idx, mode in enumerate(focus_modes):
        ax = axes[idx]
        
        if mode not in skill_dict:
            ax.axis('off')
            continue
        
        mode_data = skill_dict[mode]
        
        # 检测数据格式
        if isinstance(mode_data, dict):
            # 格式1: {'avg': ..., 'lower': ..., 'upper': ...}
            if 'avg' in mode_data:
                mode_values = mode_data['avg']
            else:
                ax.axis('off')
                continue
        else:
            # 格式2: 直接是数组
            mode_values = np.array(mode_data)
        
        # 绘制baseline
        ax.plot(lead_times, baseline_values, linewidth=2.5, markersize=5,
               color=baseline_color, label='DESN', alpha=0.85,
               marker='o')
        
        # 绘制当前模式（橙红色实线）
        ax.plot(lead_times, mode_values, linewidth=2.5, markersize=7,
               color=mode_color, label=f'$D_{{{mode}}}$', alpha=0.85, marker='x')
        
        # 添加参考线
        if ref_line is not None:
            ax.axhline(y=ref_line, color='black', linestyle='--', linewidth=1.5)
        
        # 设置标题
        ax.set_title(f'Effect of {mode} Decoupling', fontsize=14, pad=10)
        
        # 设置y轴标签（只在第一列显示）
        if idx % n_cols == 0:
            ax.set_ylabel(ylabel, fontsize=12)
        else:
            ax.set_ylabel('')
        
        # 设置x轴标签（只在最后一行显示）
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Lead Time (months)', fontsize=11)
        
        # 设置坐标轴范围
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(1, len(lead_times) - 1)
        
        if ylim is not None:
            ax.set_ylim(ylim)
        
        # 设置主刻度
        if xlim is not None:
            ax.set_xticks(np.arange(xlim[0], xlim[1] + 1, 2))
        else:
            ax.set_xticks(np.arange(1, len(lead_times), 2))
        
        if ylim is not None:
            ax.set_yticks(np.arange(ylim[0], ylim[1] + 0.01, 0.1))
        
        # 设置次刻度
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        
        # 添加图例
        ax.legend(loc=legend_loc, fontsize=12, framealpha=0.9)
        
        # 隐藏顶部和右侧轴线
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 设置刻度大小
        ax.tick_params(labelsize=10)
        ax.tick_params(which='minor', length=3)
        ax.tick_params(which='major', length=5)
    
    # 隐藏多余的子图
    for idx in range(n_modes, len(axes)):
        axes[idx].axis('off')
    
    # 设置总标题
    if title is not None:
        plt.suptitle(title, fontsize=16, y=0.995)
    
    plt.tight_layout()
    
    return fig

def visualize_skill_comparison_vs_baseline(
    comparison_skill,
    n_cols=3,
    figsize=None,
    ylabel='Correlation',
    ylim=(0.2, 1.0),
    xlim=(1, 19),
    ref_line=None,
    title=None,
    subtitle_template='$D_{{{mode}}}$',
    colors=None,
    markers=None,
    show_difference=False,
    diff_reverse=False,
    show_errorbar=False,
    show_baseline=True,
):
    """
    Visualize and compare prediction skill across multiple models.

    Parameters
    ----------
    comparison_skill : dict
        Nested dict of {model_name: skill_dict}, where skill_dict maps
        mode names (str) to either:
          - dict with keys 'avg', and optionally 'lower'/'upper' (confidence bounds)
          - array-like of skill values
        A special key 'baseline' is plotted as a dedicated first subplot when
        show_baseline=True.
    n_cols : int
        Number of subplot columns. Default 3.
    figsize : tuple or None
        Figure size (width, height). Auto-computed if None.
    ylabel : str
        Y-axis label. Default 'Correlation'.
    ylim : tuple
        Y-axis limits. Default (0.2, 1.0).
    xlim : tuple
        X-axis limits in lead-time months. Default (1, 19).
    ref_line : float or None
        Y-value of a horizontal reference line. Omitted if None.
    title : str or None
        Overall figure title. Omitted if None.
    subtitle_template : str
        Format string for subplot titles; '{mode}' is replaced by the mode name.
        Default '$D_{{{mode}}}$'. Example: '$A_{{{mode}}}$'.
    colors : list or None
        Per-model line colors. Cycles if fewer than number of models.
    markers : list or None
        Per-model markers. Cycles if fewer than number of models.
    show_difference : bool
        Whether to draw filled difference curves. Default False.
    diff_reverse : bool
        If False (default), difference is baseline - model (positive = baseline better).
        If True, difference is model - baseline (positive = model better).
    show_errorbar : bool
        Whether to draw confidence bound error bars. Default False.
    show_baseline : bool
        Whether to add a dedicated first subplot for baseline curves. Default True.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MultipleLocator

    def _extract(data):
        """Parse skill entry; return (values, lower, upper) as arrays."""
        if isinstance(data, dict):
            vals  = np.array(data['avg'])
            lower = np.array(data['lower']) if 'lower' in data else None
            upper = np.array(data['upper']) if 'upper' in data else None
        else:
            vals, lower, upper = np.array(data), None, None
        return vals, lower, upper

    def _plot_with_errbar(ax, x, vals, lower, upper, show_eb=False, **kwargs):
        """Plot line; optionally add error bars when bounds are available."""
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
        """Apply common axis decorations."""
        if ref_line is not None:
            ax.axhline(ref_line, color='black', linestyle='--', linewidth=1.5, zorder=0)
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

    # ── layout ───────────────────────────────────────────────────────────────
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

    # ── subplot 0: baseline panel ─────────────────────────────────────────────
    if show_baseline:
        ax = axes[0]
        ax_offset = 1
        for m_idx, m_name in enumerate(model_names):
            bl_data = comparison_skill[m_name].get('baseline')
            if bl_data is None:
                continue
            vals, lower, upper = _extract(bl_data)
            _plot_with_errbar(ax, np.arange(len(vals)), vals, lower, upper,
                              show_eb=show_errorbar,
                              label=m_name,
                              marker=markers[m_idx % len(markers)],
                              linewidth=2.5, markersize=5,
                              color=colors[m_idx % len(colors)],
                              alpha=0.85, zorder=2)
        _decorate_ax(ax, 0, n_rows, n_cols, 'Baseline')

    # ── mode subplots ─────────────────────────────────────────────────────────
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
                for mn in model_names if mode in comparison_skill[mn]
            )
            vals  = vals[:min_len]
            lower = lower[:min_len] if lower is not None else None
            upper = upper[:min_len] if upper is not None else None
            x      = np.arange(min_len)
            color  = colors[m_idx % len(colors)]
            marker = markers[m_idx % len(markers)]

            _plot_with_errbar(ax, x, vals, lower, upper,
                              show_eb=show_errorbar,
                              label=m_name,
                              marker=marker, linewidth=2.5, markersize=5,
                              color=color, alpha=0.85, zorder=2)

            # -- difference fill --
            if show_difference:
                bl_data = comparison_skill[m_name].get('baseline')
                if bl_data is not None:
                    bl_vals = _extract(bl_data)[0][:min_len]
                    diff    = (vals - bl_vals) if diff_reverse else (bl_vals - vals)
                    diff_curve = ylim[0] + diff
                    ax.plot(x, diff_curve,
                            linestyle='--', linewidth=1.5,
                            color=color, alpha=0.7, zorder=2)
                    ax.fill_between(x, ylim[0], diff_curve,
                                    color=color, alpha=0.2, zorder=1)
                    if m_idx == 0:
                        ax.axhline(ylim[0], color='gray', linestyle='-',
                                   linewidth=0.8, alpha=0.3, zorder=0)

        _decorate_ax(ax, ax_idx, n_rows, n_cols,
                     subtitle_template.format(mode=mode))

    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    if title is not None:
        plt.suptitle(title, fontsize=16, y=0.995)

    plt.tight_layout()
    return fig

def calculate_calendar_month_skill(
    Ypred_dict,
    Ytest,
    start_time,
    showdim=0,
    steps=22,
    wl=0,
    isMv3=True
):
    """
    计算按日历月份（target month）组织的相关性和RMSE技能。
    """
    import pandas as pd
    
    start_time = pd.to_datetime(start_time)
    
    correlation_skill = {}
    rmse_skill = {}
    
    # 生成初始月份序列
    init_months = (np.arange(Ytest[wl:].shape[0]) % 12) + 1  # 1-12
    
    # 生成目标月份矩阵
    target_months = np.zeros((12, steps), dtype=int)
    for init_month in range(1, 13):
        for lead in range(steps):
            target_month = ((init_month - 1 + lead) % 12) + 1
            target_months[init_month - 1, lead] = target_month
    
    for scenario_name, Ypred in Ypred_dict.items():
        print(f"Calculating calendar month skill for: {scenario_name}")
        
        # 初始化技能矩阵为NaN
        correlation_matrix = np.full((12, steps), np.nan)
        rmse_matrix = np.full((12, steps), np.nan)
        
        # 对每个lead time计算
        for lead in range(steps):
            # 提取预测值和真实值
            if lead == 0:
                # lead=0时，预测值就是初始条件，相关性为1，RMSE为0
                y_pred = Ypred[wl:, showdim, 0]
                y_test = Ytest[wl:, showdim]
            else:
                y_pred = Ypred[wl:-lead, showdim, lead]
                y_test = Ytest[wl+lead:, showdim]
            
            init_months_valid = init_months[:len(y_pred)]
            
            # 如果需要，应用3个月滑动平均
            if isMv3:
                y_pred = pd.Series(y_pred).rolling(3, min_periods=1).mean().values
                y_test = pd.Series(y_test).rolling(3, min_periods=1).mean().values
            
            # 对每个初始月份计算技能
            for init_month in range(1, 13):
                # 筛选特定初始月份的数据
                month_mask = init_months_valid == init_month
                y_pred_month = y_pred[month_mask]
                y_test_month = y_test[month_mask]
                
                if len(y_pred_month) > 1:
                    # 计算相关系数
                    R = np.corrcoef(y_pred_month, y_test_month)[0, 1]
                    correlation_matrix[init_month - 1, lead] = R
                    
                    # 计算RMSE
                    rmse = np.sqrt(np.mean((y_test_month - y_pred_month) ** 2))
                    rmse_matrix[init_month - 1, lead] = rmse
                elif len(y_pred_month) == 1:
                    # 只有一个数据点时，无法计算相关性，但可以计算RMSE
                    rmse = np.abs(y_test_month[0] - y_pred_month[0])
                    rmse_matrix[init_month - 1, lead] = rmse
        
        correlation_skill[scenario_name] = correlation_matrix
        rmse_skill[scenario_name] = rmse_matrix
    
    return correlation_skill, rmse_skill, target_months


def plot_calendar_month_skill(
    skill_matrix,
    metric='correlation',
    title='DESN(1979-2022) - Correlation',
    figsize=(10, 6),
    vmin=None,
    vmax=None
):
    """
    绘制按日历月份组织的技能热图（斜向上排列）。
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patches as mpatches
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 创建colormap
    if metric == 'correlation':
        original_cmap = plt.cm.coolwarm
        red_colors = original_cmap(np.linspace(0.5, 1, 256))
        white = np.array([1, 1, 1, 1])
        colors = np.vstack([np.linspace(white, red_colors[0], 32), red_colors])
        cmap = LinearSegmentedColormap.from_list("white_to_red", colors)
        vmin = vmin or 0
        vmax = vmax or 1
    else:
        cmap = plt.cm.YlOrRd
        vmin = vmin or 0
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # 创建并填充扩展矩阵
    steps = skill_matrix.shape[1]
    extended_matrix = np.full((12, steps + 11), np.nan)
    
    for init_month in range(12):
        for lead in range(steps):
            row_idx = 11 - init_month
            col_idx = lead + init_month
            if col_idx < extended_matrix.shape[1]:
                extended_matrix[row_idx, col_idx] = skill_matrix[init_month, lead]
    
    # 绘制热图
    im = ax.imshow(extended_matrix, aspect='auto', cmap=cmap, 
                   vmin=vmin, vmax=vmax, interpolation='nearest')
    
    # 添加边框和遮罩
    for i in range(12):
        actual_init_month = 11 - i
        for j in range(extended_matrix.shape[1]):
            lead_time = j - actual_init_month
            
            # 有效数据区域：添加黑色边框
            if 0 <= lead_time < steps:
                rect = mpatches.Rectangle((j-0.5, i-0.5), 1, 1, 
                                          linewidth=0.5, edgecolor='black',
                                          facecolor='none', zorder=12)
                ax.add_patch(rect)
            # 无效区域：添加灰色遮罩
            else:
                rect = mpatches.Rectangle((j-0.5, i-0.5), 1, 1, 
                                          linewidth=0, facecolor='lightgray', 
                                          alpha=0.9, zorder=13)
                ax.add_patch(rect)
        
        # 绘制外边界
        left_x = actual_init_month - 0.5
        right_x = min(actual_init_month + steps, extended_matrix.shape[1]) - 0.5
        
        # 左右边界的垂直线
        ax.plot([left_x, left_x], [i-0.5, i+0.5], 'k-', linewidth=1.2, zorder=14)
        ax.plot([right_x, right_x], [i-0.5, i+0.5], 'k-', linewidth=1.2, zorder=14)
        
        # 对角线段（连接到下一行）
        if i < 11:
            next_init_month = 11 - (i + 1)
            next_left_x = next_init_month - 0.5
            next_right_x = min(next_init_month + steps, extended_matrix.shape[1]) - 0.5
            ax.plot([left_x, next_left_x], [i+0.5, i+0.5], 'k-', linewidth=1.2, zorder=14)
            ax.plot([right_x, next_right_x], [i+0.5, i+0.5], 'k-', linewidth=1.2, zorder=14)
    
    # 顶部和底部边界
    ax.plot([10.5, min(10.5 + steps, extended_matrix.shape[1] - 0.5)], 
           [-0.5, -0.5], 'k-', linewidth=1.2, zorder=14)
    ax.plot([-0.5, min(-0.5 + steps, extended_matrix.shape[1] - 0.5)], 
           [11.5, 11.5], 'k-', linewidth=1.2, zorder=14)
    
    # 设置坐标轴
    ax.set_yticks(range(12))
    ax.set_yticklabels(month_names[::-1], fontsize=10)
    ax.set_ylabel('Initial time', fontsize=11)
    
    # 设置x轴刻度
    x_ticks, x_labels = [], []
    for year in range(3):
        for month_idx in [4, 7, 10, 1]:  # May, Aug, Nov, Feb
            pos = month_idx + year * 12
            if pos < extended_matrix.shape[1]:
                x_ticks.append(pos)
                x_labels.append(f"{month_names[month_idx]}$^{year}$")
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Target month')
    ax.set_xlim(-0.5, extended_matrix.shape[1] - 0.5)
    
    # 设置边框
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
    
    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric.capitalize(), fontsize=11)
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(1.5)
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()
    
    return fig, ax

def reorder_and_rename_results(results, key_order=None, remove_prefix='-'):
    """
    重新排序和重命名结果字典
    
    参数:
    - results: dict, 原始结果字典
        例如: {'-NPMM': array, '-SPMM': array, 'baseline': array, ...}
    - key_order: list, 期望的键顺序列表
        例如: ['baseline', 'NPMM', 'SPMM', 'WWV', ...]
        如果为None，则只重命名不排序
    - remove_prefix: str, 要移除的前缀，默认'-'
        如果为None，则不进行重命名
    
    返回:
    - reordered_results: dict, 重新排序和重命名后的字典
    """
    from collections import OrderedDict
    
    # 第一步：重命名（移除前缀）
    renamed_results = {}
    for key, value in results.items():
        if remove_prefix and key.startswith(remove_prefix):
            new_key = key[len(remove_prefix):]  # 移除前缀
        else:
            new_key = key
        renamed_results[new_key] = value
    
    # 第二步：排序
    if key_order is not None:
        reordered_results = OrderedDict()
        
        # 按照 key_order 的顺序添加
        for key in key_order:
            if key in renamed_results:
                reordered_results[key] = renamed_results[key]
        
        # 添加不在 key_order 中但存在于 renamed_results 的键
        for key in renamed_results:
            if key not in reordered_results:
                reordered_results[key] = renamed_results[key]
        
        return dict(reordered_results)
    else:
        return renamed_results

def calculate_ensemble_skill(results, Ytest, results_mean=None, wl=0, 
                             lower_percentile=2.5, upper_percentile=97.5,
                             showdim=0, ismv3=True, return_members=False,
                             bootstrap_size=None, random_state=None):
    """
    计算集合预测的技巧指标（相关系数R和RMSE）
    
    参数:
    - results: dict of arrays, 各组合的所有成员预测 (n_members, n_samples, n_vars, steps)
    - Ytest: 真实值数组 (n_samples, n_vars, ...)
    - results_mean: dict of arrays, 各组合的集合平均预测 (可选)
                    如果为None，则从results计算集合平均
    - wl: warmup length for skill calculation
    - lower_percentile: 下界百分位数，默认2.5
    - upper_percentile: 上界百分位数，默认97.5
    - showdim: 用于计算skill的维度
    - ismv3: 是否使用mv3格式
    - return_members: 是否返回所有成员的技巧指标，默认False
    - bootstrap_size: 每次bootstrap采样的成员数量，如果为None则使用所有成员（原逻辑）
                      bootstrap重复次数自动计算为C(n_members, bootstrap_size)
    - random_state: 随机种子，用于可重复的bootstrap采样
    
    返回:
    - R: dict, 每个模式的相关系数 
        {'mode': {'avg': ..., 'lower': ..., 'upper': ..., ['members': ...], ['bootstrap_samples': ...]}}
    - rmse: dict, 每个模式的RMSE 
        {'mode': {'avg': ..., 'lower': ..., 'upper': ..., ['members': ...], ['bootstrap_samples': ...]}}
    """
    from itertools import combinations
    
    R = {}
    rmse = {}
    
    # 设置随机种子（如果提供）
    if random_state is not None:
        np.random.seed(random_state)
    
    for mode in results.keys():
        ensemble_predictions = results[mode]  # (n_members, n_samples, n_vars, steps)
        n_members = ensemble_predictions.shape[0]
        
        # 1. 计算或获取集合平均
        if results_mean is not None and mode in results_mean:
            # 使用提供的集合平均
            ensemble_mean = results_mean[mode]
        else:
            # 从results计算集合平均
            ensemble_mean = np.mean(ensemble_predictions, axis=0)  # (n_samples, n_vars, steps)
        
        # 2. 计算集合平均的技巧
        R_avg, rmse_avg = ndforecast_skill(
            ensemble_mean, Ytest, 
            wl=wl, showdim=showdim, ismv3=ismv3, plot=False
        )
        
        # 3. 计算每个成员的技巧
        R_members = []
        rmse_members = []
        
        for member_idx in range(n_members):
            member_pred = ensemble_predictions[member_idx]  # (n_samples, n_vars, steps)
            R_member, rmse_member = ndforecast_skill(
                member_pred, Ytest,
                wl=wl, showdim=showdim, ismv3=ismv3, plot=False
            )
            R_members.append(R_member)
            rmse_members.append(rmse_member)
        
        # 转换为数组 (n_members, steps)
        R_members = np.array(R_members)
        rmse_members = np.array(rmse_members)
        
        # 4. Bootstrap采样（如果指定）
        if bootstrap_size is not None and bootstrap_size < n_members:
            # 生成所有可能的组合 C(n_members, bootstrap_size)
            all_combinations = list(combinations(range(n_members), bootstrap_size))
            n_bootstrap = len(all_combinations)
            
            print(f"Mode {mode}: Computing {n_bootstrap} bootstrap combinations (C({n_members}, {bootstrap_size}))")
            
            # 存储所有bootstrap样本的集合平均技巧
            R_bootstrap_samples = []
            rmse_bootstrap_samples = []
            
            # 遍历所有组合
            for combo_indices in all_combinations:
                # 获取当前组合的成员预测
                combo_predictions = ensemble_predictions[list(combo_indices)]  # (bootstrap_size, n_samples, n_vars, steps)
                
                # 计算该组合的集合平均
                combo_mean = np.mean(combo_predictions, axis=0)  # (n_samples, n_vars, steps)
                
                # 计算集合平均的技巧
                R_combo, rmse_combo = ndforecast_skill(
                    combo_mean, Ytest,
                    wl=wl, showdim=showdim, ismv3=ismv3, plot=False
                )
                
                R_bootstrap_samples.append(R_combo)
                rmse_bootstrap_samples.append(rmse_combo)
            
            # 转换为数组 (n_bootstrap, steps)
            R_bootstrap_samples = np.array(R_bootstrap_samples)
            rmse_bootstrap_samples = np.array(rmse_bootstrap_samples)
            
            # 从bootstrap样本中计算百分位数（上下界）
            R_lower = np.percentile(R_bootstrap_samples, lower_percentile, axis=0)
            R_upper = np.percentile(R_bootstrap_samples, upper_percentile, axis=0)
            
            rmse_lower = np.percentile(rmse_bootstrap_samples, lower_percentile, axis=0)
            rmse_upper = np.percentile(rmse_bootstrap_samples, upper_percentile, axis=0)
            
        else:
            # 使用所有成员（原逻辑）
            R_lower = np.percentile(R_members, lower_percentile, axis=0)
            R_upper = np.percentile(R_members, upper_percentile, axis=0)
            
            rmse_lower = np.percentile(rmse_members, lower_percentile, axis=0)
            rmse_upper = np.percentile(rmse_members, upper_percentile, axis=0)
        
        # 5. 存储结果
        R[mode] = {
            'avg': R_avg,
            'lower': R_lower,
            'upper': R_upper
        }
        
        rmse[mode] = {
            'avg': rmse_avg,
            'lower': rmse_lower,
            'upper': rmse_upper
        }
        
        # 6. 可选：保存所有成员的结果和bootstrap样本
        if return_members:
            R[mode]['members'] = R_members
            rmse[mode]['members'] = rmse_members
            if bootstrap_size is not None and bootstrap_size < n_members:
                # 保存所有bootstrap样本的技巧
                R[mode]['bootstrap_samples'] = R_bootstrap_samples
                rmse[mode]['bootstrap_samples'] = rmse_bootstrap_samples
                R[mode]['n_bootstrap'] = n_bootstrap
                rmse[mode]['n_bootstrap'] = n_bootstrap
    
    return R, rmse

def plot_reduction_effects(
    correlation_skill,
    scenarios_to_plot=None,
    baseline='baseline',
    figsize=(15, 12),
    vmin=-0.20,
    vmax=0.20,
    add_vertical_lines=True,
    vertical_line_months=[8, 11, 2, 5],
    suptitle='Effect of Variable Removal on ENSO Prediction Skill'
):
    """
    绘制变量削减对预测技能的影响（相对于baseline的差异）。
    
    参数:
    - correlation_skill: dict, 相关系数字典 {'baseline': array, 'NPMM': array, ...}
        每个数组 shape: (n_init_months, n_lead_times)
    - scenarios_to_plot: list, 要绘制的场景列表，None则绘制所有（除baseline外）
    - baseline: str, baseline的键名，默认'baseline'
    - figsize: tuple, 图形大小
    - vmin, vmax: float, colormap范围
    - add_vertical_lines: bool, 是否添加垂直虚线
    - vertical_line_months: list, 垂直虚线的月份位置
    - suptitle: str, 总标题
    
    返回: fig, axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
    import matplotlib.patches as mpatches
    import string
    
    # 初始化
    if scenarios_to_plot is None:
        scenarios_to_plot = [k for k in correlation_skill.keys() if k != baseline]
    
    baseline_skill = correlation_skill[baseline]
    n_scenarios = len(scenarios_to_plot)
    ncols = 3
    nrows = int(np.ceil(n_scenarios / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten() if n_scenarios > 1 else [axes]
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # 创建自定义colormap
    nodes = np.array([-0.20, -0.15, -0.10, -0.05, -0.025, 0.025, 0.05, 0.10, 0.15, 0.20])
    colors = ['#2400d8', '#5e50ff', '#b5f6ff', '#ddf9ff', '#f0fcff',
              '#fffcf0', '#fff9dd', '#ffecb5', '#ff7e5e', '#a50021']
    custom_cmap = LinearSegmentedColormap.from_list('custom_diverging', colors, N=256)
    norm = BoundaryNorm(nodes, custom_cmap.N)
    
    # 绘制每个场景
    for idx, scenario in enumerate(scenarios_to_plot):
        ax = axes[idx]
        diff_skill = baseline_skill - correlation_skill[scenario]
        
        # 创建扩展矩阵
        steps = diff_skill.shape[1]
        extended_matrix = np.full((12, steps + 11), np.nan)
        for init_month in range(12):
            for lead in range(steps):
                extended_matrix[11 - init_month, lead + init_month] = diff_skill[init_month, lead]
        
        im = ax.imshow(extended_matrix, aspect='auto', cmap=custom_cmap, 
                      norm=norm, interpolation='nearest')
        
        # 添加遮罩和边界
        for i in range(12):
            actual_init_month = 11 - i
            right_edge = actual_init_month + steps - 1
            
            # 遮罩无效区域
            for j in range(extended_matrix.shape[1]):
                if j - actual_init_month < 0 or j - actual_init_month >= steps:
                    ax.add_patch(mpatches.Rectangle((j-0.5, i-0.5), 1, 1, 
                                linewidth=0, facecolor='lightgray', alpha=0.9, zorder=13))
            
            # 绘制左右边界
            ax.plot([actual_init_month - 0.5]*2, [i - 0.5, i + 0.5], 
                   'k-', linewidth=1.2, zorder=14)
            if right_edge < extended_matrix.shape[1]:
                ax.plot([right_edge + 0.5]*2, [i - 0.5, i + 0.5], 
                       'k-', linewidth=1.2, zorder=14)
            
            # 绘制水平边界
            if i < 11:
                next_init_month = 11 - (i + 1)
                ax.plot([actual_init_month - 0.5, next_init_month - 0.5], 
                       [i + 0.5]*2, 'k-', linewidth=1.2, zorder=14)
                if right_edge < extended_matrix.shape[1]:
                    next_right_edge = next_init_month + steps - 1
                    if next_right_edge < extended_matrix.shape[1]:
                        ax.plot([right_edge + 0.5, next_right_edge + 0.5], 
                               [i + 0.5]*2, 'k-', linewidth=1.2, zorder=14)
        
        # 顶部和底部边界
        ax.plot([10.5, min(10.5 + steps, extended_matrix.shape[1] - 0.5)], 
               [-0.5, -0.5], 'k-', linewidth=1.2, zorder=14)
        ax.plot([-0.5, min(-0.5 + steps, extended_matrix.shape[1] - 0.5)], 
               [11.5, 11.5], 'k-', linewidth=1.2, zorder=14)
        
        # 垂直虚线
        if add_vertical_lines:
            for vline_month in vertical_line_months:
                if vline_month < extended_matrix.shape[1]:
                    ax.axvline(vline_month, color='cyan', linestyle='--', 
                             linewidth=1.5, alpha=0.7, zorder=15)
        
        # 设置坐标轴
        ax.set_yticks(range(12))
        ax.set_yticklabels(month_names[::-1])
        
        # x轴刻度（只在最底行显示）
        if idx >= n_scenarios - ncols or idx == n_scenarios - 1:
            x_ticks, x_labels = [], []
            for year in range(3):
                for month_idx in [4, 7, 10, 1]:
                    pos = month_idx + year * 12
                    if pos < extended_matrix.shape[1]:
                        x_ticks.append(pos)
                        x_labels.append(f"{month_names[month_idx]}$^{year}$")
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels)
            ax.set_xlabel('Target month')
        
        # 标题
        vars_list = scenario.split('+') if '+' in scenario else [scenario]
        title = f"Effect of {' + '.join(vars_list)} Decoupling \n(DESN - $D_{{{' + '.join(vars_list)}}}$)"
        ax.set_title(title, fontsize=14, pad=10)
        
        # y轴标签（只在最左列）
        if idx % ncols == 0:
            ax.set_ylabel('Initial time')
        
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.0)
    
    # 隐藏多余的子图
    for idx in range(n_scenarios, len(axes)):
        axes[idx].set_visible(False)
    
    # 总标题
    fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.98)
    
    # Colorbar
    fig.subplots_adjust(bottom=0.06, hspace=0.35, wspace=0.15)
    cbar_ax = fig.add_axes([0.25, 0.018, 0.5, 0.008])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', 
                       boundaries=nodes, ticks=nodes)
    cbar.set_label('Correlation difference', labelpad=5)
    cbar.set_ticklabels(['-0.20', '-0.15', '-0.10', '-0.05', '-0.025', 
                        '0.025', '0.05', '0.10', '0.15', '0.20'])
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(1.0)
    
    plt.tight_layout(rect=[0, 0.045, 1, 0.97])
    
    return fig, axes

def return_states(reservoir, TS, tl, trainning_part=True):
    states = []
    if trainning_part:
        states = reservoir.run(TS[:tl,])
    else:
        states = reservoir.run(TS)
    return states

def Minimal_RC_Lyaps(TS, tl=None, num_lyaps = 40,iterations=1000,norm_time=10,esn=None):
    if tl is None:
        tl = TS.shape[0]
    if esn is None:
        esn = Create_New_ESN(output_dim=TS.shape[1])
    Xtrain = TS[:tl-1,]
    Ytrain = TS[1:tl,]
    esn = esn.fit(Xtrain,Ytrain)
    W_out = esn.nodes[1].Wout
    W_res = esn.nodes[0].W
    W_in = esn.nodes[0].Win
    n = W_res.shape[0]
    m = W_out.shape[1]
    delta = orth(np.random.rand(n+m,num_lyaps))
    x_t = TS[0,:].reshape(1,m)
    Sum = 0
    le_record = np.zeros((num_lyaps, iterations // norm_time))
    for t in tqdm(range(iterations), desc="LE Iterations"):
        r_t = esn.nodes[0].run(x_t).reshape(n,)
        x_t = esn.run(x_t)
        J = compute_jacobian(W_out, W_res, r_t=r_t, x_t=x_t, W_in=W_in, alpha=1.0)
        delta = J @ delta
        if (t + 1) % norm_time == 0:
            Q, R = qr(delta, mode='economic')
            delta = Q[:, :num_lyaps]
            R_ii = np.log(np.abs(np.diag(R[:num_lyaps, :num_lyaps])))
            Sum += R_ii
            le_record[:, (t + 1) // norm_time - 1] = np.real(Sum) / (t + 1) 
    LE=np.real(Sum)/(iterations)
    return LE, le_record

def RC_Lyaps_with_Record(TS, tl=None, num_lyaps = 40,iterations=1000,norm_time=10,esn=None):
    if tl is None:
        tl = TS.shape[0]
    if esn is None:
        esn = Create_New_ESN(output_dim=TS.shape[1])
    Xtrain = TS[:tl-1,]
    Ytrain = TS[1:tl,]
    esn = esn.fit(Xtrain,Ytrain)
    W_out = esn.nodes[1].Wout
    W_res = esn.nodes[0].W
    W_in = esn.nodes[0].Win
    n = W_res.shape[0]
    m = W_out.shape[1]
    delta = orth(np.random.rand(n+m,num_lyaps))
    x_t = TS[0,:].reshape(1,m)
    Sum = 0
    le_record = np.zeros((num_lyaps, iterations // norm_time))
    R_ii_record = np.zeros((num_lyaps, iterations // norm_time, norm_time))  
    for t in tqdm(range(iterations), desc="LE Iterations"):
        r_t = esn.nodes[0].run(x_t).reshape(n,)
        x_t = esn.run(x_t)
        J = compute_jacobian(W_out, W_res, r_t=r_t, x_t=x_t, W_in=W_in, alpha=1.0)
        delta = J @ delta
        if (t + 1) % norm_time == 0:
            Q, R = qr(delta, mode='economic')
            delta = Q[:, :num_lyaps]
            R_ii = np.log(np.abs(np.diag(R[:num_lyaps, :num_lyaps])))
            Sum += R_ii
            R_ii_record[:, (t + 1) // norm_time - 1, t % norm_time] = np.real(R_ii)
            le_record[:, (t + 1) // norm_time - 1] = np.real(Sum) / (t + 1)
        else:
            _, R = qr(delta, mode='economic')
            R_ii = np.log(np.abs(np.diag(R[:num_lyaps, :num_lyaps])))
            R_ii_record[:, (t + 1) // norm_time - 1, t % norm_time] = np.real(R_ii) 
    LE=np.real(Sum)/(iterations)
    return LE, le_record, R_ii_record

def random_diagonal_matrix(n,k):
    """
    生成一个n*n的随机对角矩阵, 对角线元素的值在[-k, k]之间。
    """
    random_diagonal_elements = np.random.uniform(-1, 1, n)
    random_diagonal_matrix = k*np.diag(random_diagonal_elements)
    return random_diagonal_matrix

def random_permutation(D):
    """
    对一个对角矩阵进行随机重排。
    """
    n = D.shape[0]
    return D[np.random.permutation(n)]

def generate_random_orthogonal_matrix(n, c):
    # Step 1: Generate a random orthogonal matrix
    Q = ortho_group.rvs(n)
    # Step 2: Convert each element to either -1 or 1
    Q = np.sign(Q)
    
    return Q

def generate_ring_adjacency_matrix_random_sign(n):
    """
    生成一个大小为 n*n 的环形网络邻接矩阵，连接的边为 1 或 -1，其余为 0。

    参数:
        n (int): 矩阵的大小 (节点数量)。

    返回:
        np.ndarray: 环形网络的邻接矩阵。
    """
    # 初始化邻接矩阵为全 0
    adjacency_matrix = np.zeros((n, n), dtype=int)
    
    # 设置环形网络的连接关系
    for i in range(n):
        # 左邻和右邻的随机连接值为 1 或 -1
        adjacency_matrix[i, (i - 1) % n] = np.random.choice([1, -1])  # 左邻
        adjacency_matrix[i, (i + 1) % n] = np.random.choice([1, -1])  # 右邻
    
    return adjacency_matrix

def generate_ring_adjacency_matrix(n):
    """
    生成一个大小为 n*n 的环形网络邻接矩阵，连接的边为 1 ，其余为 0。

    参数:
        n (int): 矩阵的大小 (节点数量)。

    返回:
        np.ndarray: 环形网络的邻接矩阵。
    """
    # 初始化邻接矩阵为全 0
    adjacency_matrix = np.zeros((n, n), dtype=int)
    
    # 设置环形网络的连接关系
    for i in range(n):
        # 左邻和右邻的连接值为 1
        adjacency_matrix[i, (i - 1) % n] = -1  # 左邻
        adjacency_matrix[i, (i + 1) % n] = 1  # 右邻
    
    return adjacency_matrix

def convert_to_standard_calendar(time):
    """
    将时间变量转换为标准日历。
    支持 cftime 非标准日历时间的转换。
    
    Parameters:
        time (array-like): 时间变量(可能是 numpy.datetime64、datetime.datetime 或 cftime.datetime)。
    
    Returns:
        list: 转换为标准 datetime.datetime 的时间列表。
    """
    standard_time = []
    for t in time:
        if isinstance(t, cftime.datetime):
            # 将 cftime.datetime 转换为标准 datetime.datetime
            standard_time.append(datetime(t.year, t.month, t.day))
        elif isinstance(t, np.datetime64):
            # 将 numpy.datetime64 转换为标准 datetime.datetime
            standard_time.append(pd.to_datetime(str(t)).to_pydatetime())
        elif isinstance(t, np.float32):
            # 将 numpy.datetime64 转换为标准 datetime.datetime
            standard_time.append(pd.to_datetime(str(t)).to_pydatetime())
        elif isinstance(t, datetime):
            # 如果已经是标准 datetime.datetime，直接添加
            standard_time.append(t)
        else:
            raise ValueError(f"无法识别的时间类型: {type(t)}")
    return standard_time
def standardize_time_to_month_start(ds, time_dim='time'):
    """
    将数据集的时间维度标准化为每个月的第一天，并转换为标准日历格式。
    
    Parameters:
        ds (xarray.Dataset): 输入数据集。
        time_dim (str): 时间维度名称，默认是 "time"。
    
    Returns:
        xarray.Dataset: 时间标准化后的数据集。
    """
    # 获取时间变量
    time = ds[time_dim].values

    # 将时间转换为标准日历
    standard_time = convert_to_standard_calendar(time)

    # 将时间标准化为每个月的第一天
    standardized_time = [datetime(t.year, t.month, 1) for t in standard_time]

    # 替换时间维度
    ds = ds.assign_coords({time_dim: standardized_time})
    
    return ds


def cycle_encode(TS,TP):
    TS_TP = TS
    for i in range(TP.shape[1]):
        TS_TP = np.hstack((TS_TP,TS[:,:]*TP[:,i].reshape(-1, 1)))
    return TS_TP

def ESN_cycle_Generate(Ytest,esn,TPtest,steps=22,fitdX=False):
    x = Ytest
    Ypred = np.zeros((x.shape[0],x.shape[1],steps))
    for j in range(steps):
        if j == 0:
            Ypred[:,:,j] = x
        else:
            x = cycle_encode(x,TPtest[j-1:j+Ytest.shape[0]-1])
            if fitdX:
                x = x[:,:Ytest.shape[1]] + esn.run(x)
            else:
                x = esn.run(x)
            Ypred[:,:,j] = x
    return Ypred

def cycleRC_Forecast_Train_Test(TS, tl, dl=0,steps=22,
                                order=2,
                                omega=2*np.pi/12,
                                units=4000,sr=0.95,
                                rc_connectivity=0.14,
                                noise_rc=0.01,
                                noise_in=0.01,
                                ridge=6e-06,
                                use_raw_input=False,
                                esn=None,
                                fitdX=False,
                                seed=None):
    if esn is None:
        esn = Create_New_ESN(output_dim=TS.shape[1], 
                                    units=units,
                                    sr=sr, 
                                    rc_connectivity=rc_connectivity,
                                    noise_rc=noise_rc,
                                    noise_in=noise_in,
                                    ridge=ridge,
                                    use_raw_input=use_raw_input,
                                    seed=seed)
    TP = get_RCTP(TS,order=order,omega=omega,steps=steps)
    if not esn.fitted:
        Xtrain = cycle_encode(TS[dl:tl-1,],TP[dl:tl-1,])
        Ytrain = TS[dl+1:tl,]
        if fitdX:
            dXtrain = TS[dl+1:tl,] - TS[dl:tl-1,]
            esn = esn.fit(Xtrain,dXtrain)
        else:
            esn = esn.fit(Xtrain,Ytrain)
    Ytest = TS[tl:,]
    Ypred = ESN_cycle_Generate(Ytest,esn,TP[tl:,],steps=steps,fitdX=fitdX)
    return Ypred,Ytest,esn

def cycleRC_ReForecast_Train_Test(TS, tl=None, dl=0,steps=22,
                                order=2,
                                omega=2*np.pi/12,
                                units=4000,sr=0.95,
                                rc_connectivity=0.14,
                                noise_rc=0.01,
                                noise_in=0.01,
                                ridge=6e-06,
                                use_raw_input=False,
                                esn=None,
                                seed=None):
    if tl is None:
        tl = TS.shape[0]
    if esn is None:
        esn = Create_New_ESN(output_dim=TS.shape[1], 
                            units=units,
                            sr=sr, 
                            rc_connectivity=rc_connectivity,
                            noise_rc=noise_rc,
                            noise_in=noise_in,
                            ridge=ridge,
                            use_raw_input=use_raw_input,
                            seed=seed)
    TP = get_RCTP(TS,order=order,omega=omega,steps=steps)
    Xtrain = cycle_encode(TS[dl:tl-1,],TP[dl:tl-1,])
    Ytrain = TS[dl+1:tl,]
    esn = esn.fit(Xtrain,Ytrain)
    Ytest = TS[dl:tl,]
    Ypred = ESN_cycle_Generate(Ytest,esn,TP[dl:tl+steps,],steps=steps)
    return Ypred,Ytest,esn
def Create_ESN_List(hypers,num):
    esn_list = []
    for i in range(num):
        esn = get_esn_from_hypers(hypers)
        esn_list.append(esn)
    return esn_list

def RC_List_Train_Test(TS, hypers,tl, dl=0,steps=22,):
    esn_list = Create_ESN_List(hypers,num=steps)
    for i in tqdm(range(steps)):
        if i > 0:
            xtrain = TS[dl:tl-i,:]
            ytrain = TS[dl+i:tl,:]
            pred = esn_list[i].train(xtrain,ytrain)
    Ytest = TS[tl:,:]
    Ypred = np.zeros((Ytest.shape[0],Ytest.shape[1],steps))
    for j in range(steps):
        if j == 0:
            Ypred[:,:,j] = Ytest
        else:
            Ypred[:,:,j] = esn_list[j].run(Ytest)
    return Ypred,Ytest,esn_list

def cycleRC_list_Train_Test(TS, tl, dl=0,steps=22,esn=None,
                            order=2,omega=2*np.pi/12,):
    esn_list = []
    TP = get_RCTP(TS,order=order,omega=omega,steps=steps)
    for i in tqdm(range(0,steps)):
        if i > 0:
            esn = esn.reset()
            xtrain = cycle_encode(TS[dl:tl-i,:],TP[dl:tl-i,:])
            ytrain = TS[dl+i:tl,:]
            esn = esn.fit(xtrain,ytrain)
        esn_list.append(deepcopy(esn))
    Ytest = TS[tl:,:]
    Ypred = np.zeros((Ytest.shape[0],Ytest.shape[1],steps))
    for j in range(steps):
        if j == 0:
            Ypred[:,:,j] = Ytest
        else:
            Ypred[:,:,j] = esn_list[j].run(cycle_encode(Ytest,TP[tl:tl+Ytest.shape[0],:]))
    return Ypred,Ytest,esn_list


def lag_correlation_and_extend(data, focus_dim, max_lag=24):
    """
    计算多变量时间序列中每个变量与关注变量的滞后自相关系数，
    找到最大滞后值范围内的首次局部最小滞后值（如果没有局部最小值，则使用最小值），
    并扩展原始时间序列，同时返回滞后值列表。

    参数:
        data (numpy.ndarray): 输入的多变量时间序列，二维数组，形状为 (时间长度, 变量数)。
        focus_dim (int): 关注的变量维度（列索引，从 0 开始）。
        max_lag (int): 最大滞后值（默认为 24）。

    返回:
        tuple:
            - numpy.ndarray: 扩展后的时间序列，形状为 (时间长度, 原变量数 + 滞后变量数)。
            - list: 滞后值列表，每个变量的最优滞后值。
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("输入数据必须是一个 numpy 数组")
    if len(data.shape) != 2:
        raise ValueError("输入数据必须是一个二维数组，形状为 (时间长度, 变量数)")
    if not (0 <= focus_dim < data.shape[1]):
        raise ValueError("关注变量维度 focus_dim 超出范围")
    if max_lag <= 0:
        raise ValueError("max_lag 必须是正整数")
    
    # 时间长度和变量数
    time_length, num_vars = data.shape
    if max_lag >= time_length:
        raise ValueError("max_lag 不能超过时间序列的长度")
    
    # 关注变量的时间序列
    focus_series = data[:, focus_dim]
    
    # 存储每个变量的最优滞后值
    lag_values = []
    
    for var_idx in range(num_vars):
        if var_idx == focus_dim:
            # 对于关注变量本身，滞后值为 0
            lag_values.append(0)
            continue
        
        # 当前变量时间序列
        var_series = data[:, var_idx]
        
        # 计算滞后自相关系数
        correlations = []
        for lag in range(1, max_lag + 1):  # 滞后范围从 1 到 max_lag
            truncated_focus = focus_series[lag:]  # 截取对齐的关注变量
            truncated_var = var_series[:-lag]     # 截取对齐的当前变量
            
            # 计算皮尔逊相关系数
            corr = np.corrcoef(truncated_focus, truncated_var)[0, 1]
            correlations.append(corr)
        
        # 初始化为无效的局部最小值
        best_lag = 0  # 默认滞后值为 0
        found_local_min = False
        
        # 找到首次局部最小值的滞后点
        for lag in range(1, len(correlations) - 1):  # 遍历滞后值 (1 到 max_lag - 1)
            if correlations[lag] < correlations[lag - 1] and correlations[lag] < correlations[lag + 1]:
                best_lag = lag + 1  # 滞后值是索引 + 1
                found_local_min = True
                break
        
        # 如果没有找到局部最小值，则使用全局最小值
        if not found_local_min:
            min_lag_index = np.argmin(correlations)  # 全局最小值的索引
            best_lag = min_lag_index + 1  # 滞后值是索引 + 1
        
        lag_values.append(best_lag)
    
    # 应用滞后操作并扩展时间序列
    extended_data = data.copy()
    for var_idx, lag in enumerate(lag_values):
        lagged_series = np.zeros_like(focus_series)
        if lag == 0:
            lagged_series[:] = data[:, var_idx]
        if lag > 0:
            # 对当前变量进行滞后
            lagged_series[lag:] = data[:-lag, var_idx]
        # 将滞后序列添加到扩展数据中
        extended_data = np.hstack((extended_data, lagged_series[:, np.newaxis]))
    
    return extended_data, lag_values


def TPRCLag_Forecast_Train_Test(TS, tl, max_lag=24, steps=22, dl=0, units=4000, sr=0.95,
                               rc_connectivity=0.14,
                               noise_rc=0.01,
                               noise_in=0.01,
                               ridge=6e-06, seed=None,
                               esn=None,
                               use_raw_input=False,
                               tp_omega=2 * np.pi / 12,
                               tp_order=2,
                               tp_bias=0):
    """
    时间序列预测函数，支持滞后变量自动计算和多步预测。

    参数：
        TS (numpy.ndarray): 原始时间序列，形状为 (时间长度, 变量数)。
        tl (int): 训练数据的长度。
        max_lag (int): 最大滞后值，用于计算滞后值列表。
        steps (int): 预测的时间步数。
        dl (int): 数据延迟参数（默认 0）。
        units (int): ESN 的隐藏单元数。
        sr (float): 光谱半径。
        rc_connectivity (float): 隐藏层的稀疏连接比例。
        noise_rc (float): 隐藏层噪声。
        noise_in (float): 输入噪声。
        ridge (float): 岭回归系数。
        seed (int): 随机种子。
        esn: 已训练的 ESN 模型（可选）。
        use_raw_input (bool): 是否使用原始输入。
        tp_omega (float): 时间特征生成的频率参数。
        tp_order (int): 时间特征的阶数。
        tp_bias (float): 时间特征偏置。

    返回：
        tuple:
            - Ypred (numpy.ndarray): 预测的时间序列，形状为 (预测时间长度, 变量数, steps)。
            - Ytest (numpy.ndarray): 测试的时间序列（真实值）。
            - esn: 训练好的 ESN 模型。
            - lags (list): 自动计算的滞后值列表。
    """
    # 自动计算滞后值列表 lags
    extended_data, lags = lag_correlation_and_extend(TS, focus_dim=0, max_lag=max_lag)

    # 原始时间序列维度
    num_variables = TS.shape[1]

    # 如果未提供 ESN 模型，则创建新的 ESN
    if esn is None:
        esn = Create_New_ESN(output_dim=num_variables,
                             units=units,
                             sr=sr,
                             rc_connectivity=rc_connectivity,
                             noise_rc=noise_rc,
                             noise_in=noise_in,
                             ridge=ridge,
                             use_raw_input=use_raw_input,
                             seed=seed)

    # 生成时间特征 TP
    TP = get_RCTP(TS, steps=steps, omega=tp_omega, order=tp_order, bias=tp_bias)

    # 原始序列 + 滞后变量 + 时间特征
    TS_TP = np.hstack((extended_data, TP[:-steps]))

    # 训练数据
    Xtrain = TS_TP[:tl - 1, :]  # 输入
    Ytrain = TS[1:tl, :]        # 目标

    # 训练 ESN 模型
    esn = esn.fit(Xtrain, Ytrain, warmup=dl)

    # 测试数据
    Ytest = TS[tl:]  # 测试真实值
    x = TS_TP[tl:, :]  # 初始化测试输入

    # 初始化预测结果
    Ypred = np.zeros((x.shape[0], num_variables, steps))

    # 逐步预测
    for j in range(steps):
        if j == 0:
            # 第一步预测直接使用测试输入
            Ypred[:, :, j] = Ytest
        else:
            # 使用 ESN 预测当前时间步
            ts_prediction = esn.run(x)

            # 动态更新滞后数据
            lagged_values = np.zeros((x.shape[0], len(lags)))
            for var_idx, lag in enumerate(lags):
                if lag > 0:
                    for t in range(x.shape[0]):
                        if j - lag >= 0:  # 从预测结果中提取
                            lagged_values[t, var_idx] = Ypred[t, var_idx, j - lag]
                        else:  # 从历史数据中提取
                            lagged_values[t, var_idx] = extended_data[t - lag, var_idx]

            # 更新输入 x，包含 ts_prediction、滞后变量和时间特征
            x = np.hstack((ts_prediction, lagged_values, TP[tl + j:-steps + j]))

            # 保存当前时间步的预测
            Ypred[:, :, j] = ts_prediction

    return Ypred, Ytest, esn, lags

def ndforecast_stats_skill(Ypred_ensumble,Ytest,showdim=0,ismv3=True,wl=12,plot=False):
    # Get the nmembers of sim
    nmembers = Ypred_ensumble.shape[0]

    # Initialize the arrays for the correlation coefficient and the root mean square error
    R = {}
    rmse = {}
    R_all = []
    rmse_all = []
    
    for i in range(nmembers):
        R_sim, rmse_sim = ndforecast_skill(Ypred_ensumble[i], Ytest, showdim=showdim, ismv3=ismv3, wl=wl)
        R_all.append(R_sim)
        rmse_all.append(rmse_sim)
            
    R =  {'avg': np.mean(R_all, axis=0),
                  'lower': np.percentile(R_all, 2.5, axis=0),
                  'upper': np.percentile(R_all, 97.5, axis=0)} 
    rmse = {'avg': np.mean(rmse_all, axis=0),
                  'lower': np.percentile(rmse_all, 2.5, axis=0),
                  'upper': np.percentile(rmse_all, 97.5, axis=0)}   
    # Return the correlation coefficient and the root mean square error
    return R,rmse

def plot_several_stats_skills(skill_dict: dict, 
                             colors: dict = None, 
                             linestyles: dict = None,
                             markers: dict = None,
                             markersizes: dict = None,
                             alphas:dict = None, 
                             skill_name='Correlation',
                             title=None, 
                             steps=22,
                             start_step=1, 
                             isSmoothed=True,
                             interpolation_points=10,         
                             smooth_method='cubic',
                             legend=True,
                             xticks=None,
                             xlim=None,
                             ylim=None,
                             yticks=None,
                             grid_on=False,figsize=(10, 4)):
    """
    绘制多条技能曲线，支持平均值曲线及其 95% 置信区间。
    如果数据不包含 'avg', 'lower', 'upper' 三个键，则直接绘制传入的数据。

    Parameters:
        skill_dict (dict): 技能数据字典，每个键对应一个模型的数据，支持以下两种结构：
                           1. 包含 'avg', 'lower', 'upper' 键的字典
                           2. 单个曲线的数组或列表
        colors (dict): 每条曲线的颜色字典。
        linestyles (dict): 每条曲线的线型字典。
        markers (dict): 每条曲线的标记字典。
        skill_name (str): 技能名称，用于标题。
        title (str): 图标题。
        steps (int): 原始数据的最大步数。
        start_step (int): 开始绘制的步数。
        isSmoothed (bool): 是否平滑曲线。
        window (int): 移动平均窗口大小。
        interpolation_points (int): 每两个原始点之间插值的点数。
        smooth_method (str): 插值方法，默认 'cubic'（三次样条插值）。
                                   支持 'linear', 'quadratic', 'cubic'。
    """
    # 初始化最大值
    max_value = 0
    plt.figure(figsize=figsize)

    # 如果没有传入 colors, linestyles, markers，则使用默认样式
    if colors is None:
        colors = {key: plt.cm.tab10(i) for i, key in enumerate(skill_dict.keys())}
    if linestyles is None:
        linestyles = {key: '-' for key in skill_dict.keys()}
    if markers is None:
        markers = {key: '' for key in skill_dict.keys()}  # 默认不使用标记
    if markersizes is None:
        markersizes = {key: 10 for key in skill_dict.keys()}
    if alphas is None:
        alphas = {key: 1.0 for key in skill_dict.keys()} 

    # 保存初始的 steps 值
    initial_steps = steps

    # 用于存储图例项
    legend_elements = []

    for model_name, model_data in skill_dict.items():
        # 如果模型数据是包含 'avg', 'lower', 'upper' 的字典
        if isinstance(model_data, dict) and all(key in model_data for key in ['avg', 'lower', 'upper']):
            # 去除 NaN 值
            avg_values = np.array(model_data['avg'])[~np.isnan(model_data['avg'])]
            lower_values = np.array(model_data['lower'])[~np.isnan(model_data['lower'])]
            upper_values = np.array(model_data['upper'])[~np.isnan(model_data['upper'])]
            
            # 动态调整 steps 以适应 values 的长度
            if len(avg_values) < steps:
                steps = len(avg_values)

            # 更新最大值
            if max_value < max(upper_values):
                max_value = max(upper_values)
            
            # 提取前 `steps` 个点的数据
            x_original = np.arange(start_step, steps)  # 原始 x 轴
            y_avg = avg_values[start_step:steps]       # 平均值
            y_lower = lower_values[start_step:steps]   # 下置信区间
            y_upper = upper_values[start_step:steps]   # 上置信区间
            
            # 插值计算
            if isSmoothed:
                x_interp = np.linspace(x_original[0], x_original[-1], num=(steps - 1) * interpolation_points + steps)
                interp_avg = interp1d(x_original, y_avg, kind=smooth_method, fill_value="extrapolate")
                interp_lower = interp1d(x_original, y_lower, kind=smooth_method, fill_value="extrapolate")
                interp_upper = interp1d(x_original, y_upper, kind=smooth_method, fill_value="extrapolate")
                y_avg_interp = interp_avg(x_interp)
                y_lower_interp = interp_lower(x_interp)
                y_upper_interp = interp_upper(x_interp)
            else:
                x_interp = x_original
                y_avg_interp = y_avg
                y_lower_interp = y_lower
                y_upper_interp = y_upper
            
            # 绘制平均值曲线
            plt.plot(x_interp, y_avg_interp, 
                     color=colors[model_name], 
                     linestyle=linestyles[model_name],
                     alpha=alphas[model_name],
                     label=f'{model_name}')
            plt.plot(x_original, y_avg, 
                        color=colors[model_name], 
                        marker=markers[model_name],
                        markersize=markersizes[model_name], 
                        alpha=alphas[model_name],
                        linestyle='')  # 不使用线型
            
            # 绘制置信区间
            plt.fill_between(x_interp, y_lower_interp, y_upper_interp, 
                             color=colors[model_name], alpha=0.2, label=f'{model_name}')
            
            # 创建图例项
            legend_elements.append(Line2D([0], [0], color=colors[model_name], 
                                          linestyle=linestyles[model_name],marker=markers[model_name], markersize=markersizes[model_name],
                                          alpha=alphas[model_name], label=f'{model_name}'))
        
        # 如果模型数据是单个数组或列表（直接绘制）
        else:
            # 转换为 NumPy 数组并去除 NaN
            y_values = np.array(model_data)[~np.isnan(model_data)]
            
            # 动态调整 steps
            if len(y_values) < steps:
                steps = len(y_values)

            # 更新最大值
            if max_value < max(y_values):
                max_value = max(y_values)
            
            # 提取前 `steps` 个点
            x_original = np.arange(start_step, steps)
            y_original = y_values[start_step:steps]
            
            # 插值计算
            if isSmoothed:
                x_interp = np.linspace(x_original[0], x_original[-1], num=(steps - 1) * interpolation_points + steps)
                interp_values = interp1d(x_original, y_original, kind=smooth_method, fill_value="extrapolate")
                y_interp = interp_values(x_interp)
            else:
                x_interp = x_original
                y_interp = y_original
            
            # 绘制曲线
            plt.plot(x_interp, y_interp, 
                     color=colors[model_name], 
                     linestyle=linestyles[model_name],
                     marker='',
                     markersize=markersizes[model_name], 
                     alpha=alphas[model_name],  
                     label=model_name)
            plt.plot(x_original, y_original, 
                        color=colors[model_name], 
                        marker=markers[model_name],
                        markersize=markersizes[model_name], 
                        alpha=alphas[model_name],
                        linestyle='')  # 不使用线型
            # 创建图例项
            legend_elements.append(Line2D([0], [0], color=colors[model_name], linestyle=linestyles[model_name],marker=markers[model_name],
                                          markersize=markersizes[model_name],alpha=alphas[model_name],  label=model_name))
        
        # 重置 steps 为初始值
        steps = initial_steps
    
    # 添加参考线（适用于 correlation）
    if skill_name == 'Correlation':
        plt.axhline(y=0.5, color='black', linestyle='--', lw=1)

    # 添加网格、标题和标签
    if grid_on:
        plt.grid(alpha=0.3, which="both", linestyle="--", color='gray')
        # 添加网格线（可选）
        ax.grid(which='major', color='black', linestyle='--', linewidth=0.5)  # 主刻度网格线
        ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)  # 副刻度网格线
    if title:
        plt.title(title, fontsize=16)
    plt.xlabel('Lead time (months)', fontsize=14)
    plt.ylabel(skill_name, fontsize=14)

    # 设置右开口样式：隐藏顶部和右侧轴线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 隐藏顶部轴线
    ax.spines['right'].set_visible(False)  # 隐藏右侧轴线

    # 设置 X 轴和 Y 轴的主刻度和副刻度
    ax.xaxis.set_major_locator(MultipleLocator(2))  # 主刻度间隔 2
    if xticks is not None:
        ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_minor_locator(MultipleLocator(1))  # 副刻度间隔 1

    # 设置 Y 轴主刻度为间隔 0.5，副刻度为间隔 0.1
    ax.yaxis.set_major_locator(MultipleLocator(0.2))  # 主刻度间隔 0.2
    if yticks is not None:
        ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))  # 副刻度间隔 0.1

    

    # 设置 y 轴范围
    if ylim is not None:
        plt.ylim(ylim)
    # 设置 x 轴范围
    if xlim is not None:
        plt.xlim(xlim)


    # 添加图例
    if legend:
        plt.legend(handles=legend_elements,fontsize=12)
    plt.show()


def plot_seasonal_correlation_skill_in_dict(Ypred_sim_all, Ytest, steps=22, isMv3=True, plot=True, figsize=(10, 6)):
    
    # 计算每个模型的相关系数矩阵
    correlation_matrix_dict = {}
    for key in Ypred_sim_all.keys():
        Ypred_sim_mean = np.mean(Ypred_sim_all[key], axis=0)
        months = (np.arange(Ypred_sim_mean.shape[0]) % 12) + 1
        correlation_matrix = np.ones((12, steps))
        for j in range(0, steps):
            if j == 0:
                continue
            y_pred = Ypred_sim_mean[:-j, 0, j]
            y_test = Ytest[j:, 0]
            if isMv3:
                y_pred = pd.Series(y_pred).rolling(3, min_periods=1).mean().values
                y_test = pd.Series(y_test).rolling(3, min_periods=1).mean().values
            for month in range(1, 13):
                y_pred_month = y_pred[months[j:] == month]
                y_test_month = y_test[months[j:] == month]
                R = np.corrcoef(y_pred_month, y_test_month)[0, 1]
                correlation_matrix[month - 1, j] = R

        correlation_matrix_dict[key] = correlation_matrix
        if plot:
            # 绘制二维颜色图
            original_cmap = plt.cm.coolwarm
            red_cmap = LinearSegmentedColormap.from_list("red_coolwarm", original_cmap(np.linspace(0.5, 1, 256)))
            plt.figure(figsize=figsize)

            im = plt.imshow(correlation_matrix, aspect="auto", cmap=red_cmap, vmax=1,vmin=0, origin="lower",
                            extent=[0, steps, 1, 13])

            # 添加颜色条（仅显示正值部分）
            cbar = plt.colorbar(im, extend="both")  # 使用 extend="max" 表示仅扩展正值方向
            cbar.set_label("Correlation")

            # 设置 colorbar 的刻度范围和标签
            cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            cbar.set_ticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])

            # 调整横轴和纵轴刻度
            plt.xticks(np.arange(0, steps, 5), labels=np.arange(0, steps, 5))
            plt.yticks(np.arange(1.5, 13, 1), labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])

            # 添加斜线标记：对相关系数大于 0.5 的方格画斜线
            for i in range(12):  # 遍历月份（行）
                for j in range(steps):  # 遍历 lead time（列）
                    if correlation_matrix[i, j] > 0.5:  # 阈值为 0.5
                        x = j + 0.5  # 方格的中心 x 坐标
                        y = i + 1.5  # 方格的中心 y 坐标
                        plt.plot([x - 0.5, x + 0.5], [y - 0.5, y + 0.5], color="black", linewidth=1, linestyle='--')

            # 设置标题和标签
            plt.title(f"Seasonal correlation skill of minimal input {key} forecast")
            plt.xlabel("Lead time (months)")
            plt.ylabel("Target month")

            # 调整布局
            plt.tight_layout()
            plt.show()
    return correlation_matrix_dict

def subplots_seasonal_correlation_skill_in_dict(Ypred_sim_all, Ytest, steps=22, isMv3=True, figsize=(12, 4),sim_mean=False,title="Seasonal correlation skill of minimal model"):
    """
    绘制多模型的季节相关性技能图，并在同一张图上显示所有模型的结果。
    
    Args:
        Ypred_sim_all (dict): 包含多个模型预测结果的字典，键为模型名。
        Ytest (ndarray): 真实值。
        steps (int): 最大预测步长。
        isMv3 (bool): 是否使用滑动平均。
        figsize (tuple): 图像的大小。
    
    Returns:
        dict: 每个模型的相关性矩阵字典。
    """
    # 计算每个模型的相关系数矩阵
    correlation_matrix_dict = {}
    num_models = len(Ypred_sim_all.keys())  # 模型数量
    fig, axes = plt.subplots(1, num_models, figsize=figsize, sharey=True, gridspec_kw={'wspace': 0.1})  
    # `wspace` 控制子图之间的水平间距

    # 创建截取的 colormap（仅红色部分）
    original_cmap = plt.cm.coolwarm
    red_cmap = LinearSegmentedColormap.from_list("red_coolwarm", original_cmap(np.linspace(0.5, 1, 256)))

    for idx, key in enumerate(Ypred_sim_all.keys()):
        if sim_mean:
            Ypred_sim_mean = np.mean(Ypred_sim_all[key], axis=0)
        else:
            Ypred_sim_mean = Ypred_sim_all[key]
        months = (np.arange(Ypred_sim_mean.shape[0]) % 12) + 1
        correlation_matrix = np.ones((12, steps))
        
        # 计算相关性矩阵
        for j in range(0, steps):
            if j == 0:
                continue
            y_pred = Ypred_sim_mean[:-j, 0, j]
            y_test = Ytest[j:, 0]
            if isMv3:
                y_pred = pd.Series(y_pred).rolling(3, min_periods=1).mean().values
                y_test = pd.Series(y_test).rolling(3, min_periods=1).mean().values
            for month in range(1, 13):
                y_pred_month = y_pred[months[j:] == month]
                y_test_month = y_test[months[j:] == month]
                R = np.corrcoef(y_pred_month, y_test_month)[0, 1]
                correlation_matrix[month - 1, j] = R

        correlation_matrix_dict[key] = correlation_matrix

        # 绘制子图
        ax = axes[idx]
        im = ax.imshow(correlation_matrix, aspect="auto", cmap=red_cmap, vmax=1, vmin=0, origin="lower",
                       extent=[0, steps, 1, 13])

        # 设置子图标题
        ax.set_title(key, fontsize=12)

        # 设置横轴和纵轴刻度
        ax.set_xticks(np.arange(0, steps, 5))
        ax.set_xticklabels(np.arange(0, steps, 5))
        if idx == 0:  # 仅在第一个子图显示月份标签和 y 轴刻度
            ax.set_yticks(np.arange(1.5, 13, 1))
            ax.set_yticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
            ax.set_ylabel("Target month")

        # 添加斜线标记：对相关系数大于 0.5 的方格画斜线
        for i in range(12):  # 遍历月份（行）
            for j in range(steps):  # 遍历 lead time（列）
                if correlation_matrix[i, j] > 0.5:  # 阈值为 0.5
                    x = j + 0.5  # 方格的中心 x 坐标
                    y = i + 1.5  # 方格的中心 y 坐标
                    ax.plot([x - 0.5, x + 0.5], [y - 0.5, y + 0.5], color="black", linewidth=1, linestyle='--')

        # 设置横轴标签
        ax.set_xlabel("Lead time (months)")

    # 添加颜色条到最后一个子图旁边
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])  # [左, 下, 宽度, 高度]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical", extend="min")
    # cbar.set_label("Correlation")
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar.set_ticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])

    # 设置主标题
    if title:
        fig.suptitle(title=title, fontsize=14)

    # 调整布局
    plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.15)  # 调整子图和标题位置
    plt.show()

    return correlation_matrix_dict

def plot_heatmap(lead_time, skill_result,title='',label='',cmap='coolwarm'):

    # 提取变量名
    variable_names = list(skill_result[0].keys())
    
    # 创建一个数据框
    data = {var: [] for var in variable_names}
    
    for result in skill_result:
        for var in variable_names:
            data[var].append(result[var][lead_time])
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 设置索引为列表索引加2
    df.index = range(2, len(df) + 2)
    
    # 绘制热图
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, cmap=cmap, cbar_kws={'label': label},alpha=0.8)
    
    # 设置坐标轴标签
    plt.ylabel('dimensions')
    plt.title(title)
    
    # 显示图形
    plt.show()

def plot_several_stats_skills_with_legend(skill_dict: dict, 
                             styles=None,
                             add_hline=True, 
                             skill_name='Correlation',
                             title=None, 
                             steps=22, 
                             legend=True,
                             xticks=None,
                             xlim=None,
                             ylim=None,
                             yticks=None,
                             legend_loc='center',
                             grid_on=False,figsize=(10, 4)):
    """
    绘制多条技能曲线，支持平均值曲线及其 95% 置信区间。
    如果数据不包含 'avg', 'lower', 'upper' 三个键，则直接绘制传入的数据。

    Parameters:
        skill_dict (dict): 技能数据字典，每个键对应一个模型的数据，支持以下两种结构：
                           1. 包含 'avg', 'lower', 'upper' 键的字典
                           2. 单个曲线的数组或列表
        colors (dict): 每条曲线的颜色字典。
        linestyles (dict): 每条曲线的线型字典。
        markers (dict): 每条曲线的标记字典。
        skill_name (str): 技能名称，用于标题。
        title (str): 图标题。
        steps (int): 原始数据的最大步数。
        start_step (int): 开始绘制的步数。
        isSmoothed (bool): 是否平滑曲线。
        window (int): 移动平均窗口大小。
        interpolation_points (int): 每两个原始点之间插值的点数。
        smooth_method (str): 插值方法，默认 'cubic'（三次样条插值）。
                                   支持 'linear', 'quadratic', 'cubic'。
    """
    # 初始化最大值
    max_value = 0
    # 创建图形和子图
    fig, (ax, ax_legend) = plt.subplots(1, 2, figsize=figsize,dpi=300, gridspec_kw={"width_ratios": [4, 1]})
    # 最大步数，steps 为字典中值的最大长度
    steps = max(len(values) for values in skill_dict.values())
    x_axis = np.arange(0, steps)

    # 创建颜色迭代器（默认使用 Matplotlib 的 tab10 调色板）
    default_colors = cycle(plt.cm.tab10.colors)  # 循环 tab10 的颜色

    # 绘制每条曲线（按照 skill_dict 的顺序）
    for key, values in skill_dict.items():
        # 如果模型数据是包含 'avg', 'lower', 'upper' 的字典
        if isinstance(values, dict) and all(key in values for key in ['avg', 'lower', 'upper']):
            # 去除 NaN 值
            avg_values = np.array(values['avg'])[~np.isnan(values['avg'])]
            lower_values = np.array(values['lower'])[~np.isnan(values['lower'])]
            upper_values = np.array(values['upper'])[~np.isnan(values['upper'])]

            # 处理 NaN 值：用 np.nan 补全到最长的长度
            y_values = np.full(steps, np.nan)  # 创建一个全是 NaN 的数组
            y_values[:len(avg_values)] = avg_values   # 将有效值填充到数组中
            
            # 根据 xlim 截取对应范围的数据
            if xlim is not None:
                start, end = xlim
                x_indices = (x_axis >= start) & (x_axis < end)  # 筛选 x 范围内的数据
                x_axis_trimmed = x_axis[x_indices]
                y_values_trimmed = y_values[x_indices]
                lower_values_trimmed = lower_values[x_indices]
                upper_values_trimmed = upper_values[x_indices]
            else:
                # 如果没有传入 xlim，使用全部数据
                x_axis_trimmed = x_axis
                y_values_trimmed = y_values
                lower_values_trimmed = lower_values
                upper_values_trimmed = upper_values

            # 获取曲线样式
            style = styles.get(key, {}) if styles else {}
            color = style.get('color', next(default_colors))  # 使用迭代颜色作为默认值
            linestyle = style.get('linestyle', '-')  # 默认线型为实线
            marker = style.get('marker', '')  # 默认无标记
            linewidth = style.get('linewidth', 1.5)  # 默认线宽
            markersize = style.get('markersize', 7)  # 默认标记大小
            alpha = style.get('alpha', 1.0)  # 默认透明度
            hollowmarker = style.get('hollowmarker', False)  # 默认标记实心

            # 绘制曲线
            ax.plot(x_axis_trimmed, y_values_trimmed,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=linewidth,
                    markersize=markersize,
                    markerfacecolor='none' if hollowmarker else color, # 空心标记
                    alpha=alpha,
                    label=key)
            # 绘制置信区间
            ax.fill_between(x_axis_trimmed, lower_values_trimmed, upper_values_trimmed, 
                             color=color, alpha=0.2, label=f'{key}')


        else:
            # 处理 NaN 值：用 np.nan 补全到最长的长度
            y_values = np.full(steps, np.nan)  # 创建一个全是 NaN 的数组
            y_values[:len(values)] = values   # 将有效值填充到数组中
            
            # 根据 xlim 截取对应范围的数据
            if xlim is not None:
                start, end = xlim
                x_indices = (x_axis >= start) & (x_axis < end)  # 筛选 x 范围内的数据
                x_axis_trimmed = x_axis[x_indices]
                y_values_trimmed = y_values[x_indices]
            else:
                # 如果没有传入 xlim，使用全部数据
                x_axis_trimmed = x_axis
                y_values_trimmed = y_values

            # 获取曲线样式
            style = styles.get(key, {}) if styles else {}
            color = style.get('color', next(default_colors))  # 使用迭代颜色作为默认值
            linestyle = style.get('linestyle', '-')  # 默认线型为实线
            marker = style.get('marker', '')  # 默认无标记
            linewidth = style.get('linewidth', 1.5)  # 默认线宽
            markersize = style.get('markersize', 7)  # 默认标记大小
            alpha = style.get('alpha', 1.0)  # 默认透明度
            hollowmarker = style.get('hollowmarker', False)  # 默认标记实心

            # 绘制曲线
            ax.plot(x_axis_trimmed, y_values_trimmed,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=linewidth,
                    markersize=markersize,
                    markerfacecolor='none' if hollowmarker else color, # 空心标记
                    alpha=alpha,
                    label=key)
        
    # 主图设置
    # 添加 y=0.5 的水平虚线
    if add_hline:
        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1, label='y=0.5')

    # 添加网格、标题和标签
    if grid_on:
        ax.grid(True)
    if title:
        ax.set_title(title, fontsize=16)
    ax.set_xlabel('Lead time (months)', fontsize=14)
    ax.set_ylabel(skill_name, fontsize=14)

    # 设置 xy 轴主刻度和副刻度
    if yticks is None:
        yticks = np.arange(0, 1.01, 0.1)  # 设置主刻度
    ax.set_yticks(yticks)  
    if xticks is None:
        xticks = np.arange(1, steps + 1, 2)  # 默认每隔 2 个月一个主刻度
    ax.set_xticks(xticks)  # 设置主刻度

    # 设置 y 轴范围
    if ylim is not None:
        ax.set_ylim(ylim)
    # 设置 x 轴范围
    if xlim is not None:
        ax.set_xlim(xlim)

    # 设置右开口样式：隐藏顶部和右侧轴线
    ax.spines['top'].set_visible(False)  # 隐藏顶部轴线
    ax.spines['right'].set_visible(False)  # 隐藏右侧轴线

    # 按 styles 顺序绘制图例
    legend_elements = []
    for key in styles.keys():
        # 获取样式
        style = styles[key]
        color = style.get('color', next(default_colors))
        linestyle = style.get('linestyle', '-')
        marker = style.get('marker', '')
        linewidth = style.get('linewidth', 1.5)
        markersize = style.get('markersize', 7)
        alpha = style.get('alpha', 1.0)
        hollowmarker = style.get('hollowmarker', False)

        # 创建图例项
        legend_elements.append(Line2D([0], [0],
                                      color=color,
                                      linestyle=linestyle,
                                      marker=marker,
                                      linewidth=linewidth,
                                      markersize=markersize,
                                      markerfacecolor='none' if hollowmarker else color,
                                      alpha=alpha,
                                      label=key))

    # 右侧图例
    ax_legend.axis("off")  # 关闭右侧子图的坐标轴
    if legend:
        ax_legend.legend(handles=legend_elements, loc=legend_loc, fontsize=12)  # 在指定位置绘制图例

    # 显示图表
    plt.tight_layout()
    plt.show()

def plot_main_skills_with_bounds_and_legend(skill_dict: dict,
                                            styles: dict = None,
                                            skill_name='Skill',
                                            title=None,
                                            figsize=(12, 5),
                                            grid_on=True,
                                            legend=True,
                                            xticks=None,
                                            yticks=None,
                                            x_vline=13,
                                            xlim=None,
                                            ylim=(0, 1),
                                            add_hline=True,
                                            add_vline=False,
                                            vline_ymax=1,
                                            legend_loc="upper center",
                                            use_errorbar=False):
    """
    在主图中绘制曲线和上下界区域（若提供），并在右侧单独绘制图例。
    图例顺序严格按照 `styles` 字典的顺序，只显示曲线，不包含上下界。
    
    参数:
    - use_errorbar: bool, 是否使用errorbar绘制上下界，默认False使用fill_between
    """
    from matplotlib.ticker import MultipleLocator
    
    # 最大步数，steps 为字典中值的最大长度
    steps = max(len(values['avg'] if isinstance(values, dict) else values) 
                for values in skill_dict.values() if values is not None)
    x_axis = np.arange(0, steps)

    # 创建颜色迭代器（默认使用 Matplotlib 的 tab10 调色板）
    default_colors = cycle(plt.cm.tab10.colors)

    # 创建图形和子图
    if legend:
        fig, (ax, ax_legend) = plt.subplots(1, 2, figsize=figsize, 
                                            gridspec_kw={"width_ratios": [4, 1]})
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # 绘制每条曲线（按照 skill_dict 的顺序）
    for key, values in skill_dict.items():
        # 获取曲线样式
        style = styles.get(key, {}) if styles else {}
        color = style.get('color', next(default_colors))
        linestyle = style.get('linestyle', '-')
        marker = style.get('marker', '')
        linewidth = style.get('linewidth', 1.5)
        markersize = style.get('markersize', 7)
        alpha = style.get('alpha', 1.0)
        hollowmarker = style.get('hollowmarker', False)
        zorder = style.get('zorder', 1)

        # 如果值是字典，绘制上下界
        if isinstance(values, dict):
            avg = np.full(steps, np.nan)
            lower = np.full(steps, np.nan)
            upper = np.full(steps, np.nan)

            # 填充有效数据
            avg[:len(values['avg'])] = values['avg']
            lower[:len(values['lower'])] = values['lower']
            upper[:len(values['upper'])] = values['upper']

            # 根据 xlim 截取对应范围的数据
            if xlim is not None:
                start, end = xlim
                x_indices = (x_axis >= start) & (x_axis < end)
                x_axis_trimmed = x_axis[x_indices]
                avg = avg[x_indices]
                lower = lower[x_indices]
                upper = upper[x_indices]
            else:
                x_axis_trimmed = x_axis

            if use_errorbar:
                # 使用 errorbar 绘制上下界
                yerr_lower = avg - lower
                yerr_upper = upper - avg
                yerr = [yerr_lower, yerr_upper]
                
                # 绘制平均值曲线和误差棒
                ax.errorbar(x_axis_trimmed, avg, yerr=yerr,
                           color=color, linestyle=linestyle, marker=marker,
                           linewidth=linewidth, markersize=markersize,
                           alpha=alpha, label=key,
                           markerfacecolor=color if not hollowmarker else 'none',
                           elinewidth=1.5, capsize=3, capthick=1.5,
                           ecolor=color, zorder=zorder)
            else:
                # 使用 fill_between 绘制上下界区域
                ax.fill_between(x_axis_trimmed, lower, upper, color=color, 
                               alpha=0.5, label=None, zorder=zorder - 0.5)

                # 绘制平均值曲线
                ax.plot(x_axis_trimmed, avg, color=color, linestyle=linestyle, 
                       marker=marker, linewidth=linewidth, markersize=markersize, 
                       alpha=alpha, label=key,
                       markerfacecolor=color if not hollowmarker else 'none',
                       zorder=zorder)
        elif values is not None:
            # 如果值不是字典，直接绘制曲线
            y_values = np.full(steps, np.nan)
            y_values[:len(values)] = values

            # 根据 xlim 截取对应范围的数据
            if xlim is not None:
                start, end = xlim
                x_indices = (x_axis >= start) & (x_axis < end)
                x_axis_trimmed = x_axis[x_indices]
                y_values_trimmed = y_values[x_indices]
            else:
                x_axis_trimmed = x_axis
                y_values_trimmed = y_values

            # 绘制曲线
            ax.plot(x_axis_trimmed, y_values_trimmed, color=color, 
                   linestyle=linestyle, marker=marker, linewidth=linewidth, 
                   markersize=markersize, alpha=alpha, label=key, 
                   markerfacecolor=color if not hollowmarker else 'none',
                   zorder=zorder)

    # 主图设置
    # 添加 y=0.5 的水平虚线
    if add_hline:
        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1.5, zorder=0)
    
    # 添加 x=13 的垂直虚线
    if add_vline:
        ax.plot([x_vline, x_vline], [0, vline_ymax], color='royalblue', 
               linestyle=':', linewidth=1.5, zorder=0)

    # 添加网格、标题和标签
    if grid_on:
        ax.grid(True, zorder=0)
    if title:
        ax.set_title(title, fontsize=16)
    ax.set_xlabel('Lead time (months)', fontsize=14)
    ax.set_ylabel(skill_name, fontsize=14)

    # 设置 y 轴主刻度
    if yticks is None:
        yticks = np.arange(0, 1.01, 0.1)
    ax.set_yticks(yticks)
    
    # 设置 x 轴主刻度和次刻度
    if xticks is None:
        # 主刻度：每2个月
        xticks = np.arange(1, steps + 1, 2)
    ax.set_xticks(xticks)
    
    # 次刻度：每1个月
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    # 设置 y 轴范围
    if ylim is not None:
        ax.set_ylim(ylim)
    # 设置 x 轴范围
    if xlim is not None:
        ax.set_xlim(xlim)

    # 设置刻度样式
    ax.tick_params(axis='both', which='major', length=5, width=1.5)
    ax.tick_params(axis='both', which='minor', length=3, width=1)

    # 设置右开口样式：隐藏顶部和右侧轴线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 按 styles 顺序绘制图例
    legend_elements = []
    for key in styles.keys():
        # 获取样式
        if key in skill_dict.keys():
            style = styles[key]
            color = style.get('color', next(default_colors))
            linestyle = style.get('linestyle', '-')
            marker = style.get('marker', '')
            linewidth = style.get('linewidth', 1.5)
            markersize = style.get('markersize', 7)
            alpha = style.get('alpha', 1.0)
            hollowmarker = style.get('hollowmarker', False)

            # 创建图例项
            legend_elements.append(Line2D([0], [0],
                                        color=color,
                                        linestyle=linestyle,
                                        marker=marker,
                                        linewidth=linewidth,
                                        markersize=markersize,
                                        alpha=alpha,
                                        label=key,
                                        markerfacecolor=color if not hollowmarker else 'none'))

    # 右侧图例
    if legend:
        ax_legend.axis("off")
        ax_legend.legend(handles=legend_elements, loc=legend_loc, fontsize=12)

    # 显示图表
    plt.tight_layout()
    plt.show()

def filter_metrics(metrics, target_var='WWV'):
    """
    处理 metrics 数据，保留目标变量的上下界，其他变量的平均值转为数组并保留原键
    
    参数:
    - metrics: 字典，结构为 {'corr': {var: {'avg':..., 'lower':..., 'upper':...}}, 'rmse': ...}
    - target_var: 需要保留置信区间的变量名，默认为 'WWV'
    
    返回:
    - filtered_metrics: 字典，结构为 {
          'corr': {
              'WWV': {'avg': ..., 'lower': ..., 'upper': ...},  # 目标变量保留完整统计
              'NPMM': np.array([avg1, avg2, ...]),             # 其他变量保留键，值为平均值数组
              'EMI': np.array([...]),
              ...
          },
          'rmse': { ... }
      }
    """
    filtered_metrics = {}
    
    for metric_type in ['corr', 'rmse']:
        # 获取当前指标类型的数据（如 corr/rmse）
        metric_data = metrics.get(metric_type, {})
        filtered = {}
        
        for var, var_data in metric_data.items():
            if var == target_var:
                # 目标变量：保留完整统计信息
                filtered[var] = var_data
            else:
                # 非目标变量：提取平均值并转为数组
                avg_array = var_data['avg']  
                filtered[var] = avg_array
        
        filtered_metrics[metric_type] = filtered
    
    return filtered_metrics

def sort_dict_by_keys(dictionary, priority_keys, move_to_end=True):
    """
    根据 priority_keys 的列表和 move_to_end 参数控制排序方向：
    - move_to_end=True 时，将 priority_keys 所指定的键移动到末尾。
    - move_to_end=False 时，将 priority_keys 所指定的键移动到开头。

    Args:
        dictionary (dict): 原始字典。
        priority_keys (list): 要调整位置的键列表。
        move_to_end (bool): 控制排序方向，True 表示移动到末尾，False 表示移动到开头。

    Returns:
        dict: 排序后的字典。
    """
    # 筛选 priority_keys 中的键值对（如果这些键存在于字典中）
    priority_dict = {key: dictionary[key] for key in priority_keys if key in dictionary}

    # 筛选非 priority_keys 的键值对
    non_priority_dict = {key: value for key, value in dictionary.items() if key not in priority_keys}

    # 根据 move_to_end 参数控制顺序
    if move_to_end:
        # 将 priority_keys 对应的键值对放到末尾
        sorted_dict = {**non_priority_dict, **priority_dict}
    else:
        # 将 priority_keys 对应的键值对放到开头
        sorted_dict = {**priority_dict, **non_priority_dict}
    
    return sorted_dict

def generate_combinations_xr(ds, var, dimensions):
    """
    从 xarray.Dataset 数据集中挑选 dimensions 个变量的组合，其中关注的变量 var 必须包含在每个组合中。
    
    Parameters:
        ds (xr.Dataset): 输入数据集。
        var (str): 关注的变量，必须包含在每个组合中。
        dimensions (int): 每个组合的变量数量。
        
    Returns:
        list: 每个元素为一种包含 dimensions 个变量的新 xarray.Dataset。
    """
    # 检查 var 是否在数据集的变量中
    if var not in ds.data_vars:
        raise ValueError(f"The variable '{var}' is not in the dataset.")
    
    # 获取数据集中所有变量名
    variables = list(ds.data_vars)
    
    # 排除 var 后的其余变量
    other_variables = [v for v in variables if v != var]
    
    # 从其余变量中选出 (dimensions - 1) 个变量，与 var 组合
    combinations = itertools.combinations(other_variables, dimensions - 1)
    
    # 构造新数据集列表
    new_datasets = []
    for combo in combinations:
        selected_vars = [var] + list(combo)  # 将 var 添加到组合中
        new_ds = ds[selected_vars]  # 根据变量名选择子数据集
        new_datasets.append(new_ds)
    
    return new_datasets


def extract_variables(dim_R, var_name='WWV'):
    R_subdict = {}
    for i in range(len(dim_R)):
        R_subdict[str(i+2)+'dim']= dim_R[i][var_name]
    return R_subdict

def pack_dim_R(dim_R,dim_rmse):
    keys = dim_R[0].keys()
    length = len(dim_R[0]['WWV'])
    R_dict = {key: np.zeros((8,length)) for key in keys}
    rmse_dict = {key: np.zeros((8,length)) for key in keys}
    for key in keys:
        for i in range(len(dim_R)):
            R_dict[key][i,:] = dim_R[i][key]
            rmse_dict[key][i,:] = dim_rmse[i][key]

    R_stats = {key: {'avg': np.mean(R_dict[key], axis=0),
                    'lower': np.percentile(R_dict[key], 2.5, axis=0),
                    'upper': np.percentile(R_dict[key], 97.5, axis=0)} for key in keys}

    rmse_stats = {key: {'avg': np.mean(rmse_dict[key], axis=0),
                        'lower': np.percentile(rmse_dict[key], 2.5, axis=0),
                        'upper': np.percentile(rmse_dict[key], 97.5, axis=0)} for key in keys}

    R_stats_wwv = {
        key: {
            'avg': np.mean(R_dict[key], axis=0),
            'lower': np.percentile(R_dict[key], 0, axis=0),
            'upper': np.percentile(R_dict[key], 100, axis=0)
        } if key == 'WWV' else np.mean(R_dict[key], axis=0)
        for key in keys
    }
    # R_stats_wwv['nRO'] = xr.open_dataset('Fig\SourceData\Fig3\desn_forecast_skill_ds-0102.nc')['nRO fitted on 1950-99'].values
    rmse_stats_wwv = {
        key: {
            'avg': np.mean(rmse_dict[key], axis=0),
            'lower': np.percentile(rmse_dict[key], 0, axis=0),
            'upper': np.percentile(rmse_dict[key], 100, axis=0)
        } if key == 'WWV' else np.mean(rmse_dict[key], axis=0)
        for key in keys
    }

    R_avg = {key: np.mean(R_dict[key], axis=0) for key in keys}

    rmse_avg = {key: np.mean(rmse_dict[key], axis=0) for key in keys}

    return {'stat':[R_stats,rmse_stats],'wwv_stat':[R_stats_wwv,rmse_stats_wwv],'avg':[R_avg,rmse_avg]}

def calculate_dim_average_skill(R_dim, rmse_dim, start_dim=2):
    """
    计算每个维度的平均skill
    
    参数:
    - R_dim: 列表，元素为字典，键为变量名，值为含有该变量的序列的R平均
    - rmse_dim: 列表，元素为字典，键为变量名，值为含有该变量的序列的RMSE平均
    - start_dim: 起始维度，默认为2
    
    返回:
    - R_avg: 字典，键为"dim_n"，值为该维度所有变量的平均R
    - rmse_avg: 字典，键为"dim_n"，值为该维度所有变量的平均RMSE
    """
    R_avg = {}
    rmse_avg = {}
    
    for idx, (R_dict, rmse_dict) in enumerate(zip(R_dim, rmse_dim)):
        dim_num = start_dim + idx
        dim_key = f"{dim_num}dim"
        
        # 对该维度所有变量的skill求平均
        R_avg[dim_key] = np.mean(list(R_dict.values()), axis=0)
        rmse_avg[dim_key] = np.mean(list(rmse_dict.values()), axis=0)
    
    return R_avg, rmse_avg

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

def plot_correlation_heatmap(lead_time, skill_result, title='', label='',ylabel='',cbar=True, 
                            cmap=None, vmax=None, vmin=None, annotate=True,figsize=(5,3)):
    
    # 收集所有数据点
    all_values = []
    data = {var: [] for var in skill_result[0].keys()}
    
    for result in skill_result:
        for var in data:
            value = result[var][lead_time]
            data[var].append(value)
            all_values.append(value)
    
    # 计算动态范围
    data_min = min(all_values)
    data_max = max(all_values)
    
    # 设置颜色范围
    vmin = data_min if vmin is None else vmin
    vmax = data_max if vmax is None else vmax
    
    # 创建标准化映射
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if (vmin < 0 < vmax) else None
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    df.index = range(2, len(df) + 2)
    
    # 排序：先按列（变量）排序，再按行（维度）排序
    # 计算每列的平均值并排序列
    col_means = df.mean(axis=0)
    sorted_cols = col_means.sort_values(ascending=True).index
    df = df[sorted_cols]
    
    # 按行排序（维度从小到大）
    df = df.sort_index(ascending=True)
    
    # 生成格式化注释（保留两位小数）
    annot_df = df.applymap(lambda x: f"{x:.2f}")  
    
    # 绘制热图
    plt.figure(figsize=figsize)
    sns.heatmap(
        df, 
        annot=annot_df if annotate else False, 
        cmap="coolwarm" if cmap is None else cmap,
        cbar=cbar,
        cbar_kws={'label': label},
        annot_kws={'color': 'black'} if annotate else None,
        linewidths=1,
        linecolor='black',
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        fmt=''  
    )
    
    # 设置坐标轴边框样式
    ax = plt.gca()
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_linewidth(1)    
        spine.set_visible(True)   
    
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def multi_vars_forecast_skill_xro(ds, tl, xro_model, consern_var='Nino34',sel_dimensions=None,
                            maskb=['Nino34','IOD'], n_month=19):
    """
    使用XRO模型进行多变量预测并评估预测技能
    
    参数:
    - ds: 数据集
    - tl: 训练时长（时间片段）
    - xro_model: XRO模型实例（如 XRO(ncycle=12, ac_order=1, is_forward=True)）
    - consern_var: 关注的变量，默认为 'Nino34'
    - maskNT: 非线性温度项掩码
    - maskNH: 非线性热含量项掩码
    - n_month: 预测月数
    
    返回:
    - R_result: 相关系数结果列表
    - rmse_result: 均方根误差结果列表
    """
    R_result = []
    rmse_result = []
    if sel_dimensions==None:
        sel_dimensions = np.arange(2, len(ds.data_vars) + 1)
    for sel_dim in sel_dimensions:
        new_datasets = generate_combinations_xr(ds, var=consern_var, dimensions=sel_dim)
        
        Ypred = {}
        Ytest = {}
        R = {}
        rmse = {}
        showdim = 0
        var_names = list(ds.drop_vars(consern_var).data_vars)
        
        for new_ds in tqdm(new_datasets, desc=f'Training XRO models with {sel_dim} vars'):
            var_indices = {var: idx for idx, var in enumerate(new_ds.data_vars)}
            key_name = str(list(var_indices.keys()))
            
            # XRO训练：fit_matrix
            train_ds = new_ds.sel(time=tl)
            fit_result = xro_model.fit_matrix(train_ds, maskb=maskb)
            
            # XRO预测：reforecast
            test_ds = new_ds.isel(time=slice(len(train_ds.time), None))
            forecast_ds = xro_model.reforecast(fit_ds=fit_result, init_ds=test_ds, 
                                            n_month=n_month, ncopy=1, noise_type='zero')
            
            # 提取预测和真实值
            Ypred[key_name] = forecast_ds.to_array().values.transpose(1, 0, 2)
            Ytest[key_name] = test_ds.to_array().values.transpose(1, 0)
            
            # 计算技能
            R[key_name], rmse[key_name] = ndforecast_skill(Ypred[key_name], Ytest[key_name],
                                                        showdim=showdim, ismv3=True, wl=0)
        
        # 计算每个变量的平均技能
        R_score = {var_name: np.zeros_like(list(R.values())[0]) for var_name in var_names}
        rmse_score = {var_name: np.zeros_like(list(rmse.values())[0]) for var_name in var_names}
        
        for var_name in var_names:
            matching_keys = [key for key in R.keys() if var_name in eval(key)]
            for matching_key in matching_keys:
                R_score[var_name] += R[matching_key]
                rmse_score[var_name] += rmse[matching_key]
            R_score[var_name] = R_score[var_name] / len(matching_keys)
            rmse_score[var_name] = rmse_score[var_name] / len(matching_keys)
        
        R_result.append(R_score)
        rmse_result.append(rmse_score)
        
    
    return R_result, rmse_result, R, rmse, Ypred


def multi_vars_forecast_xro(ds, tl, xro_model, consern_var='Nino34', sel_dimensions=None,
                            maskb=['Nino34','IOD'], n_month=19):
        """
        使用XRO模型进行多变量预测并返回完整结果
        
        参数:
        - ds: 数据集
        - tl: 训练时长（时间片段）
        - xro_model: XRO模型实例
        - consern_var: 关注的变量，默认为 'Nino34'
        - sel_dimensions: 选择的维度列表
        - maskb: 掩码列表
        - n_month: 预测月数
        
        返回:
        - Ypred_dims: 列表，索引+2=维数，元素为字典，键为该维所有组合
        - Ypred_vars: 字典，键为变量名，值为列表（索引+2=维数），列表元素为字典（包含该变量的组合）
        - R_vars: 字典，键为变量名，值为列表，列表元素为该变量在各维度的平均相关系数
        - rmse_vars: 字典，键为变量名，值为列表，列表元素为该变量在各维度的平均RMSE
        - R_dim: 列表，索引+2=维数，元素为字典，键为变量名，值为含有该变量的序列的skill平均
        - rmse_dim: 列表，索引+2=维数，元素为字典，键为变量名，值为含有该变量的序列的skill平均
        """
        var_names = list(ds.drop_vars(consern_var).data_vars)
        
        if sel_dimensions is None:
            sel_dimensions = np.arange(2, len(ds.data_vars) + 1)
        
        # 初始化结果结构
        Ypred_dims = []
        Ypred_vars = {var: [] for var in var_names}
        R_vars = {var: [] for var in var_names}
        rmse_vars = {var: [] for var in var_names}
        R_dim = []
        rmse_dim = []
        
        for sel_dim in sel_dimensions:
            new_datasets = generate_combinations_xr(ds, var=consern_var, dimensions=sel_dim)
            
            # 当前维度的结果
            Ypred_dim = {}
            Ypred_vars_dim = {var: {} for var in var_names}
            R_dim_dict = {}
            rmse_dim_dict = {}
            
            for new_ds in tqdm(new_datasets, desc=f'Training XRO models with {sel_dim} vars'):
                var_indices = {var: idx for idx, var in enumerate(new_ds.data_vars)}
                combo_vars = [var for var in var_indices.keys() if var != consern_var]
                combo_vars_sorted = sorted(combo_vars)
                combo_key = '_'.join(combo_vars_sorted) if combo_vars_sorted else 'None'
                
                # XRO训练：fit_matrix
                train_ds = new_ds.sel(time=tl)
                fit_result = xro_model.fit_matrix(train_ds, maskb=maskb)
                
                # XRO预测：reforecast
                test_ds = new_ds.isel(time=slice(len(train_ds.time), None))
                forecast_ds = xro_model.reforecast(fit_ds=fit_result, init_ds=test_ds, 
                                                n_month=n_month, ncopy=1, noise_type='zero')
                
                # 提取预测和真实值
                Ypred = forecast_ds.to_array().values.transpose(1, 0, 2)
                Ytest = test_ds.to_array().values.transpose(1, 0)
                
                # 存储预测结果
                Ypred_dim[combo_key] = Ypred
                
                # 按变量存储
                if not combo_vars:
                    for var in var_names:
                        Ypred_vars_dim[var][combo_key] = Ypred
                else:
                    for var in combo_vars:
                        if var in var_names:
                            Ypred_vars_dim[var][combo_key] = Ypred
                
                # 计算技能
                R, rmse = ndforecast_skill(Ypred, Ytest, showdim=0, ismv3=True, wl=0)
                R_dim_dict[combo_key] = R
                rmse_dim_dict[combo_key] = rmse
            
            # 存储当前维度结果
            Ypred_dims.append(Ypred_dim)
            for var in var_names:
                Ypred_vars[var].append(Ypred_vars_dim[var])
            
            # 计算当前维度按变量的平均技能
            R_vars_dim = {}
            rmse_vars_dim = {}
            
            for var_name in var_names:
                matching_keys = [key for key in R_dim_dict.keys() if var_name in key or key == 'None']
                R_sum = np.zeros_like(list(R_dim_dict.values())[0])
                rmse_sum = np.zeros_like(list(rmse_dim_dict.values())[0])
                
                for matching_key in matching_keys:
                    R_sum += R_dim_dict[matching_key]
                    rmse_sum += rmse_dim_dict[matching_key]
                
                R_vars_dim[var_name] = R_sum / len(matching_keys)
                rmse_vars_dim[var_name] = rmse_sum / len(matching_keys)
            
            # 存储结果
            for var in var_names:
                R_vars[var].append(R_vars_dim[var])
                rmse_vars[var].append(rmse_vars_dim[var])
            
            R_dim.append(R_vars_dim)
            rmse_dim.append(rmse_vars_dim)
        
        return Ypred_dims, Ypred_vars, R_vars, rmse_vars, R_dim, rmse_dim

def dataset_to_time_series_array(ds):
    """
    将 xarray.Dataset 转换为形状为 [t, d] 的 NumPy 数组，
    其中 t 为时间序列长度, d 为变量数。同时返回变量名和其对应的维度信息。
    
    Parameters:
        ds (xr.Dataset): 输入的 xarray.Dataset。
        
    Returns:
        tuple:
            - np.ndarray: 形状为 [t, d] 的二维数组。
            - dict: 键为变量名，值为对应的维度信息 (原始形状)。
    """
    # 存储变量名和对应的维度信息
    var_indices = {}

    # 提取所有变量的数据
    arrays = []
    for idx,var_name in enumerate(ds.data_vars):
        # 将每个变量的数据转换为 NumPy 数组 (假设是一维时间序列)
        var_array = ds[var_name].values
        
        # 检查变量是否为一维时间序列
        if var_array.ndim != 1:
            raise ValueError(f"Variable '{var_name}' is not a 1D time series. Found shape: {var_array.shape}")
        
        # 添加到数组列表
        arrays.append(var_array)
        
        # 记录变量位于哪个维度上
        var_indices[var_name] = idx

    # 将所有变量沿第二维度拼接，形成 [t, d] 的二维数组
    combined_array = np.stack(arrays, axis=1)

    return combined_array, var_indices

def plot_several_stats_subplots(skill_dict, styles=None, add_hline=True, skill_name='Correlation',
                            variables=['WWV','NPMM','SPMM','IOB','TNA','ATL3','IOD','SIOD','SASD'],
                            title=None, steps=22, legend=True, xticks=None, xlim=None, ylim=None,
                            yticks=None, grid_on=False, figsize=(12, 8)):
    """
    绘制3x3子图，图例显示在第一个子图中
    
    Parameters:
        skill_dict (dict): 嵌套字典结构 {model_name: {var1: data, ..., var9: data}}
        styles (dict): 样式配置字典 {model: {color, linestyle, marker, ...}}
    """
    import matplotlib.gridspec as gridspec
    # 处理样式配置
    models = list(skill_dict.keys())
    default_colors = plt.cm.tab10.colors
    if styles is None:
        styles = {}
    
    # 自动分配颜色
    for idx, model in enumerate(models):
        if model not in styles:
            styles[model] = {}
        if 'color' not in styles[model]:
            color_idx = idx % len(default_colors)
            styles[model]['color'] = default_colors[color_idx]

    # 创建figure和网格布局
    fig = plt.figure(figsize=figsize, dpi=300)
    gs = gridspec.GridSpec(3, 3)  # 3x3网格
    
    # 准备图例元素
    legend_elements = [
        Line2D([0], [0],
            color=styles[model]['color'],
            lw=styles[model].get('linewidth', 1.2),
            linestyle=styles[model].get('linestyle', '-'),
            marker=styles[model].get('marker', ''),
            markersize=styles[model].get('markersize', 4),
            label=model)
        for model in models
    ]

    # 绘制子图
    for i in range(9):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        var = variables[i]
        
        # 设置子图属性
        ax.set_title(var, fontsize=12)
        ax.set_xlabel('Lead time (months)', fontsize=12) if row == 2 else None
        ax.set_ylabel(skill_name, fontsize=12) if col == 0 else None
        
        
        # 设置刻度
        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)
        ax.tick_params(axis='both', labelsize=12)
        
        # 添加参考线
        if add_hline:
            ax.axhline(0.5, color='gray', linestyle=':', lw=0.8, alpha=0.5)
        
        # 绘制数据
        for model in models:
            data = skill_dict[model].get(var)
                
            style = styles[model]
            x = np.arange(steps)
            
            # 处理数据
            if isinstance(data, dict):
                # 带置信区间的数据
                y = np.full(steps, np.nan)
                y_lower = np.full(steps, np.nan)
                y_upper = np.full(steps, np.nan)
                
                valid_data = data['avg'][~np.isnan(data['avg'])]
                y[:len(valid_data)] = valid_data
                
                # 绘制置信区间
                if 'lower' in data and 'upper' in data:
                    y_lower[:len(data['lower'])] = data['lower']
                    y_upper[:len(data['upper'])] = data['upper']
                    ax.fill_between(x, y_lower, y_upper, 
                                color=style['color'], alpha=0.15)
                
                # 绘制均值线
                ax.plot(x, y, 
                    color=style['color'],
                    linestyle=style.get('linestyle', '-'),
                    lw=style.get('linewidth', 1),
                    marker=style.get('marker', ''),
                    markersize=style.get('markersize', 3),
                    alpha=0.9)
            else:
                # 普通数据
                y = np.full(steps, np.nan)
                valid_data = data[~np.isnan(data)]
                y[:len(valid_data)] = valid_data
                ax.plot(x, y,
                    color=style['color'],
                    linestyle=style.get('linestyle', '-'),
                    lw=style.get('linewidth', 1),
                    marker=style.get('marker', ''),
                    markersize=style.get('markersize', 3),
                    alpha=0.9)
                
        # 坐标轴设置
        ax.set_xlim(xlim or (0, steps))
        ax.set_ylim(ylim or (0, 1))
        ax.spines['top'].set_visible(False)  
        ax.spines['right'].set_visible(False) 
        
        # 只在第一个子图添加图例
        if i == 0 and legend:
            ax.legend(handles=legend_elements,
                    loc='best',
                    fontsize=12,
                    frameon=False,
                    borderaxespad=0.5,
                    handlelength=1.5,
                    labelspacing=0.3)

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    if title:
        plt.suptitle(title, y=1.02, fontsize=16)
    plt.show()
