#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

import numpy as np

from .. import Utils as utils

#####################################################
# MULTISCALE MODELING
#####################################################
# from .Lorenz3D     import utils_lorenz3D     as utils_lorenz3D
from .AD import utils_processing_AD as utils_processing_AD
# from .KS             import KSGP64L22         as KSGP64L22
# from .KS             import KSGP512L100         as KSGP512L100
# from .FHN         import FHN_LB             as FHN_LB
#####################################################

######################################################
## PROCESSING UTILITIES FOR EACH SYSTEM
######################################################
from .Alanine import utils_processing_alanine as utils_processing_alanine
from .TRP import utils_processing_trp as utils_processing_trp
from .BMP import utils_processing_bmp as utils_processing_bmp
from .KS import utils_processing_ks as utils_processing_ks
from .FHN import utils_processing_fhn as utils_processing_fhn
from .CGW import utils_processing_cgw as utils_processing_cgw

######################################################
## PLOTTING UTILITIES FOR EACH SYSTEM
######################################################
from .AD import utils_plotting_AD as utils_plotting_AD
from .Alanine import utils_plotting_alanine as utils_plotting_alanine
from .TRP import utils_plotting_trp as utils_plotting_trp
from .BMP import utils_plotting_bmp as utils_plotting_bmp
from .KS import utils_plotting_ks as utils_plotting_ks
from .CGW import utils_plotting_cgw as utils_plotting_cgw


def AD3Ddatasets():
    temp = [
        "AD3D-D5",
        "AD3D-D10",
        "AD3D-D50",
        "AD3D-D100",
        "AD3D-D200",
        "AD3D-N10k-D5",
        "AD3D-N10k-D10",
        "AD3D-N10k-D50",
        "AD3D-N10k-D100",
        "AD3D-N10k-D200",
    ]
    return temp


def getSystemDataInfo(model):

    # Data are of three possible data types for each application:
    # Type 1:
    #         (T, input_dim)
    # Type 2:
    #         (T, input_dim, Cx) (input_dim is the N_particles, perm_inv, etc.)
    # Type 3:
    #         (T, input_dim, Cx, Cy, Cz)

    data_info_dict = {
        'contour_plots': False,
        'density_plots': False,
        'statistics_cummulative': False,
        'statistics_per_state': False,
        'statistics_per_channel': False,  # Not implemented
        'statistics_per_timestep': False,
        'compute_r2': False,  # Whether to compute the R2 error loss
    }

    if model.system_name in [
            "MackeyGlass",
            "MackeyGlass_SNR5",
            "MackeyGlass_SNR10",
            "MackeyGlass_SNR60",
    ]:
        assert (model.scaler == "MinMaxZeroOne")
        data_info_dict.update({
            'scaler':
            utils.scaler(
                scaler_type="MinMaxZeroOne",
                data_min=0.0,
                data_max=2.0,
                common_scaling_per_input_dim=1,
            ),
            'dt':
            1.0,
            'statistics_cummulative':
            True,
        })
        # OTHER OPTION
        # data_info_dict.update({
        # 'scaler': utils.scaler(
        #     scaler_type="MinMaxZeroOne",
        #     data_min=[0.0],
        #     data_max=[2.0],
        #     common_scaling_per_input_dim=0,
        #     ),
        # 'dt': 1.0,
        # })
    elif model.system_name in ["PolandElectricity"]:
        assert (model.scaler == "MinMaxZeroOne")
        data_info_dict.update({
            'scaler':
            utils.scaler(
                scaler_type="MinMaxZeroOne",
                data_min=0.5,
                data_max=1.4,
                common_scaling_per_input_dim=1,
            ),
            'dt':
            1.0,
            'statistics_cummulative':
            True,
        })
    elif model.system_name in ["SantaFe"]:
        assert (model.scaler == "MinMaxZeroOne")
        data_info_dict.update({
            'scaler':
            utils.scaler(
                scaler_type="MinMaxZeroOne",
                data_min=0.0,
                data_max=255.0,
                common_scaling_per_input_dim=1,
            ),
            'dt':
            1.0,
            'statistics_cummulative':
            True,
        })
    elif model.system_name in ["Leuven"]:
        assert (model.scaler == "MinMaxZeroOne")
        data_info_dict.update({
            'scaler':
            utils.scaler(
                scaler_type="MinMaxZeroOne",
                data_min=-0.7,
                data_max=0.7,
                common_scaling_per_input_dim=1,
            ),
            'dt':
            1.0,
            'statistics_cummulative':
            True,
        })
    elif model.system_name in ["Darwin"]:
        assert (model.scaler == "MinMaxZeroOne")
        data_info_dict.update({
            'scaler':
            utils.scaler(
                scaler_type="MinMaxZeroOne",
                data_min=3.0,
                data_max=15.6,
                common_scaling_per_input_dim=1,
            ),
            'dt':
            1.0,
            'statistics_cummulative':
            True,
        })
    elif model.system_name == "KSGP64L22":
        assert (model.scaler == "MinMaxZeroOne")
        data_info_dict.update({
            'scaler':
            utils.scaler(
                scaler_type="MinMaxZeroOne",
                data_min=-3.5,
                data_max=3.5,
                common_scaling_per_input_dim=1,
            ),
            'dt':
            0.5,
            'contour_plots':
            True,
            'statistics_cummulative':
            True,
        })
    elif model.system_name == "Lorenz3D":
        assert (model.scaler == "MinMaxZeroOne")
        data_info_dict.update({
            'scaler':
            utils.scaler(
                scaler_type="MinMaxZeroOne",
                data_min=[-25],
                data_max=[25],
            ),
            'dt':
            0.05,
            'statistics_per_state':
            True,
        })
    elif model.system_name in AD3Ddatasets():
        assert (model.scaler == "MinMaxZeroOne")
        data_info_dict.update({
            'scaler':
            utils.scaler(
                scaler_type="MinMaxZeroOne",
                data_min=-0.5,
                data_max=0.5,
                channels=1,
                common_scaling_per_input_dim=
                1,  # Common scaling for all particles
                common_scaling_per_channels=1,  # Common scaling for all channels
            ),
            'dt':
            1.0,
            'density_plots':
            True,
            'statistics_per_state':
            True,
            'statistics_per_timestep':
            True,
            'truncate': [100000,
                         400],  # truncate sequence_length or input dimension
        })
    elif model.system_name in ["AD1D"]:
        assert (model.scaler == "MinMaxZeroOne")
        data_info_dict.update({
            'scaler':
            utils.scaler(
                scaler_type="MinMaxZeroOne",
                data_min=-0.5,
                data_max=0.5,
                channels=1,
                common_scaling_per_input_dim=
                1,  # Common scaling for all particles
                common_scaling_per_channels=1,  # Common scaling for all channels
            ),
            'dt':
            1.0,
            'density_plots':
            True,
            'statistics_per_state':
            True,
            'statistics_per_timestep':
            True,
        })
    elif model.system_name == "CGW50":
        assert (model.scaler == "MinMaxZeroOne")
        data_info_dict.update({
            'scaler':
            utils.scaler(
                scaler_type="MinMaxZeroOne",
                data_min=[-6.0, -6.0, -6.0],
                data_max=[200.0, 200.0, 200.0],
                channels=1,
                common_scaling_per_input_dim=
                1,  # Common scaling for all particles
                common_scaling_per_channels=0,  # Common scaling for all channels
            ),
            'dt':
            1.0,
            'truncate': [
                100000, 50
            ],  # To truncate the input time series (sequence lengt) or the input dimension
            'density_plots':
            True,
            'statistics_per_state':
            True,
        })
    elif model.system_name == "CGW":
        assert (model.scaler == "MinMaxZeroOne")
        # N_particles is the input_dim (input_dimension)
        # Channels is the positions/velocities
        data_info_dict.update({
            'scaler':
            utils.scaler(
                scaler_type="MinMaxZeroOne",
                data_min=[-6.0, -6.0, -6.0],
                data_max=[200.0, 200.0, 200.0],
                channels=1,
                common_scaling_per_input_dim=
                1,  # Common scaling for all particles
                common_scaling_per_channels=0,  # Common scaling for all channels
            ),
            'dt':
            1.0,
            'density_plots':
            True,
            'statistics_per_state':
            True,
        })

    elif model.system_name == "BMP":

        assert (model.scaler == "MinMaxZeroOne")
        data_info_dict.update({
            'scaler':
            utils.scaler(
                scaler_type="MinMaxZeroOne",
                data_min=[-2.72914855, -0.73498089],
                data_max=[1.17632292, 2.84032683],
                common_scaling_per_input_dim=
                0,  # Common scaling for all particles
                common_scaling_per_channels=0,  # Common scaling for all channels
                slack=0.1,
            ),
            'dt':
            0.5,
            'density_plots':
            False,
            'statistics_per_state':
            True,
            # 'truncate': [21,50], # To truncate the input time series (sequence length) or the input dimension
        })
        if model.params["truncate_data"] == 1:
            data_info_dict.update({'truncate': [41, 50]})

    elif model.system_name == "Alanine":
        import numpy as np
        import os.path
        # Checking if the necessary files for scaler exist
        for fname in [
                model.data_path_gen + "/data_min_bonds.txt",
                model.data_path_gen + "/data_max_bonds.txt",
                model.data_path_gen + "/data_min_angles.txt",
                model.data_path_gen + "/data_max_angles.txt",
        ]:
            if not os.path.isfile(fname):
                raise ValueError(
                    "Tried to find file {:} to load the scaler. File not found."
                    .format(fname))

        assert (model.scaler == "MinMaxZeroOne")
        data_info_dict.update({
            'scaler':
            utils.scalerBAD(
                scaler_type="MinMaxZeroOne",
                dims_total=24,
                dims_bonds=9,
                dims_angles=8,
                dims_dehedrals=7,
                data_min_bonds=np.loadtxt(model.data_path_gen +
                                          "/data_min_bonds.txt"),
                data_max_bonds=np.loadtxt(model.data_path_gen +
                                          "/data_max_bonds.txt"),
                data_min_angles=np.loadtxt(model.data_path_gen +
                                           "/data_min_angles.txt"),
                data_max_angles=np.loadtxt(model.data_path_gen +
                                           "/data_max_angles.txt"),
                slack=0.05,
            ),
            'dt':
            0.1 * 10e-12,
            'density_plots':
            False,
            'statistics_per_state':
            True,
        })
    elif model.system_name == "TRP":
        import numpy as np
        import os.path
        # Checking if the necessary files for scaler exist
        for fname in [
                model.data_path_gen + "/data_min_bonds.txt",
                model.data_path_gen + "/data_max_bonds.txt",
                model.data_path_gen + "/data_min_angles.txt",
                model.data_path_gen + "/data_max_angles.txt",
        ]:
            if not os.path.isfile(fname):
                raise ValueError(
                    "Tried to find file {:} to load the scaler. File not found."
                    .format(fname))
        # ////////////////////////
        # // Columns of BAD.txt are:
        # // 1    - 153 (153 bonds)
        # // 154  - 305 (152 angles)
        # // 306  - 456 (151 dihedrals)

        assert (model.scaler == "MinMaxZeroOne")
        data_info_dict.update({
            'scaler':
            utils.scalerBAD(
                scaler_type="MinMaxZeroOne",
                dims_total=456,
                dims_bonds=153,
                dims_angles=152,
                dims_dehedrals=151,
                data_min_bonds=np.loadtxt(model.data_path_gen +
                                          "/data_min_bonds.txt"),
                data_max_bonds=np.loadtxt(model.data_path_gen +
                                          "/data_max_bonds.txt"),
                data_min_angles=np.loadtxt(model.data_path_gen +
                                           "/data_min_angles.txt"),
                data_max_angles=np.loadtxt(model.data_path_gen +
                                           "/data_max_angles.txt"),
                slack=0.05,
            ),
            'dt':
            0.1 * 10e-12,
            'density_plots':
            False,
            'statistics_per_state':
            True,
            # 'truncate': [31,10000], # To truncate the input time series (sequence length) or the input dimension
        })
        if model.params["truncate_data"] == 1:
            data_info_dict.update({'truncate': [41, 1000]})

    else:
        raise ValueError(
            "Data info for this dataset not found (see system_processing.py script)."
        )
    return data_info_dict


def computeStateDistributionStatisticsSystem(model, state_dist_statistics,
                                             targets_all, predictions_all):
    print("# computeStateDistributionStatisticsSystem() #")

    # Adding system specific state distributions (e.g. Ux-Uxx plot in Kuramoto-Sivashisnky)
    if model.system_name == "KSGP64L22":
        state_dist_statistics = utils_processing_ks.computeStateDistributionStatisticsSystemKS(
            state_dist_statistics, targets_all, predictions_all)

    if utils.isAlanineDataset(model.system_name):
        state_dist_statistics = utils_processing_alanine.computeStateDistributionStatisticsSystemAlanine(
            state_dist_statistics, targets_all, predictions_all)

    if model.system_name in ["AD1D"] or (model.system_name in AD3Ddatasets()):
        state_dist_statistics = utils_processing_AD.computeStateDistributionStatisticsSystemAD(
            state_dist_statistics, targets_all, predictions_all)

    return state_dist_statistics


def addResultsSystem(model, results, statistics, testing_mode, set_name=None):

    if utils.isAlanineDataset(model.system_name):
        results = utils_processing_alanine.addResultsSystemAlanine(model, results, statistics, testing_mode)
        # results = utils_processing_alanine.addResultsSystemAlanineMSMAnalysis(model, results, statistics, testing_mode)

        # utils_processing_alanine.transitionTimesTrain(model, n_splits=8)
        # utils_processing_alanine.transitionTimesTrainVal(model, n_splits=8)
        # utils_processing_alanine.transitionTimesPredTest(model, n_splits=8)
        # utils_processing_alanine.transitionTimesPredTrain(model, n_splits=8)

        # utils_processing_alanine.transitionTimesTrain(model, n_splits=16)
        # utils_processing_alanine.transitionTimesTrainVal(model, n_splits=16)
        # utils_processing_alanine.transitionTimesPredTest(model, n_splits=16)
        # utils_processing_alanine.transitionTimesPredTrain(model, n_splits=16)
        
        # utils_processing_alanine.transitionTimesTrain(model, n_splits=32)
        # utils_processing_alanine.transitionTimesTrainVal(model, n_splits=32)
        # utils_processing_alanine.transitionTimesPredTest(model, n_splits=32)
        # utils_processing_alanine.transitionTimesPredTrain(model, n_splits=32)
        

        # utils_processing_alanine.transitionTimesReference(model)
        # utils_processing_alanine.transitionTimesReferenceSmallDt(model)
        # utils_processing_alanine.addResultsSystemAlanineTrainingDataAnalysis(model)

        pass
    elif model.system_name == "TRP":
        results = utils_processing_trp.addResultsSystemTRP(
            model, results, statistics, testing_mode)

    elif model.system_name == "BMP":
        results = utils_processing_bmp.addResultsSystemBMP(
            model, results, statistics, testing_mode)

    elif model.system_name in ["CGW", "CGW50"]:
        results = utils_processing_cgw.addResultsSystemCGW(
            model, results, statistics, testing_mode)

    return results


def prepareMicroSolver(model, ic_idx, dt_model, set_name):
    print("# prepareMicroSolver() #")

    if model.system_name in AD3Ddatasets():
        D_str = model.system_name[6:]
        D = int(D_str) / 1000.0
        print("Diffusion: {:}".format(D))
        ic_idx_path = model.getDataPath(set_name) + "/ic_idx.txt"
        ic_idx_in_original_time_series = np.loadtxt(ic_idx_path,
                                                    dtype=float).astype(int)
        t_start = dt_model * (
            model.n_warmup - 1 +
            ic_idx_in_original_time_series[ic_idx - 1]) + dt_model
        micro_solver = utils_processing_AD.prepareSolverAD3D(D, t_start)
        return micro_solver

    if "AD1D" in model.system_name:
        D = 0.2
        ic_idx_path = model.getDataPath(set_name) + "/ic_idx.txt"
        ic_idx_in_original_time_series = np.loadtxt(ic_idx_path, dtype=float)
        t_start = dt_model * (
            model.n_warmup - 1 +
            ic_idx_in_original_time_series[ic_idx - 1]) + dt_model
        micro_solver = utils_processing_AD.prepareSolverAD1D(D, t_start)
        return micro_solver


def evolveSystem(model,
                 micro_solver,
                 initial_state,
                 tend,
                 dt_model,
                 t_jump=0.0):

    if model.system_name in AD3Ddatasets():
        u = utils_processing_AD.evolveAD(micro_solver,
                                         initial_state,
                                         tend,
                                         dt_model,
                                         t_jump=t_jump,
                                         dimension=3)

    if "AD1D" in model.system_name:
        u = utils_processing_AD.evolveAD(micro_solver,
                                         initial_state,
                                         tend,
                                         dt_model,
                                         t_jump=t_jump,
                                         dimension=1)

    # if model.system_name == "Lorenz3D":
    #     u = utils_lorenz3D.evolveLorenz3D(u0, tend, dt_model)

    # elif model.system_name == "AdvectionDiffusionReflective":
    #     u = utils_AD1D.evolveAdvectionDiffusionReflective(
    #         u0, tend, dt_model, tstart)

    # elif model.system_name == "AdvectionDiffusionReflective2D":
    #     u = utils_AD2D.evolveAdvectionDiffusionReflective2D(
    #         u0, tend, dt_model, tstart)

    # elif model.system_name == "AdvectionDiffusionReflective3D":
    #     u = utils_processing_AD.evolveAdvectionDiffusionReflective3D(
    #         u0, tend, dt_model, tstart)

    # elif model.system_name == "AD3DD2":
    #     u = utils_processing_AD.evolveAD3DD(u0, tend, dt_model, tstart, D=0.002)

    # elif model.system_name == "AD3DD20":
    #     u = utils_processing_AD.evolveAD3DD(u0, tend, dt_model, tstart, D=0.02)

    # elif model.system_name == "AD3DD200":
    #     u = utils_processing_AD.evolveAD3DD(u0, tend, dt_model, tstart, D=0.2)

    # elif model.system_name == "AD3DD2000":
    #     u = utils_processing_AD.evolveAD3DD(u0, tend, dt_model, tstart, D=2.0)

    # elif model.system_name == "KSGP64L22":
    #     u = evolveKSGP64L22(u0, tend, dt_model, tstart)

    # elif model.system_name == "KSGP512L100":
    #     u = evolveKSGP512L100(u0, tend, dt_model, tstart)

    # elif model.system_name == "FHN":
    #     u = evolveFitzHughNagumo(u0, tend, dt_model, tstart)
    return u


def addFieldsToCompare(model, fields_to_compare):

    if model.system_name == "FHN":
        # For FHN system comparing additionally the activator and inhibitor MNAD errors (Mean normalized absolute difference)
        fields_to_compare.append("mnad_avg_act")
        fields_to_compare.append("mnad_avg_in")
        fields_to_compare.append("mnad_avg_over_ics_act")
        fields_to_compare.append("mnad_avg_over_ics_in")

    elif (model.system_name in ["AD1D"]) or (model.system_name
                                             in AD3Ddatasets()):
        # For advection diffusion comparing the L1-Histogram error and the Wasserstein Distance
        # fields_to_compare.append("state_dist_L1_hist_error")
        # fields_to_compare.append("state_dist_wasserstein_distance")

        fields_to_compare.append("state_dist_L1_hist_error_all")
        fields_to_compare.append("state_dist_wasserstein_distance_all")

        fields_to_compare.append("state_dist_L1_hist_error_avg")
        fields_to_compare.append("state_dist_wasserstein_distance_avg")

    elif model.system_name == "KSGP64L22":
        # For the Kuramoto-Sivashinsky comparing the errors on the distribution and the errors over initial conditions
        fields_to_compare.append("state_dist_L1_hist_error")
        fields_to_compare.append("state_dist_wasserstein_distance")
        fields_to_compare.append("rmnse_avg_over_ics")
        fields_to_compare.append("mnad_avg_over_ics")

    return fields_to_compare
