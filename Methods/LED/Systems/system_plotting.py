#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

from .. import Utils as utils

######################################################################################################
## PLOTTING UTILITIES FOR EACH SYSTEM
######################################################################################################
from .AD import utils_plotting_AD as utils_plotting_AD
from .KS import utils_plotting_ks as utils_plotting_ks
from .BMP import utils_plotting_bmp as utils_plotting_bmp
from .Alanine import utils_plotting_alanine as utils_plotting_alanine
from .TRP import utils_plotting_trp as utils_plotting_trp
from .CGW import utils_plotting_cgw as utils_plotting_cgw


def plotLatentDynamicsComparisonSystem(model, set_name):

    if utils.isAlanineDataset(model.system_name):
        utils_plotting_alanine.plotLatentDynamicsComparisonSystemAlanine(
            model, set_name)

    if model.system_name == "BMP":
        utils_plotting_bmp.plotLatentDynamicsComparisonSystemBMP(
            model, set_name)

    if model.system_name == "TRP":
        utils_plotting_trp.computeLatentDynamicsDistributionErrorTRP(
            model, set_name)

    return 0


def plotStateDistributionsSystem(model, results, set_name, testing_mode):

    if utils.isAlanineDataset(model.system_name):
        utils_plotting_alanine.plotStateDistributionsSystemAlanine(
            model, results, set_name, testing_mode)

    if model.system_name == "BMP":
        utils_plotting_bmp.plotStateDistributionsSystemBMP(
            model, results, set_name, testing_mode)

    if model.system_name == "AdvectionDiffusionReflective3D":
        utils_plotting_AD.plotStateDistributionsSystemAdvectionDiffusion3D(
            model, results, set_name, testing_mode)

    if model.system_name == "AdvectionDiffusionReflective":
        utils_plotting_AD.plotStateDistributionsSystemAdvectionDiffusion(
            model, results, set_name, testing_mode)

    return 0


def plotSystem(model, results, set_name, testing_mode):
    print("# plotSystem() #")

    if model.system_name in ["BMP"]:

        utils.plotLatentDynamicsFreeEnergy(model, results, set_name,
                                           testing_mode)
        utils.plotLatentTransitionTimes(model, results, set_name, testing_mode)

        utils_plotting_bmp.plotLatentMetaStableStatesBMP(
            model, results, set_name, testing_mode)

        utils_plotting_bmp.plotLatentTransitionTimesBMP(
            model, results, set_name, testing_mode)
        utils_plotting_bmp.plotTransitionTimesBMP(model, results, set_name,
                                                  testing_mode)

    elif model.system_name in [
            # "Alanine_badNOrotTr_waterNVT",
            "Alanine",
    ]:

        # utils.plotLatentDynamicsFreeEnergy(model, results, set_name, testing_mode)
        # utils.plotLatentTransitionTimes(model, results, set_name, testing_mode)

        utils_plotting_alanine.plotLatentMetaStableStatesAndLatentTransitionTimesAlanine(model, results, set_name, testing_mode)

        # utils_plotting_alanine.plotTransitionTimesAlanine(model, results, set_name, testing_mode)

        # if model.params["plot_protein_trajectories"]:
        #     utils_plotting_alanine.generateXYZfromABDtrajectoriesAlanine(
        #         model, results, set_name, testing_mode)

    elif model.system_name in ["TRP"]:

        # utils.plotLatentDynamicsFreeEnergy(model, results, set_name, testing_mode)

        if model.params["plot_protein_trajectories"]:
            utils_plotting_trp.generateXYZfromABDtrajectoriesTRP(
                model, results, set_name, testing_mode)

        # utils.plotLatentTransitionTimes(model, results, set_name, testing_mode)

    elif model.system_name in ["CGW", "CGW50"]:

        # utils_plotting_cgw.plot_density_trajectory_CGW(model, results, set_name, testing_mode)
        # utils_plotting_cgw.output_diffusion_coefficient_CGW(model, results, set_name, testing_mode)
        utils_plotting_cgw.plot_com_traj_CGW(model, results, set_name,
                                             testing_mode)
    else:
        pass
    return 0
