#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python


def checkSystemName(model):
    isGood = False
    if isAlanineDataset(model.system_name): isGood = True
    if model.system_name in [
            "AD1D",
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
            "CGW",
            "CGW50",
            "Lorenz3D",
            "TRP",
            "Alanine",
            "BMP",
            "KSGP64L22",
            "MackeyGlass",
            "MackeyGlass_SNR5",
            "MackeyGlass_SNR10",
            "MackeyGlass_SNR60",
            "PolandElectricity",
            "SantaFe",
            "Leuven",
            "Darwin",
    ]:
        isGood = True

    if not isGood:
        raise ValueError("system_name {:} not found.".format(
            model.system_name))
    return isGood


def isAlanineDataset(system_name):
    protein_datasets = [
        "Alanine",
        # "Alanine_badNOrotTr_waterNVT_F4",
        # "Alanine_badNOrotTr_waterNVT_F3",
        # "Alanine_badNOrotTr_waterNVT_F2",
        # "Alanine_badNOrotTr_waterNVT_F1",
        # "Alanine_badNOrotTr_waterNVT_F0",
        # "Alanine_badNOrotTr_waterNVT",
        # "Alanine_badNOrotTr_waterNVT_reduced",
        # "badNOrotTr_waterNVE",
        # "badNOrotTr_NVE",
        # "badNOrotTr_NVT",
    ]
    bool_ = True if system_name in protein_datasets else False
    return bool_
