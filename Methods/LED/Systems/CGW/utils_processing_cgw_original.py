#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np
# from  ... import Utils as utils


def addResultsSystemCGW(model, results, statistics, testing_mode):
    if "autoencoder" in testing_mode:
        targets_all = results["input_sequence_all"]
        predictions_all = results["input_decoded_all"]
    else:
        targets_all = results["targets_all"]
        predictions_all = results["predictions_all"]

    dt = results["dt"]

    # Some result you compute
    erminas_result = np.random.rand(10, 3)
    results["erminas_result"] = erminas_result

    # Do you want to save it in a log-file ? Like a number that determines the effiency ? Probably not..
    # results["fields_2_save_2_logfile"].append("erminas_result")
    return results
