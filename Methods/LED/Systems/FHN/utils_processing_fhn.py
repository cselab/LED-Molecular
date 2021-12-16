#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

import numpy as np


def addResultsSystemFHN(model, results, testing_mode):

    if "autoencoder" in testing_mode:
        targets_all = results["input_sequence_all"]
        predictions_all = results["input_decoded_all"]
    else:
        targets_all = results["targets_all"]
        predictions_all = results["predictions_all"]

    targets_all = results["targets_all"]
    predictions_all = results["predictions_all"]
    # ONLY RELEVANT FOR THE FHN SYSTEM
    targets_all_act = targets_all[:, :, :int(model.input_dim / 2)]
    predictions_all_act = predictions_all[:, :, :int(model.input_dim / 2)]
    print(np.shape(targets_all_act))
    print(np.shape(predictions_all_act))
    mnad_all_act = np.mean(np.abs(targets_all_act - predictions_all_act) /
                           (np.max(targets_all_act) - np.min(targets_all_act)),
                           axis=2)
    mnad_avg_over_ics_act = np.mean(mnad_all_act, axis=0)
    mnad_avg_act = np.mean(mnad_all_act)
    print("(MNAD) MEAN NORMALISED ABSOLUTE DIFFERENCE ACTIVATOR: {:}".format(
        mnad_avg_act))
    targets_all_in = targets_all[:, :, int(model.input_dim / 2):]
    predictions_all_in = predictions_all[:, :, int(model.input_dim / 2):]
    print(np.shape(targets_all_in))
    print(np.shape(predictions_all_in))
    mnad_all_in = np.mean(np.abs(targets_all_in - predictions_all_in) /
                          (np.max(targets_all_in) - np.min(targets_all_in)),
                          axis=2)
    mnad_avg_over_ics_in = np.mean(mnad_all_in, axis=0)
    mnad_avg_in = np.mean(mnad_all_in)
    print("(MNAD) MEAN NORMALISED ABSOLUTE DIFFERENCE INHIBITOR: {:}".format(
        mnad_avg_in))
    # Adding the computed results
    results["fields_2_save_2_logfile"].append("mnad_avg_act")
    results["fields_2_save_2_logfile"].append("mnad_avg_in")
    results.update({
        "mnad_avg_act": mnad_avg_act,
        "mnad_avg_in": mnad_avg_in,
        "mnad_avg_over_ics_act": mnad_avg_over_ics_act,
        "mnad_avg_over_ics_in": mnad_avg_over_ics_in
    })
    return results
