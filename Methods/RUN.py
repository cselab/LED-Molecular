#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

import LED
import Parser.argparser as argparser
from Config.global_conf import global_params


def getModel(params):
    str_ = "#" * 10 + "    {:}    ".format(params["model_name"]) + "#" * 10
    print("#" * len(str_))
    print(str_)
    print("#" * len(str_))
    if params["model_name"] == "md_arnn":
        return LED.Networks.md_arnn.md_arnn(params)
    else:
        raise ValueError(
            "Model {:} not found.\nImplemented models are:\n| md_arnn |".
            format(params["model_name"]))


def runModel(params_dict):

    if params_dict["mode"] in ["debug"]:
        # Train the model
        debugModel(params_dict)

    if params_dict["mode"] in ["train", "all"]:
        # Train the model
        trainModel(params_dict)

    if params_dict["mode"] in ["test", "test_only", "all"]:
        # Test the model
        testModel(params_dict)

    if params_dict["mode"] in ["test", "plot", "all"]:
        # Mode to plot the results
        plotModel(params_dict)

    return 0


def debugModel(params_dict):
    model = getModel(params_dict)
    model.debug()
    model.delete()
    del model
    return 0


def trainModel(params_dict):
    model = getModel(params_dict)
    model.train()
    model.delete()
    del model
    return 0


def testModel(params_dict):
    model = getModel(params_dict)
    model.test()
    model.delete()
    del model
    return 0


def plotModel(params_dict):
    model = getModel(params_dict)
    model.plot()
    model.delete()
    del model
    return 0


def main():
    parser = argparser.defineParser()
    args = parser.parse_args()
    args_dict = args.__dict__

    # for key in args_dict:
    # print(key)

    # DEFINE PATHS AND DIRECTORIES
    args_dict["saving_path"] = global_params.saving_path.format(
        args_dict["system_name"])
    args_dict["model_dir"] = global_params.model_dir
    args_dict["fig_dir"] = global_params.fig_dir
    args_dict["results_dir"] = global_params.results_dir
    args_dict["logfile_dir"] = global_params.logfile_dir
    system_name_data = args.system_name if (
        args.system_name_data == "None") else args.system_name_data

    args_dict["data_path_train"] = global_params.data_path_train.format(
        system_name_data)
    args_dict["data_path_test"] = global_params.data_path_test.format(
        system_name_data)
    args_dict["data_path_val"] = global_params.data_path_val.format(
        system_name_data)
    args_dict["data_path_gen"] = global_params.data_path_gen.format(
        system_name_data)
    args_dict["worker_id"] = 0
    args_dict["project_path"] = global_params.project_path

    runModel(args_dict)


if __name__ == '__main__':
    main()
