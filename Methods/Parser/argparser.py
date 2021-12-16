#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import argparse


def getLEDParser(parser):
    parser.add_argument("--mode",
                        help="train, test, all",
                        type=str,
                        required=True)
    parser.add_argument("--system_name",
                        help="system_name",
                        type=str,
                        required=True)
    parser.add_argument("--write_to_log",
                        help="write_to_log",
                        type=int,
                        required=False,
                        default=0)

    parser.add_argument("--input_dim",
                        help="input_dim",
                        type=int,
                        required=True)
    parser.add_argument("--channels",
                        help="Channels in case input more than 1-D.",
                        type=int,
                        required=False,
                        default=0)
    parser.add_argument("--Dx",
                        help="Channel dimension 1.",
                        type=int,
                        required=False,
                        default=0)
    parser.add_argument("--Dy",
                        help="Channel dimension 2.",
                        type=int,
                        required=False,
                        default=0)
    parser.add_argument("--Dz",
                        help="Channel dimension 3.",
                        type=int,
                        required=False,
                        default=0)

    parser.add_argument("--AE_convolutional",
                        help="AE_convolutional",
                        default=0,
                        type=int,
                        required=False)
    parser.add_argument("--kernel_size",
                        help="kernel_size",
                        default=0,
                        type=int,
                        required=False)

    parser.add_argument(
        "--MDN_weight_sharing",
        help=
        "Weight sharing at the output of the MDN. For permutation invariant output has to be set to one (so that every output component has the same likelihood).",
        type=int,
        default=0,
        required=False)
    parser.add_argument(
        "--MDN_multivariate",
        help=
        "Multivariate modelling, capturing dependency in the random variables at the output.",
        type=int,
        default=0,
        required=False)
    parser.add_argument("--MDN_kernels",
                        help="MDN number of gaussian kernels (default 0)",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument(
        "--MDN_hidden_units",
        help="MDN number of hidden units for transformation (default 0)",
        type=int,
        default=0,
        required=False)
    parser.add_argument("--MDN_sigma_max",
                        help="Maximum variance (scaled).",
                        type=float,
                        default=0.4,
                        required=False)

    parser.add_argument("--RNN_MDN_kernels",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument("--RNN_MDN_multivariate",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument("--RNN_MDN_hidden_units",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument("--RNN_MDN_sigma_max",
                        type=float,
                        default=0.4,
                        required=False)

    parser.add_argument(
        "--output_forecasting_loss",
        help="loss of RNN forecasting (if 0, autoencoder mode)",
        type=int,
        default=1,
        required=False)
    parser.add_argument(
        "--latent_forecasting_loss",
        help="loss of dynamic consistency in the latent dynamics",
        type=int,
        default=0,
        required=False)
    parser.add_argument("--reconstruction_loss",
                        help="reconstruction_loss",
                        type=int,
                        default=0,
                        required=False)

    parser.add_argument("--RNN_cell_type",
                        help="type of the rnn cell",
                        type=str,
                        required=False,
                        default="lstm")
    parser.add_argument("--RNN_activation_str",
                        help="RNN_activation_str",
                        type=str,
                        required=False,
                        default="tanh")
    parser.add_argument('--RNN_layers_size',
                        type=int,
                        help='size of the RNN layers',
                        required=False,
                        default=0)
    parser.add_argument('--RNN_layers_num',
                        type=int,
                        help='number of the RNN layers',
                        required=False,
                        default=0)

    parser.add_argument('--AE_layers_size',
                        type=int,
                        help='The size of the autoencoder layers',
                        required=False,
                        default=0)
    parser.add_argument('--AE_layers_num',
                        type=int,
                        help='The number of the autoencoder layers',
                        required=False,
                        default=0)
    parser.add_argument("--activation_str_general",
                        help="Activation of Autoencoder/MLP/MDN layers",
                        type=str,
                        required=False,
                        default="selu")
    parser.add_argument("--AE_residual",
                        help="AE_residual",
                        type=int,
                        required=False,
                        default=0)
    parser.add_argument(
        '--AE_perm_invariant_latent_dim',
        type=int,
        help=
        'If a permutation invariant autoencoder should be used. If yes, the dimension of the latent representation should be given.',
        required=False,
        default=0)
    parser.add_argument(
        '--AE_perm_invariant_feature',
        type=str,
        help=
        'The method used to contruct the feature space of the permutation invariant layer (either max, min, or mean).',
        required=False,
        default="mean")

    parser.add_argument("--latent_state_dim",
                        help="latent_state_dim",
                        type=int,
                        required=False,
                        default=0)

    parser.add_argument("--zoneout_keep_prob",
                        help="zoneout_keep_prob",
                        type=float,
                        required=False,
                        default=1)
    parser.add_argument("--dropout_keep_prob",
                        help="dropout_keep_prob",
                        type=float,
                        required=False,
                        default=1)

    parser.add_argument("--sequence_length",
                        help="sequence_length",
                        type=int,
                        required=True)
    parser.add_argument("--scaler", help="scaler", type=str, required=True)

    parser.add_argument("--learning_rate",
                        help="learning_rate",
                        type=float,
                        required=True)
    parser.add_argument("--weight_decay",
                        help="weight_decay",
                        type=float,
                        required=False,
                        default=0.0)
    parser.add_argument("--batch_size",
                        help="batch_size",
                        type=int,
                        required=True)
    parser.add_argument("--overfitting_patience",
                        help="overfitting_patience",
                        type=int,
                        required=True)
    parser.add_argument("--max_epochs",
                        help="max_epochs",
                        type=int,
                        required=True)
    parser.add_argument("--max_rounds",
                        help="max_rounds",
                        type=int,
                        required=True)
    parser.add_argument("--retrain",
                        help="retrain",
                        type=int,
                        required=False,
                        default=0)
    parser.add_argument("--num_test_ICS",
                        help="num_test_ICS",
                        type=int,
                        required=False,
                        default=0)

    parser.add_argument("--prediction_horizon",
                        help="prediction_horizon",
                        type=int,
                        required=False,
                        default=0)
    parser.add_argument(
        "--display_output",
        help="control the verbosity level of output , default True",
        type=int,
        required=False,
        default=1)

    parser.add_argument("--reference_train_time",
                        help="The reference train time in hours",
                        type=float,
                        default=24,
                        required=False)
    parser.add_argument(
        "--buffer_train_time",
        help="The buffer train time to save the model in hours",
        type=float,
        default=0.5,
        required=False)

    parser.add_argument("--random_seed",
                        help="random_seed",
                        type=int,
                        default=1,
                        required=False)

    parser.add_argument("--random_seed_in_name",
                        help="random_seed_in_name",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument("--optimizer_str",
                        help="adam or sgd with cyclical learning rate",
                        type=str,
                        default="adam",
                        required=False)

    parser.add_argument("--make_videos",
                        help="make_videos.",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument("--train_AE_only",
                        help="train_AE_only.",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument("--train_rnn_only",
                        help="train_rnn_only.",
                        type=int,
                        default=0,
                        required=False)

    parser.add_argument("--teacher_forcing_forecasting",
                        help="to test the the model in teacher forcing.",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument("--multiscale_forecasting",
                        help="to test the model in multiscale.",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument(
        "--iterative_state_forecasting",
        help=
        "to test the model in iterative forecasting, propagating the output state of the model.",
        type=int,
        default=0,
        required=False)
    parser.add_argument(
        "--iterative_latent_forecasting",
        help=
        "to test the model in iterative forecasting, propagating the latent space od the model.",
        type=int,
        default=0,
        required=False)
    parser.add_argument("--cudnn_benchmark",
                        help="cudnn_benchmark",
                        type=int,
                        default=0,
                        required=False)

    parser.add_argument("--multiscale_macro_steps_list",
                        action='append',
                        help="list of macro steps to perform",
                        type=float,
                        default=[],
                        required=False)
    parser.add_argument("--multiscale_micro_steps_list",
                        action='append',
                        help="list of micro steps to perform",
                        type=float,
                        default=[],
                        required=False)

    parser.add_argument("--MDN_fixed_kernels",
                        help="MDN_fixed_kernels",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument("--MDN_train_kernels",
                        help="MDN_train_kernels",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument("--MDN_multivariate_covariance_layer",
                        help="MDN_multivariate_covariance_layer",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument("--MDN_distribution",
                        help="MDN_distribution",
                        type=str,
                        default="normal",
                        required=False)
    parser.add_argument("--MDN_multivariate_pretrain_diagonal",
                        help="MDN_multivariate_pretrain_diagonal",
                        type=int,
                        default=0,
                        required=False)

    parser.add_argument("--RNN_MDN_fixed_kernels",
                        help="RNN_MDN_fixed_kernels",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument("--RNN_MDN_train_kernels",
                        help="RNN_MDN_train_kernels",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument("--RNN_MDN_multivariate_covariance_layer",
                        help="RNN_MDN_multivariate_covariance_layer",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument("--RNN_MDN_multivariate_pretrain_diagonal",
                        help="RNN_MDN_multivariate_pretrain_diagonal",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument("--RNN_MDN_distribution",
                        help="RNN_MDN_distribution",
                        type=str,
                        default="normal",
                        required=False)

    parser.add_argument("--compute_spectrum",
                        help="compute_spectrum.",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument("--plot_state_distributions",
                        help="plot_state_distributions.",
                        type=int,
                        default=1,
                        required=False)
    parser.add_argument("--plot_state_distributions_system",
                        help="plot_state_distributions_system.",
                        type=int,
                        default=1,
                        required=False)
    parser.add_argument("--plot_system",
                        help="plot_system.",
                        type=int,
                        default=1,
                        required=False)
    parser.add_argument("--plot_testing_ics_examples",
                        help="plot_testing_ics_examples.",
                        type=int,
                        default=1,
                        required=False)

    parser.add_argument("--plot_latent_dynamics_single_traj",
                        help="plot_latent_dynamics_single_traj",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument("--plot_latent_dynamics_comparison_system",
                        help="plot_latent_dynamics_comparison_system.",
                        type=int,
                        default=0,
                        required=False)
    parser.add_argument("--plot_protein_trajectories",
                        help="plot_protein_trajectories.",
                        type=int,
                        default=0,
                        required=False)

    parser.add_argument("--plot_multiscale_results_comparison",
                        help="plot_multiscale_results_comparison.",
                        type=int,
                        default=0,
                        required=False)

    parser.add_argument("--save_format",
                        help="save format, hickle or pickle",
                        type=str,
                        default="pickle",
                        required=False)

    parser.add_argument("--system_name_data",
                        help="system_name_data",
                        type=str,
                        required=False,
                        default="None")

    parser.add_argument("--test_on_train",
                        help="test_on_train",
                        type=int,
                        required=False,
                        default=0)
    parser.add_argument("--test_on_val",
                        help="test_on_val",
                        type=int,
                        required=False,
                        default=0)
    parser.add_argument("--test_on_test",
                        help="test_on_test",
                        type=int,
                        required=False,
                        default=1)

    parser.add_argument("--truncate_data",
                        help="truncate_data",
                        type=int,
                        required=False,
                        default=0)

    parser.add_argument("--learning_rate_in_name",
                        help="learning_rate_in_name",
                        type=int,
                        default=1,
                        required=False)
    # parser.add_argument("--iterative_loss_in_name",
    #                     help="iterative_loss_in_name",
    #                     type=int,
    #                     default=1,
    #                     required=False)

    # parser.add_argument("--iterative_loss_length",
    #                     help="length of the ITERATIVE training loss",
    #                     type=int,
    #                     default=0,
    #                     required=False)

    # parser.add_argument("--iterative_loss_length_weight",
    #                     help="weight of the ITERATIVE training loss",
    #                     type=float,
    #                     default=0.0,
    #                     required=False)
    parser.add_argument("--iterative_loss_schedule_and_gradient",
                        help="iterative_loss_schedule_and_gradient",
                        type=str,
                        default="none",
                        required=False)
    parser.add_argument("--iterative_loss_validation",
                        help="iterative_loss_validation",
                        type=int,
                        default=0,
                        required=False)

    parser.add_argument(
        "--iterative_propagation_is_latent",
        help=
        "Unplug the encoder and propagate only the latent state during iterative forecasting (only used for training).",
        type=int,
        default=0,
        required=False)

    return parser


def defineParser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='Selection of the model.',
                                       dest='model_name')
    md_arnn = subparsers.add_parser("md_arnn")
    md_arnn = getLEDParser(md_arnn)
    return parser
