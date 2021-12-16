#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

# TORCH
import torch
import sys
print("MODEL: MD-ARNN")
print("-V- Python Version = {:}".format(sys.version))
print("-V- Torch Version = {:}".format(torch.__version__))
from torch.autograd import Variable

# LIBRARIES
import numpy as np
import os
import random
import time
from tqdm import tqdm

# UTILITIES
from .. import Utils as utils
from .. import Systems as systems

# NETWORKS
from . import md_arnn_model

# PRINTING
from functools import partial
print = partial(print, flush=True)


class md_arnn():
    def __init__(self, params):
        super(md_arnn, self).__init__()
        # Starting the timer
        self.start_time = time.time()
        # The parameters used to define the model
        self.params = params.copy()

        # The system to which the model is applied on
        self.system_name = params["system_name"]
        # Checking the system name
        utils.checkSystemName(self)
        # The save format
        self.save_format = params["save_format"]

        # A reference training time
        self.reference_train_time = utils.getReferenceTrainingTime(
            params["reference_train_time"], params["buffer_train_time"])

        # Checking whether the GPU is available and setting the default tensor datatype
        self.gpu = torch.cuda.is_available()
        if self.gpu:
            self.torch_dtype = torch.cuda.DoubleTensor
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
            if self.params["cudnn_benchmark"]:
                torch.backends.cudnn.benchmark = True
        else:
            self.torch_dtype = torch.DoubleTensor
            torch.set_default_tensor_type(torch.DoubleTensor)

        ##################################################################
        # RANDOM SEEDING
        ##################################################################
        self.random_seed = params["random_seed"]

        # Setting the random seed
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if self.gpu: torch.cuda.manual_seed(self.random_seed)

        ##################################################################
        # Parameters of the MDN-AE
        # Mixture Density Autoencoder network at the output
        ##################################################################
        # Permutation invariance input layer
        self.AE_perm_invariant_latent_dim = params[
            "AE_perm_invariant_latent_dim"]
        if params["MDN_kernels"] > 0:
            self.MDN_bool = 1
            self.MDN_kernels = params["MDN_kernels"]
            self.MDN_weight_sharing = params["MDN_weight_sharing"]
            self.MDN_multivariate = params["MDN_multivariate"]
            self.MDN_hidden_units = params["MDN_hidden_units"]
            self.MDN_sigma_max = params["MDN_sigma_max"]
            self.MDN_distribution = params["MDN_distribution"]

            # if self.AE_perm_invariant_latent_dim and not self.MDN_weight_sharing:
            #     raise ValueError("Permutation invariant autoencoder cannot be used with MDN without weight sharing in the output. The likelihood of each output component will not be the same (violation of perm. invariance.")

            self.MDN_fixed_kernels = params["MDN_fixed_kernels"]
            self.MDN_train_kernels = params["MDN_train_kernels"]
            self.MDN_multivariate_covariance_layer = params[
                "MDN_multivariate_covariance_layer"]
            self.MDN_multivariate_pretrain_diagonal = params[
                "MDN_multivariate_pretrain_diagonal"]
        else:
            self.MDN_bool = 0

        ##################################################################
        # Parameters of the RNN-MDN
        # Mixture Density Recurrent Neural Network
        ##################################################################
        if params["RNN_MDN_kernels"] > 0:
            self.RNN_MDN_bool = 1
            self.RNN_MDN_kernels = params["RNN_MDN_kernels"]
            self.RNN_MDN_multivariate = params["RNN_MDN_multivariate"]
            self.RNN_MDN_hidden_units = params["RNN_MDN_hidden_units"]
            self.RNN_MDN_sigma_max = params["RNN_MDN_sigma_max"]
            self.RNN_MDN_distribution = params["RNN_MDN_distribution"]
            self.RNN_MDN_fixed_kernels = params["RNN_MDN_fixed_kernels"]
            self.RNN_MDN_train_kernels = params["RNN_MDN_train_kernels"]
            self.RNN_MDN_multivariate_covariance_layer = params[
                "RNN_MDN_multivariate_covariance_layer"]
            self.RNN_MDN_multivariate_pretrain_diagonal = params[
                "RNN_MDN_multivariate_pretrain_diagonal"]
        else:
            self.RNN_MDN_bool = 0

        # ##################################################################
        # # PARAMETERS - ABPTT LOSS
        # ##################################################################
        # self.iterative_loss_length = params["iterative_loss_length"]
        # self.iterative_loss_length_weight = params[
        #     "iterative_loss_length_weight"]
        # if self.iterative_loss_length:
        #     print("Utilizing iterative loss with length {:} and weight {:}.".
        #           format(self.iterative_loss_length,
        #                  self.iterative_loss_length_weight))
        self.iterative_propagation_is_latent = params[
            "iterative_propagation_is_latent"]

        self.iterative_loss_schedule_and_gradient = params[
            "iterative_loss_schedule_and_gradient"]
        self.iterative_loss_validation = params["iterative_loss_validation"]
        if self.iterative_loss_schedule_and_gradient not in [
                "none",
                "exponential_with_gradient",
                "linear_with_gradient",
                "inverse_sigmoidal_with_gradient",
                "exponential_without_gradient",
                "linear_without_gradient",
                "inverse_sigmoidal_without_gradient",
        ]:
            raise ValueError(
                "Iterative loss schedule {:} not recognized.".format(
                    self.iterative_loss_schedule_and_gradient))
        else:
            if "without_gradient" in self.iterative_loss_schedule_and_gradient or "none" in self.iterative_loss_schedule_and_gradient:
                self.iterative_loss_gradient = 0
            elif "with_gradient" in self.iterative_loss_schedule_and_gradient:
                self.iterative_loss_gradient = 1
            else:
                raise ValueError(
                    "self.iterative_loss_schedule_and_gradient={:} not recognized."
                    .format(self.iterative_loss_schedule_and_gradient))

        # Optimizer to use
        self.optimizer_str = params["optimizer_str"]

        # Whether the autoencoder is convolutional or not
        self.AE_convolutional = params["AE_convolutional"]

        ##################################################################
        # SETTING THE PATHS
        ##################################################################
        # The path of the training data
        self.data_path_train = params['data_path_train']
        # The path of the training data
        self.data_path_val = params['data_path_val']
        # The path of the test data
        self.data_path_test = params['data_path_test']
        # General data path (scaler data/etc.)
        self.data_path_gen = params['data_path_gen']
        # The path to save all the results
        self.saving_path = params['saving_path']
        # The directory to save the model (inside saving_path)
        self.model_dir = params['model_dir']
        # The directory to save the figures (inside saving_path)
        self.fig_dir = params['fig_dir']
        # The directory to save the data results (inside saving_path)
        self.results_dir = params['results_dir']
        # The directory to save the logfiles (inside saving_path)
        self.logfile_dir = params["logfile_dir"]
        # Whether to write a log-file or not
        self.write_to_log = params["write_to_log"]

        # Whether to display in the output (verbocity)
        self.display_output = params["display_output"]

        # The number of IC to test on
        self.num_test_ICS = params["num_test_ICS"]

        # The prediction horizon
        self.prediction_horizon = params["prediction_horizon"]

        # The activation string of the RNN
        self.RNN_activation_str = params['RNN_activation_str']

        ##################################################################
        # DIMENSION OF THE INPUT
        ##################################################################
        # The dimension of the input (first dimension)
        # The code can handle various input types:
        #         ----------------------------------------------------------
        #         1. Data in the form (T, input_dim)
        #                 T is the number of timesteps
        #                 input_dim is the dimensionality of the time-series
        #         ----------------------------------------------------------
        #         2. Data in the form (T, input_dim, Dx)         # Particle data, channels=1
        #                 T is the number of timesteps
        #                 input_dim is the number of particles
        #                 Cx is the features of each particle
        #                 permutation invariance is applied on the dimension input_dim (number of particles)
        #         ----------------------------------------------------------

        self.input_dim = params['input_dim']
        self.channels = params['channels']
        self.Dx, self.Dy, self.Dz = self.getChannels(self.channels, params)

        # Dropout probabilities for regularizing the AE
        self.dropout_keep_prob = params["dropout_keep_prob"]
        # Zoneout probability for regularizing the RNN
        self.zoneout_keep_prob = params["zoneout_keep_prob"]
        # The sequence length
        self.sequence_length = params['sequence_length']
        # The cell type of the RNN
        self.RNN_cell_type = params['RNN_cell_type']

        # TESTING MODES
        self.teacher_forcing_forecasting = params[
            "teacher_forcing_forecasting"]
        self.iterative_state_forecasting = params[
            "iterative_state_forecasting"]
        self.iterative_latent_forecasting = params[
            "iterative_latent_forecasting"]
        self.multiscale_forecasting = params["multiscale_forecasting"]

        if self.multiscale_forecasting:
            # Performing multiscale forecasting with different macro steps / micro steps
            self.multiscale_macro_steps_list = params[
                "multiscale_macro_steps_list"]
            self.multiscale_micro_steps_list = params[
                "multiscale_micro_steps_list"]

        # Whether the AE has a permutation invariant layer
        self.is_perm_inv = bool(self.AE_perm_invariant_latent_dim > 0)

        ##################################################################
        # SCALER
        ##################################################################
        self.scaler = params["scaler"]

        ##################################################################
        # TRAINING PARAMETERS
        ##################################################################
        # Whether to retrain or not
        self.retrain = params['retrain']

        self.batch_size = params['batch_size']
        self.overfitting_patience = params['overfitting_patience']
        self.max_epochs = params['max_epochs']
        self.max_rounds = params['max_rounds']
        self.learning_rate = params['learning_rate']
        self.weight_decay = params['weight_decay']

        self.train_AE_only = params["train_AE_only"]
        self.train_rnn_only = params["train_rnn_only"]

        self.output_forecasting_loss = params["output_forecasting_loss"]
        self.latent_forecasting_loss = params["latent_forecasting_loss"]
        self.reconstruction_loss = params["reconstruction_loss"]

        if self.MDN_bool and self.RNN_MDN_bool:
            # Both a MD-RNN and a MD-Decoder
            if self.output_forecasting_loss:
                raise ValueError(
                    "output_forecasting_loss is not compatible with MD-RNN and MD decoder."
                )

        if (params["latent_state_dim"] == 0
                or params["latent_state_dim"] is None):
            self.has_autoencoder = 0
        else:
            self.has_autoencoder = 1

        self.layers_rnn = [self.params["RNN_layers_size"]
                           ] * self.params["RNN_layers_num"]
        if len(self.layers_rnn) > 0:
            # Parsing the RNN layers
            self.has_rnn = 1
            # VALID COMPINATIONS:
            if self.RNN_activation_str not in ['tanh']:
                raise ValueError('ERROR: INVALID RNN ACTIVATION!')
            if self.RNN_cell_type not in ['lstm', 'gru', 'plain']:
                raise ValueError('ERROR: INVALID RNN CELL TYPE!')
        else:
            self.has_rnn = 0

        if (self.latent_forecasting_loss == 1 or self.reconstruction_loss
                == 1) and (self.has_autoencoder == 0):
            raise ValueError(
                "latent_forecasting_loss and reconstruction_loss are not meaningfull without latent state (Autoencoder mode)."
            )

        # Adding the autoencoder latent dimension if this is not None
        if params["latent_state_dim"] is not None and (
                params["latent_state_dim"] >
                0) and not self.AE_perm_invariant_latent_dim:
            # Parsing the ENCODER layers
            self.layers_encoder = [self.params["AE_layers_size"]
                                   ] * self.params["AE_layers_num"]
            self.layers_encoder_aug = self.layers_encoder.copy()
            self.layers_encoder_aug.insert(0, self.params["input_dim"])
            if not self.AE_convolutional:
                self.params["RNN_state_dim"] = params["latent_state_dim"]
                self.layers_encoder_aug.append(self.params["RNN_state_dim"])
                self.params["decoder_input_dim"] = self.params["RNN_state_dim"]
            else:
                self.params["decoder_input_dim"] = self.params[
                    "latent_state_dim"]
                self.params["RNN_state_dim"] = params[
                    "latent_state_dim"] * self.Dx * self.Dy

                self.layers_encoder_aug.append(self.params["latent_state_dim"])

            self.layers_decoder_aug = self.layers_encoder_aug[::-1]
            self.layers_decoder_aug[0] = self.params["decoder_input_dim"]

            self.layers_perm_inv_aug = []
        elif params["latent_state_dim"] is not None and (
                params["latent_state_dim"] >
                0) and self.AE_perm_invariant_latent_dim:
            self.layers_encoder = [self.params["AE_layers_size"]
                                   ] * self.params["AE_layers_num"]
            self.layers_perm_inv = [self.params["AE_layers_size"]
                                    ] * self.params["AE_layers_num"]
            self.layers_encoder_aug = self.layers_encoder.copy()
            self.layers_perm_inv_aug = self.layers_perm_inv.copy()

            if self.channels in [0, 1]:
                self.layers_perm_inv_aug.insert(0, self.Dx)
            else:
                raise ValueError(
                    "Permutation invariant layer for channels>1 not implemented. (perm inv for images)"
                )

            # Appending the latent space of the permutation invariance
            self.layers_perm_inv_aug.append(self.AE_perm_invariant_latent_dim)

            self.layers_encoder_aug.insert(0,
                                           self.AE_perm_invariant_latent_dim)
            if not self.AE_convolutional:
                self.params["RNN_state_dim"] = params["latent_state_dim"]
                self.layers_encoder_aug.append(self.params["RNN_state_dim"])
            else:
                raise ValueError(
                    "Permutation invariant convolution not implemented.")
            self.layers_decoder_aug = self.layers_encoder[::-1]
            self.layers_decoder_aug.insert(0, self.params["RNN_state_dim"])

            self.layers_decoder_aug.append(self.params["input_dim"])
            self.iterative_propagation_is_latent = params[
                "iterative_propagation_is_latent"]

            self.AE_perm_invariant_feature = params[
                "AE_perm_invariant_feature"]
            if self.AE_perm_invariant_feature not in ["max", "min", "mean"]:
                raise ValueError(
                    "Invalid feature construction function for permutation invariant encoder: {:}"
                    .format(self.AE_perm_invariant_feature))

        elif ((params["latent_state_dim"] is None) or
              (params["latent_state_dim"]
               == 0)) and self.AE_perm_invariant_latent_dim:
            raise ValueError(
                "Cannot have permutation invariant layer without autoencoder. (AE_perm_invariant_latent_dim>0 and latent_state_dim=0)"
            )
        else:
            self.layers_encoder = []
            self.layers_encoder_aug = []
            self.layers_decoder_aug = []
            self.layers_perm_inv = []
            self.layers_perm_inv_aug = []
            self.params["AE_layers_size"] = 0
            self.params["AE_layers_num"] = 0
            self.params["RNN_state_dim"] = params["input_dim"]
            self.iterative_propagation_is_latent = 0

        if self.output_forecasting_loss and self.MDN_bool and not self.iterative_propagation_is_latent:
            raise ValueError(
                "Invalid combination: self.output_forecasting_loss and self.MDN_bool and not self.iterative_propagation_is_latent. Lattent propagation and loss is necessary. How should the MDN propagate ?"
            )

        print("## PERM_INV layers: \n{:}".format(self.layers_perm_inv_aug))
        print("## ENCODER layers: \n{:}".format(self.layers_encoder_aug))
        print("## DECODER layers: \n{:}".format(self.layers_decoder_aug))

        self.model_name = self.createModelName()
        print("## Model name: \n{:}".format(self.model_name))
        self.saving_model_path = self.getModelDir() + "/model"

        self.makeDirectories()

        self.model = md_arnn_model.md_arnn_model(params, self)
        self.model.printModuleList()

        # PRINT PARAMS BEFORE PARALLELIZATION
        self.printParams()

        #TODO: load model when retrain
        self.model_parameters = self.model.getParams()

        # Initialize model parameters
        self.model.initializeWeights()
        self.device_count = torch.cuda.device_count()

        if self.gpu:
            print("USING CUDA -> SENDING THE MODEL TO THE GPU.")
            self.model.sendModelToCuda()
            if self.device_count > 1:
                raise ValueError("Muli-GPU training not implemented. Before launching the python script, set CUDA_DEVICES=0, and run with CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 ...")

        # Saving some info file for the model
        data = {
            "model": self,
            "params": params,
            "selfParams": self.params,
            "name": self.model_name
        }
        data_path = self.getModelDir() + "/info"
        utils.saveData(data, data_path, "pickle")
        self.data_info_dict = systems.getSystemDataInfo(self)

    def getChannels(self, channels, params):
        if channels == 0:
            Dx, Dy, Dz = 1, 0, 0
        elif channels == 1:
            assert (params["Dx"] > 0)
            Dx, Dy, Dz = params["Dx"], 0, 0
        elif channels == 2:
            assert (params["Dx"] > 0)
            assert (params["Dy"] > 0)
            Dx, Dy, Dz = params["Dx"], params["Dy"], 0
        elif channels == 3:
            assert (params["Dx"] > 0)
            assert (params["Dy"] > 0)
            assert (params["Dz"] > 0)
            Dx, Dy, Dz = params["Dx"], params["Dy"], params["Dz"]
        return Dx, Dy, Dz

    def getKeysInModelName(self, with_autoencoder=True, with_rnn=True):

        keys = {
            'scaler': '-scaler_',
            # 'optimizer_str':'-OPT_',
        }

        if self.params["learning_rate_in_name"]:
            keys.update({
                'learning_rate': '-LR_',
            })

        keys.update({
            'weight_decay': '-L2_',
            # 'random_seed':'-RS_',
            # 'worker_id':'-WID_',
        })

        if self.has_autoencoder and with_autoencoder:
            if self.MDN_bool:
                keys.update({
                    'MDN_distribution': '-MDN_',
                })
                keys.update({
                    'MDN_kernels': '-KERN_',
                })
                keys.update({
                    'MDN_hidden_units': '-HIDDEN_',
                })
                keys.update({
                    'MDN_sigma_max': '-SigmaMax_',
                })
                if self.MDN_multivariate:
                    keys.update({
                        'MDN_multivariate': '-MULT_',
                    })
                if self.MDN_multivariate_covariance_layer:
                    keys.update({
                        'MDN_multivariate_covariance_layer':
                        '-MULT_COV_',
                    })

            if self.AE_convolutional:
                keys.update({
                    'kernel_size': '-CONV-AUTO-KERN_',
                })
            if self.AE_perm_invariant_latent_dim:
                # If the autoencoder is permutation invariant, do not add the input dimension
                keys.update({
                    'AE_perm_invariant_latent_dim': '-PERM_INV_',
                })
                keys.update({
                    'AE_perm_invariant_feature': '-FEAT_',
                })
            else:
                keys.update({
                    'input_dim': '-DIM_',
                })
            keys.update({
                'AE_layers_num': '-AUTO_',
                'AE_layers_size': 'x',
                'activation_str_general': '-ACT_',
                'AE_residual': '-RES_',
                'dropout_keep_prob': '-DKP_',
                'latent_state_dim': '-LD_',
                # 'output_forecasting_loss':'-ForLoss_',
                # 'latent_forecasting_loss':'-DynLoss_',
                # 'reconstruction_loss':'-RecLoss_',
            })
        if self.has_rnn and with_rnn:  # RNN MODE
            keys.update({
                'RNN_cell_type': '-C_',
                # 'RNN_activation_str':'-ACT_',
                'RNN_layers_num': '-R_',
                'RNN_layers_size': 'x',
                # 'zoneout_keep_prob':'-ZKP_',
                # 'iterative_loss_length':'-ITLL_',
                'sequence_length': '-SL_',
                # 'prediction_horizon':'-PH_',
                # 'num_test_ICS':'-NICS_',
            })
            # if self.params["iterative_loss_length"] > 0 and self.params["iterative_loss_in_name"]:
            #     keys.update({
            #         'iterative_loss_length': '-ITLL_',
            #     })
            #     keys.update({
            #         'iterative_loss_length_weight': '-W_',
            #     })
            if self.params["iterative_loss_schedule_and_gradient"] not in [
                    "none"
            ]:
                keys.update({
                    'iterative_loss_schedule_and_gradient': '-ITS_',
                })

            if self.RNN_MDN_bool:
                keys.update({
                    'RNN_MDN_distribution': '-R-MDN_',
                })
                keys.update({
                    'RNN_MDN_kernels': '-R-KERN_',
                })
                keys.update({
                    'RNN_MDN_hidden_units': '-R-HIDDEN_',
                })
                keys.update({
                    'RNN_MDN_sigma_max': '-R-SMax_',
                })
                if self.RNN_MDN_multivariate:
                    keys.update({
                        'RNN_MDN_multivariate': '-R-MULT_',
                    })
                if self.RNN_MDN_multivariate_covariance_layer:
                    keys.update({
                        'RNN_MDN_multivariate_covariance_layer':
                        '-RNN-MULT_COV_',
                    })


        if self.params["random_seed_in_name"]:
            keys.update({
                'random_seed': '-RS_',
            })
            
        return keys

    def createModelName(self):
        keys = self.getKeysInModelName()
        str_ = "GPU-" * self.gpu + "ARNN"
        for key in keys:
            str_ += keys[key] + "{:}".format(self.params[key])
        return str_

    def createAutoencoderName(self):
        keys = self.getKeysInModelName(with_rnn=False)
        str_ = "GPU-" * self.gpu + "ARNN"
        for key in keys:
            str_ += keys[key] + "{:}".format(self.params[key])
        return str_

    def makeDirectories(self):
        os.makedirs(self.getModelDir(), exist_ok=True)
        os.makedirs(self.getFigureDir(), exist_ok=True)
        os.makedirs(self.getResultsDir(), exist_ok=True)
        os.makedirs(self.getLogFileDir(), exist_ok=True)

    def getModelDir(self):
        model_dir = self.saving_path + self.model_dir + self.model_name
        return model_dir

    def getFigureDir(self, unformatted=False):
        fig_dir = self.saving_path + self.fig_dir + self.model_name
        return fig_dir

    def getResultsDir(self, unformatted=False):
        results_dir = self.saving_path + self.results_dir + self.model_name
        return results_dir

    def getLogFileDir(self, unformatted=False):
        logfile_dir = self.saving_path + self.logfile_dir + self.model_name
        return logfile_dir

    def printParams(self):
        self.n_trainable_parameters = self.model.countTrainableParams()
        self.n_model_parameters = self.model.countParams()
        # Print parameter information:
        print("# Trainable params {:}/{:}".format(self.n_trainable_parameters,
                                                  self.n_model_parameters))
        return 0

    def declareOptimizer(self, lr):
        # print("LEARNING RATE: {}".format(lr))
        if self.train_AE_only:
            params = self.model.getAutoencoderParams()
        elif self.train_rnn_only:
            params = self.model.getRNNParams()
        else:
            params = self.model_parameters

        # Weight decay only when training the autoencoder
        if self.has_rnn and not self.train_AE_only:
            weight_decay = 0.0
            if self.weight_decay > 0: print("No weight decay in RNN training.")
        else:
            weight_decay = self.weight_decay

        print("LEARNING RATE: {:}, WEIGHT DECAY: {:}".format(
            lr, self.weight_decay))

        if self.optimizer_str == "adam":
            self.optimizer = torch.optim.Adam(params,
                                              lr=lr,
                                              weight_decay=weight_decay)

        elif self.optimizer_str == "sgd":
            self.optimizer = torch.optim.SGD(params,
                                             lr=lr,
                                             momentum=0.9,
                                             weight_decay=weight_decay)

        elif self.optimizer_str == "rmsprop":
            self.optimizer = torch.optim.RMSprop(params,
                                                 lr=lr,
                                                 weight_decay=weight_decay)

        elif self.optimizer_str == "adabelief":
            from adabelief_pytorch import AdaBelief
            self.optimizer = AdaBelief(params,
                                       lr=lr,
                                       eps=1e-16,
                                       betas=(0.9, 0.999),
                                       weight_decouple=True,
                                       rectify=False)
        else:
            raise ValueError("Optimizer {:} not recognized.".format(
                self.optimizer_str))

    def getZeroRnnHiddenState(self, batch_size):
        if self.has_rnn:
            hidden_state = []
            for ln in self.layers_rnn:
                hidden_state.append(
                    self.getZeroRnnHiddenStateLayer(batch_size, ln))
            hidden_state = torch.stack(hidden_state)
            hidden_state = self.getModel().transposeHiddenState(hidden_state)
        else:
            hidden_state = []
        return hidden_state

    def getZeroRnnHiddenStateLayer(self, batch_size, hidden_units):
        hx = Variable(torch.zeros(batch_size, hidden_units))
        if self.params["RNN_cell_type"] == "lstm":
            cx = Variable(torch.zeros(batch_size, hidden_units))
            hidden_state = torch.stack([hx, cx])
            return hidden_state
        elif self.params["RNN_cell_type"] in ["gru", "plain"]:
            return hx
        else:
            raise ValueError("Unknown cell type {}.".format(
                self.params["RNN_cell_type"]))

    def plotBatchNumber(self, i, n_batches, is_train):
        if self.display_output:
            str_ = "\n" + is_train * "TRAINING: " + (
                not is_train) * "EVALUATION"
            print("{:s} batch {:d}/{:d},  {:f}%".format(
                str_, int(i + 1), int(n_batches), (i + 1) / n_batches * 100.))
            sys.stdout.write("\033[F")

    def sendHiddenStateToGPU(self, h_state):
        if self.has_rnn:
            return h_state.cuda()
        else:
            return h_state

    def detachHiddenState(self, h_state):
        if self.has_rnn:
            return h_state.detach()
        else:
            return h_state

    def parseMDNOutput(
        self,
        output,
        target,
        is_MDN_multivariate,
        MDN_kernels,
        is_perm_inv,
        MDN_dist,
    ):
        if self.channels == 1:
            K, T, N_O, Dx = target.size()
        elif self.channels == 0:
            K, T, N_O = target.size()
            Dx = 1
        else:
            raise ValueError("Invalid channel size (not implemented).")

        # The MDN output is the mixing coefficients the the MDN variables
        pi, MDN_var1, MDN_var2, MDN_var3, MDN_var4 = output

        if is_MDN_multivariate:
            K_, T_, NUM_KERNELS, N_MDN_O = MDN_var1.size()

            if is_perm_inv:
                assert (N_MDN_O == self.Dx)
            else:
                # print(MDN_var1.size())
                # assert(N_MDN_O==self.input_dim)
                assert (N_MDN_O == target.size()[2])

            assert (K == K_)
            assert (T == T_)
            assert (self.Dx == Dx)
            assert (NUM_KERNELS == MDN_kernels)

            if is_perm_inv:
                target = target.view(K * T * N_O, Dx)
                MDN_var1 = MDN_var1.view(K * T, NUM_KERNELS, N_MDN_O)
                MDN_var2 = MDN_var2.view(K * T, NUM_KERNELS, N_MDN_O, N_MDN_O)
                pi = pi.view(K * T, NUM_KERNELS)
                MDN_var1 = self.repeatAlongDim(var=MDN_var1,
                                               axis=0,
                                               repeat_times=N_O,
                                               interleave=True)
                MDN_var2 = self.repeatAlongDim(var=MDN_var2,
                                               axis=0,
                                               repeat_times=N_O,
                                               interleave=True)
                pi = self.repeatAlongDim(var=pi,
                                         axis=0,
                                         repeat_times=N_O,
                                         interleave=True)
            else:
                assert (N_O == N_MDN_O)
                target = target.view(K * T, N_O)
                MDN_var1 = MDN_var1.view(K * T, NUM_KERNELS, N_MDN_O)
                pi = pi.view(K * T, NUM_KERNELS)
                MDN_var2 = MDN_var2.view(K * T, NUM_KERNELS, N_MDN_O, N_MDN_O)

        elif MDN_dist == "normal":
            K_, T_, N_MDN_O, NUM_KERNELS = MDN_var1.size()
            assert (N_MDN_O == N_O * self.Dx)
            assert (K == K_)
            assert (T == T_)
            assert (self.Dx == Dx)
            assert (NUM_KERNELS == MDN_kernels)

            target = target.view(K * T, N_O * Dx)
            MDN_var1 = MDN_var1.view(K * T, N_MDN_O, NUM_KERNELS)
            MDN_var2 = MDN_var2.view(K * T, N_MDN_O, NUM_KERNELS)
            pi = pi.view(K * T, N_MDN_O, NUM_KERNELS)
        elif MDN_dist in ["alanine", "trp"]:
            K_, T_, N_MDN_O_BONDS, NUM_KERNELS = MDN_var1.size()
            K_, T_, N_MDN_O_ANGLES, NUM_KERNELS = MDN_var3.size()
            N_MDN_O = N_MDN_O_BONDS + N_MDN_O_ANGLES
            assert (N_MDN_O == N_O)

            target = target.view(K * T, N_O * Dx)
            MDN_var1 = MDN_var1.view(K * T, N_MDN_O_BONDS, NUM_KERNELS)
            MDN_var2 = MDN_var2.view(K * T, N_MDN_O_BONDS, NUM_KERNELS)
            MDN_var3 = MDN_var3.view(K * T, N_MDN_O_ANGLES, NUM_KERNELS)
            MDN_var4 = MDN_var4.view(K * T, N_MDN_O_ANGLES, NUM_KERNELS)
            pi = pi.view(K * T, N_MDN_O, NUM_KERNELS)
        return target, pi, MDN_var1, MDN_var2, MDN_var3, MDN_var4

    def getLoss(
        self,
        output,
        target,
        is_latent=False,
        is_MDN_loss=False,
        is_MDN_multivariate=False,
        is_perm_inv=False,
        MDN_kernels=0,
        MDN_loss_fn=None,
        MDN_dist=None,
    ):
        if not is_MDN_loss:
            # Normal loss, in case NO mixture density network (MDN)
            # Difference
            loss = output - target
            # Square difference
            loss = loss.pow(2.0)
            # Mean over all dimensions
            loss = loss.mean(2)
            # Mean over all batches
            loss = loss.mean(0)
            # Mean over all time-steps
            loss = loss.mean()
            return loss

        else:
            # MDN loss

            target, pi, MDN_var1, MDN_var2, MDN_var3, MDN_var4 = self.parseMDNOutput(
                output,
                target,
                is_MDN_multivariate,
                MDN_kernels,
                is_perm_inv,
                MDN_dist,
            )

            loss = MDN_loss_fn(target, pi, MDN_var1, MDN_var2, MDN_var3,
                               MDN_var4)
            return loss

    def repeatAlongDim(self, var, axis, repeat_times, interleave=False):
        if not interleave:
            repeat_idx = len(var.size()) * [1]
            repeat_idx[axis] = repeat_times
            var = var.repeat(*repeat_idx)
        else:
            var = var.repeat_interleave(repeat_times, dim=axis)
        return var

    def trainOnBatch(self, batch_of_sequences, is_train=False):
        batch_size = len(batch_of_sequences)
        initial_hidden_states = self.getZeroRnnHiddenState(batch_size)

        T = np.shape(batch_of_sequences)[1]

        losses_vec = []

        assert (
            T - 1
        ) % self.sequence_length == 0, "The time-steps in the sequence need to be divisible by the sequence_length ((T-1) % self.sequence_length) == 0, -> {:} % {:} ==0".format(
            T - 1, self.sequence_length)
        num_propagations = int((T - 1) / self.sequence_length)
        # print("num_propagations")
        # print(num_propagations)

        predict_on = self.sequence_length
        for p in range(num_propagations):
            # Setting the optimizer to zero grad
            self.optimizer.zero_grad()

            # Getting the batch
            input_batch = batch_of_sequences[:, predict_on -
                                             self.sequence_length:predict_on]
            target_batch = batch_of_sequences[:, predict_on -
                                              self.sequence_length +
                                              1:predict_on + 1]

            # Transform to pytorch and forward the network
            input_batch = self.torch_dtype(input_batch)
            target_batch = self.torch_dtype(target_batch)
            if self.gpu:
                # SENDING THE TENSORS TO CUDA
                input_batch = input_batch.cuda()
                target_batch = target_batch.cuda()
                initial_hidden_states = self.sendHiddenStateToGPU(
                    initial_hidden_states)

            input_batch = input_batch.contiguous()
            target_batch = target_batch.contiguous()

            if not is_train and self.iterative_loss_validation:
                # set iterative forecasting to True in case of validation
                iterative_forecasting_prob = 1.0
                iterative_propagation_is_latent = False
                is_iterative_forecasting = True
                iterative_forecasting_gradient = False

            elif self.iterative_loss_schedule_and_gradient in ["none"]:
                iterative_forecasting_prob = 0.0
                iterative_propagation_is_latent = False
                is_iterative_forecasting = False
                iterative_forecasting_gradient = 0

            elif any(x in self.iterative_loss_schedule_and_gradient
                     for x in ["linear", "inverse_sigmoidal", "exponential"]):
                # assert(self.iterative_loss_validation == 1)
                iterative_forecasting_prob = self.getIterativeForecastingProb(
                    self.epochs_iter_global,
                    self.iterative_loss_schedule_and_gradient)
                iterative_propagation_is_latent = False
                is_iterative_forecasting = True
                iterative_forecasting_gradient = self.iterative_loss_gradient

            else:
                raise ValueError(
                    "self.iterative_loss_schedule_and_gradient={:} not recognized."
                    .format(self.iterative_loss_schedule_and_gradient))
            self.iterative_forecasting_prob = iterative_forecasting_prob

            output_batch, last_hidden_state, latent_states, latent_states_pred, RNN_outputs, input_batch_decoded, time_latent_prop = self.model.forward(
                input_batch,
                initial_hidden_states,
                is_train=is_train,
                is_iterative_forecasting=is_iterative_forecasting,
                iterative_forecasting_prob=iterative_forecasting_prob,
                iterative_forecasting_gradient=iterative_forecasting_gradient,
                iterative_propagation_is_latent=iterative_propagation_is_latent,
                horizon=None,
                input_is_latent=False,
            )

            if self.output_forecasting_loss:

                # The output is:
                # 1. either the output of the RNN in case of no autoencoders,
                # 2. or the output of the autoencoder

                if self.MDN_bool and self.has_autoencoder:
                    is_MDN_loss = True
                    is_MDN_multivariate = self.MDN_multivariate
                    is_perm_inv = self.AE_perm_invariant_latent_dim
                    MDN_kernels = self.MDN_kernels
                    MDN_loss_fn = self.model.DECODER[-1].MDN_loss_fn
                    MDN_dist = self.MDN_distribution

                elif self.RNN_MDN_bool and not self.has_autoencoder:
                    is_MDN_loss = True
                    is_MDN_multivariate = self.RNN_MDN_multivariate
                    is_perm_inv = False
                    MDN_kernels = self.RNN_MDN_kernels
                    MDN_loss_fn = self.model.RNN_OUTPUT[-1].MDN_loss_fn
                    MDN_dist = self.RNN_MDN_distribution
                else:
                    is_MDN_loss = False
                    is_MDN_multivariate = False
                    is_perm_inv = False
                    MDN_kernels = False
                    MDN_loss_fn = None
                    MDN_dist = None

                loss_fwd = self.getLoss(
                    output_batch,
                    target_batch,
                    is_MDN_loss=is_MDN_loss,
                    is_MDN_multivariate=is_MDN_multivariate,
                    is_perm_inv=is_perm_inv,
                    MDN_kernels=MDN_kernels,
                    MDN_loss_fn=MDN_loss_fn,
                    MDN_dist=MDN_dist,
                )
            else:
                loss_fwd = self.torchZero()

            # Checking the output matches the target in case of no MDN
            if not self.MDN_bool and not self.RNN_MDN_bool:

                if self.has_rnn:
                    assert output_batch.size() == target_batch.size(
                    ), "ERROR: Output of network does not match with target."
                else:
                    assert input_batch.size() == input_batch_decoded.size(
                    ), "ERROR: Output of DECODER network does not match INPUT."

            # print("##################")
            # print(output_batch.size())
            # print(target_batch.size())
            # print(loss_fwd)
            # # print(latent_states_pred.size())
            # # print(output_batch[0, 0, 0])
            # # print(output_batch[0, 1, 0])
            # # print(output_batch[0, -1, 0])
            # print("##################")

            if self.latent_forecasting_loss:
                if self.RNN_MDN_bool:
                    is_MDN_loss = True
                    is_MDN_multivariate = self.RNN_MDN_multivariate
                    MDN_kernels = self.RNN_MDN_kernels
                    MDN_loss_fn = self.model.RNN_OUTPUT[-1].MDN_loss_fn
                    MDN_dist = self.RNN_MDN_distribution
                    # print(latent_states_pred[0].size())
                    # print(latent_states_pred[1].size())
                    # print(latent_states_pred[2].size())
                    # print(latent_states[:, 1:, :].size())

                    # print(latent_states_pred[0].size())
                    # print(latent_states_pred[1].size())
                    # print(len(latent_states_pred))
                    # print(ark)
                    latent_states_pred = latent_states_pred if MDN_dist in [
                        "alanine", "trp"
                    ] else latent_states_pred[:3]
                    outputs = [
                        temp[:, :-1].contiguous()
                        for temp in latent_states_pred
                    ]
                    outputs.append(None)
                    outputs.append(None)
                    # outputs = [temp[:, :-1] for temp in latent_states_pred]
                    # print(outputs[0].size())
                    # print(outputs[1].size())
                    # print(outputs[2].size())
                    # print(latent_states[:, 1:, :].size())
                    # print(ark)
                    targets = latent_states[:, 1:, :].contiguous()
                else:
                    is_MDN_loss = False
                    is_MDN_multivariate = False
                    MDN_kernels = False
                    MDN_loss_fn = None
                    MDN_dist = None
                    outputs = latent_states_pred[:, :-1, :]
                    targets = latent_states[:, 1:, :]
                loss_dyn_fwd = self.getLoss(
                    outputs,
                    targets,
                    is_latent=True,
                    is_MDN_loss=is_MDN_loss,
                    is_MDN_multivariate=is_MDN_multivariate,
                    MDN_kernels=MDN_kernels,
                    MDN_loss_fn=MDN_loss_fn,
                    MDN_dist=MDN_dist,
                )
            else:
                loss_dyn_fwd = self.torchZero()

            if self.reconstruction_loss:
                if self.MDN_bool and self.has_autoencoder:
                    is_MDN_loss = True
                    is_MDN_multivariate = self.MDN_multivariate
                    is_perm_inv = self.AE_perm_invariant_latent_dim
                    MDN_kernels = self.MDN_kernels
                    MDN_loss_fn = self.model.DECODER[-1].MDN_loss_fn
                    MDN_dist = self.MDN_distribution
                else:
                    is_MDN_loss = False
                    is_MDN_multivariate = False
                    is_perm_inv = False
                    MDN_kernels = False
                    MDN_loss_fn = None
                    MDN_dist = None

                loss_auto_fwd = self.getLoss(
                    input_batch_decoded,
                    input_batch,
                    is_MDN_loss=is_MDN_loss,
                    is_MDN_multivariate=is_MDN_multivariate,
                    is_perm_inv=is_perm_inv,
                    MDN_kernels=MDN_kernels,
                    MDN_loss_fn=MDN_loss_fn,
                    MDN_dist=MDN_dist,
                )
            else:
                loss_auto_fwd = self.torchZero()

            # CONSTRUCTING THE LOSS
            loss_batch = 0.0
            num_losses = 0.0

            # ADDING THE FORWARD LOSS (be carefull, if loss_batch=loss_fwd, it is passed by reference!)
            if self.output_forecasting_loss:
                loss_batch += loss_fwd
                num_losses += 1.0
            if self.latent_forecasting_loss:
                loss_batch += loss_dyn_fwd
                num_losses += 1.0
            if self.reconstruction_loss:
                loss_batch += loss_auto_fwd
                num_losses += 1.0

            loss_batch = loss_batch / num_losses

            if is_train:
                # loss_batch.requires_grad = True
                # loss_batch.backward(retain_graph=True)
                loss_batch.backward()
                self.optimizer.step()
                # if self.optimizer_str == "sgd": self.scheduler.step()

            loss_batch = loss_batch.cpu().detach().numpy()

            loss_fwd = loss_fwd.cpu().detach().numpy()

            loss_dyn_fwd = loss_dyn_fwd.cpu().detach().numpy()

            loss_auto_fwd = loss_auto_fwd.cpu().detach().numpy()

            losses_batch = [loss_batch, loss_fwd, loss_dyn_fwd, loss_auto_fwd]

            last_hidden_state = self.detachHiddenState(last_hidden_state)

            # APPENDING LOSSES
            losses_vec.append(losses_batch)
            initial_hidden_states = last_hidden_state

            #################################
            ### UPDATING BATCH IDX
            #################################

            predict_on = predict_on + self.sequence_length

        losses = np.mean(np.array(losses_vec), axis=0)
        return losses, iterative_forecasting_prob

    def getIterativeForecastingProb(self, epoch, schedule):
        assert (schedule in [
            "linear_with_gradient", "linear_without_gradient",
            "inverse_sigmoidal_with_gradient",
            "inverse_sigmoidal_without_gradient", "exponential_with_gradient",
            "exponential_without_gradient"
        ])
        E = self.max_epochs
        if "linear" in schedule:
            c = 1.0 / E
            prob = c * epoch

        elif "exponential" in schedule:
            k = np.exp(np.log(0.001) / E)
            prob = 1 - np.power(k, epoch)

        elif "inverse_sigmoidal" in schedule:
            # Use the lambertw function
            # Get the k coefficient, by setting the inflection point to E/2 or E/4
            from scipy.special import lambertw
            # k = np.real(E/2.0/lambertw(E/2.0))
            k = np.real(E / 4.0 / lambertw(E / 4.0))
            prob = 1 - k / (k + np.exp(epoch / k))

        return prob

    def torchZero(self):
        return self.torch_dtype([0.0])[0]

    def trainEpoch(self, data_loader, is_train=False):
        epoch_losses_vec = []
        iterative_forecasting_prob_vec = []
        # print("# trainEpoch() #")
        for batch_of_sequences in data_loader:
            # K, T, C, Dx, Dy
            losses, iterative_forecasting_prob = self.trainOnBatch(
                batch_of_sequences, is_train=is_train)
            epoch_losses_vec.append(losses)
            iterative_forecasting_prob_vec.append(iterative_forecasting_prob)
        epoch_losses = np.mean(np.array(epoch_losses_vec), axis=0)
        iterative_forecasting_prob = np.mean(
            np.array(iterative_forecasting_prob_vec), axis=0)
        time_ = time.time() - self.start_time
        return epoch_losses, iterative_forecasting_prob, time_

    def getDataLoader(self, data_path, data_info_dict, loader_params):
        data_loader = utils.getHDF5dataLoader(data_path, loader_params,
                                              data_info_dict)
        return data_loader

    def initializeTimeKernels(self):
        dict_ = {
            "times_loss_forward": [],
            "times_loss_epoch": [],
        }
        return dict_

    def train(self):

        if self.gpu:
            self.gpu_monitor_process = utils.GPUMonitor(self.params["gpu_monitor_every"])

        # Loading data in batches
        loader_params = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': 0,
        }
        data_loader_train = self.getDataLoader(self.data_path_train,
                                               self.data_info_dict,
                                               loader_params)
        data_loader_val = self.getDataLoader(self.data_path_val,
                                             self.data_info_dict,
                                             loader_params)

        self.time_kernels = self.initializeTimeKernels()

        self.declareOptimizer(self.learning_rate)

        # Check if retraining
        if self.retrain == 1:
            print("RESTORING MODEL")
            self.loadModel()
            # self.getModel().load_state_dict(torch.load(self.saving_model_path))
        elif self.train_rnn_only == 1:
            print("## LOADING AUTOENCODER MODEL: \n")
            self.loadAutoencoderModel()
            # SAVING THE INITIAL MODEL
            print("SAVING THE INITIAL MODEL")
            torch.save(self.getModel().state_dict(), self.saving_model_path)

        self.loss_total_train_vec = []
        self.loss_total_val_vec = []

        self.losses_train_vec = []
        self.losses_val_vec = []

        self.losses_time_train_vec = []
        self.losses_time_val_vec = []

        self.ifp_train_vec = []
        self.ifp_val_vec = []

        isWallTimeLimit = False

        # Termination criterion:
        # If the training procedure completed the maximum number of epochs

        # Learning rate decrease criterion:
        # If the validation loss does not improve for some epochs (patience)
        # the round is terminated, the learning rate decreased and training
        # proceeds in the next round.

        self.epochs_iter = 0
        self.epochs_iter_global = self.epochs_iter
        self.rounds_iter = 0
        # TRACKING
        self.tqdm = tqdm(total=self.max_epochs)
        while self.epochs_iter < self.max_epochs and self.rounds_iter < self.max_rounds:
            isWallTimeLimit = self.trainRound(data_loader_train,
                                              data_loader_val)
            # INCREMENTING THE ROUND NUMBER
            if isWallTimeLimit: break

        # If the training time limit was not reached, save the model...
        if not isWallTimeLimit:
            if self.epochs_iter == self.max_epochs:
                print(
                    "## Training finished. Maximum number of epochs reached.")
            elif self.rounds_iter == self.max_rounds:
                print(
                    "## Training finished. Maximum number of rounds reached.")
            else:
                if self.gpu: self.gpu_monitor_process.stop()
                print(self.rounds_iter)
                print(self.epochs_iter)
                raise ValueError("## Training finished. I do not know why!")
            self.saveModel()
            utils.plotTrainingLosses(self, self.loss_total_train_vec,
                                     self.loss_total_val_vec,
                                     self.min_val_total_loss)
            utils.plotAllLosses(self, self.losses_train_vec,
                                self.losses_time_train_vec,
                                self.losses_val_vec, self.losses_time_val_vec,
                                self.min_val_total_loss)
            utils.plotSchedule(self, self.ifp_train_vec, self.ifp_val_vec)
        if self.gpu: self.gpu_monitor_process.stop()

    def printLosses(self, label, losses):
        # PRINT ALL THE NON-ZERO LOSSES
        self.losses_labels = [
            "TOTAL",
            "FWD",
            "DYN-FWD",
            "AUTO-REC",
        ]
        idx = np.nonzero(losses)[0]
        to_print = "# {:s}-losses: ".format(label)
        for i in range(len(idx)):
            to_print += "{:}={:1.2E} |".format(self.losses_labels[idx[i]],
                                               losses[idx[i]])
        print(to_print)

    def printEpochStats(self, epoch_time_start, epochs_iter, epochs_in_round,
                        losses_train, losses_val):
        epoch_duration = time.time() - epoch_time_start
        time_covered = epoch_duration * epochs_iter
        time_total = epoch_duration * self.max_epochs
        percent = time_covered / time_total * 100
        print("###################################################")
        label = "EP={:} - R={:} - ER={:} - [ TIME= {:}, {:} / {:} - {:.2f} %] - LR={:1.2E}".format(
            epochs_iter, self.rounds_iter, epochs_in_round,
            utils.secondsToTimeStr(epoch_duration),
            utils.secondsToTimeStr(time_covered),
            utils.secondsToTimeStr(time_total), percent,
            self.learning_rate_round)
        print(label)
        self.printLosses("TRAIN", losses_train)
        self.printLosses("VAL  ", losses_val)

    def printLearningRate(self):
        for param_group in self.optimizer.param_groups:
            print("Current learning rate = {:}".format(param_group["lr"]))
        return 0

    def getModel(self):
        if (not self.gpu) or (self.device_count <= 1):
            return self.model
        elif self.gpu and self.device_count > 1:
            return self.model.module
        else:
            raise ValueError("Value of self.gpu {:} not recognized.".format(
                self.gpu))

    def trainRound(self, data_loader_train, data_loader_val):
        # Check if retraining of a model is requested else random initialization of the weights
        isWallTimeLimit = False

        # SETTING THE INITIAL LEARNING RATE
        if self.rounds_iter == 0:
            self.learning_rate_round = self.learning_rate
            self.previous_round_converged = 0
        elif self.previous_round_converged == 0:
            self.learning_rate_round = self.learning_rate_round
            self.previous_round_converged = 0
        elif self.previous_round_converged == 1:
            self.previous_round_converged = 0
            self.learning_rate_round = self.learning_rate_round / 10

        # OPTIMIZER HAS TO BE DECLARED
        self.declareOptimizer(self.learning_rate_round)

        if self.rounds_iter > 0:
            # RESTORE THE MODEL
            print("RESTORING PYTORCH MODEL")
            self.getModel().load_state_dict(torch.load(self.saving_model_path))
        else:
            # SAVING THE INITIAL MODEL
            print("SAVING THE INITIAL MODEL")
            torch.save(self.getModel().state_dict(), self.saving_model_path)
            pass

        # print(self.getModel().state_dict())
        # print(ark)

        print("##### ROUND: {:}, LEARNING RATE={:} #####".format(
            self.rounds_iter, self.learning_rate_round))

        time_loss_epoch_start = time.time()
        losses_train, ifp_train, time_train = self.trainEpoch(
            data_loader_train, is_train=False)
        if self.iterative_loss_validation: assert (ifp_train == 1.0)

        losses_val, ifp_val, time_val = self.trainEpoch(data_loader_val,
                                                        is_train=False)
        if self.iterative_loss_validation: assert (ifp_val == 1.0)

        time_loss_epoch_end = time.time()
        time_loss_epoch = time_loss_epoch_end - time_loss_epoch_start
        self.time_kernels["times_loss_epoch"].append(time_loss_epoch)

        label = "INITIAL (NEW ROUND):  EP{:} - R{:}".format(
            self.epochs_iter, self.rounds_iter)
        print(label)

        self.printLosses("TRAIN", losses_train)
        self.printLosses("VAL  ", losses_val)

        self.min_val_total_loss = losses_val[0]
        self.loss_total_train = losses_train[0]

        rnn_loss_round_train_vec = []
        rnn_loss_round_val_vec = []

        rnn_loss_round_train_vec.append(losses_train[0])
        rnn_loss_round_val_vec.append(losses_val[0])

        self.loss_total_train_vec.append(losses_train[0])
        self.loss_total_val_vec.append(losses_val[0])

        self.losses_train_vec.append(losses_train)
        self.losses_time_train_vec.append(time_train)
        self.losses_val_vec.append(losses_val)
        self.losses_time_val_vec.append(time_val)

        self.ifp_train_vec.append(ifp_train)
        self.ifp_val_vec.append(ifp_val)

        for epochs_iter in range(self.epochs_iter, self.max_epochs + 1):
            epoch_time_start = time.time()
            epochs_in_round = epochs_iter - self.epochs_iter
            self.epochs_iter_global = epochs_iter

            losses_train, ifp_train, time_train = self.trainEpoch(
                data_loader_train, is_train=True)
            losses_val, ifp_val, time_val = self.trainEpoch(data_loader_val,
                                                            is_train=False)
            rnn_loss_round_train_vec.append(losses_train[0])
            rnn_loss_round_val_vec.append(losses_val[0])
            self.loss_total_train_vec.append(losses_train[0])
            self.loss_total_val_vec.append(losses_val[0])

            self.losses_train_vec.append(losses_train)
            self.losses_time_train_vec.append(time_train)
            self.losses_val_vec.append(losses_val)
            self.losses_time_val_vec.append(time_val)

            self.ifp_val_vec.append(ifp_val)
            self.ifp_train_vec.append(ifp_train)

            self.printEpochStats(epoch_time_start, epochs_iter,
                                 epochs_in_round, losses_train, losses_val)

            if losses_val[0] < self.min_val_total_loss:
                print("SAVING MODEL!!!")
                self.min_val_total_loss = losses_val[0]
                self.loss_total_train = losses_train[0]
                torch.save(self.getModel().state_dict(),
                           self.saving_model_path)

            if epochs_in_round > self.overfitting_patience:
                if all(self.min_val_total_loss <
                       rnn_loss_round_val_vec[-self.overfitting_patience:]):
                    self.previous_round_converged = True
                    break

            # # LEARNING RATE SCHEDULER (PLATEU ON VALIDATION LOSS)
            # if self.optimizer_str == "adam": self.scheduler.step(losses_val[0])
            self.tqdm.update(1)
            isWallTimeLimit = self.isWallTimeLimit()
            if isWallTimeLimit:
                break

        self.rounds_iter += 1
        self.epochs_iter = epochs_iter
        return isWallTimeLimit

    def isWallTimeLimit(self):
        training_time = time.time() - self.start_time
        if training_time > self.reference_train_time:
            print("## Maximum train time reached: saving model... ##")
            self.tqdm.close()
            self.saveModel()
            utils.plotTrainingLosses(self, self.loss_total_train_vec,
                                     self.loss_total_val_vec,
                                     self.min_val_total_loss)
            utils.plotAllLosses(self, self.losses_train_vec,
                                self.losses_time_train_vec,
                                self.losses_val_vec, self.losses_time_val_vec,
                                self.min_val_total_loss)
            utils.plotSchedule(self, self.ifp_train_vec, self.ifp_val_vec)
            return True
        else:
            return False

    def delete(self):
        pass

    def saveModel(self):
        print("Recording time...")
        self.total_training_time = time.time() - self.start_time
        if hasattr(self, 'loss_total_train_vec'):
            if len(self.loss_total_train_vec) != 0:
                self.training_time = self.total_training_time / len(
                    self.loss_total_train_vec)
            else:
                self.training_time = self.total_training_time
        else:
            self.training_time = self.total_training_time

        print("Total training time per epoch is {:}".format(
            utils.secondsToTimeStr(self.training_time)))
        print("Total training time is {:}".format(
            utils.secondsToTimeStr(self.total_training_time)))

        self.memory = utils.getMemory()
        print("Script used {:} MB".format(self.memory))

        data = {
            "params": self.params,
            "model_name": self.model_name,
            "memory": self.memory,
            "total_training_time": self.total_training_time,
            "training_time": self.training_time,
            "n_trainable_parameters": self.n_trainable_parameters,
            "n_model_parameters": self.n_model_parameters,
            "RNN_loss_train_vec": self.loss_total_train_vec,
            "RNN_loss_val_vec": self.loss_total_val_vec,
            "RNN_min_val_error": self.min_val_total_loss,
            "RNN_train_error": self.loss_total_train,
            "losses_train_vec": self.losses_train_vec,
            "losses_time_train_vec": self.losses_time_train_vec,
            "losses_val_vec": self.losses_val_vec,
            "losses_time_val_vec": self.losses_time_val_vec,
            "ifp_val_vec": self.ifp_val_vec,
            "ifp_train_vec": self.ifp_train_vec,
        }
        fields_to_write = [
            "memory", "total_training_time", "n_model_parameters",
            "n_trainable_parameters"
        ]
        if self.write_to_log == 1:
            logfile_train = self.getLogFileDir() + "/train.txt"
            print("Writing to log-file in path {:}".format(logfile_train))
            utils.writeToLogFile(self, logfile_train, data, fields_to_write)

        data_path = self.getModelDir() + "/data"
        utils.saveData(data, data_path, "pickle")

    def loadModel(self, in_cpu=False):
        try:
            if not in_cpu and self.gpu:
                print("# LOADING model in GPU.")
                self.getModel().load_state_dict(
                    torch.load(self.saving_model_path))
            else:
                print("# LOADING model in CPU...")
                self.getModel().load_state_dict(
                    torch.load(self.saving_model_path,
                               map_location=torch.device('cpu')))
        except Exception as inst:
            print(
                "MODEL {:s} NOT FOUND. Is hippo mounted? Are you testing ? Did you already train the model?"
                .format(self.saving_model_path))
            raise ValueError(inst)
        try:
            data_path = self.getModelDir() + "/data"
            data = utils.loadData(data_path, "pickle")
            self.loss_total_train_vec = data["RNN_loss_train_vec"]
            self.loss_total_val_vec = data["RNN_loss_val_vec"]
            self.min_val_total_loss = data["RNN_min_val_error"]
            self.losses_time_train_vec = data["losses_time_train_vec"]
            self.losses_time_val_vec = data["losses_time_val_vec"]
            self.losses_val_vec = data["losses_val_vec"]
            self.losses_train_vec = data["losses_train_vec"]
            del data
        except:
            print(
                "Model data not found in path {:s}. Did you already train the model? Continuing without the data."
                .format(data_path))

        return 0

    def loadAutoencoderModel(self, in_cpu=False):
        model_name_autoencoder = self.createAutoencoderName()
        print("Loading autoencoder with name:")
        print(model_name_autoencoder)
        AE_path = self.saving_path + self.model_dir + model_name_autoencoder + "/model"
        # self.getModel().load_state_dict(torch.load(AE_path), strict=False)
        try:
            if not in_cpu and self.gpu:
                print("# LOADING autoencoder model in GPU.")
                self.getModel().load_state_dict(torch.load(AE_path),
                                                strict=False)
            else:
                print("# LOADING autoencoder model in CPU...")
                self.getModel().load_state_dict(torch.load(
                    AE_path, map_location=torch.device('cpu')),
                                                strict=False)
        except Exception as inst:
            print(
                "MODEL {:s} NOT FOUND. Is hippo mounted? Are you testing ? Did you already train the autoencoder ?"
                .format(AE_path))
            raise ValueError(inst)
        AE_data_path = self.saving_path + self.model_dir + model_name_autoencoder + "/data"
        data = utils.loadData(AE_data_path, "pickle")
        del data
        return 0

    def test(self):
        if self.loadModel() == 0:
            if self.gpu: self.gpu_monitor_process = utils.GPUMonitor(self.params["gpu_monitor_every"])

            # MODEL LOADED IN EVALUATION MODE
            with torch.no_grad():
                self.n_warmup = self.sequence_length
                print("WARMING UP STEPS (for statefull RNNs): {:d}".format(
                    self.n_warmup))

                sets_to_test = []
                if self.params["test_on_train"]: sets_to_test.append("train")
                if self.params["test_on_val"]: sets_to_test.append("val")
                if self.params["test_on_test"]: sets_to_test.append("test")
                print("# Testing on datasets {:}".format(sets_to_test))
                for set_ in sets_to_test:
                    self.testOnSet(set_)
            if self.gpu: self.gpu_monitor_process.stop()
        return 0

    def testingRoutine(
        self,
        data_loader,
        dt,
        set_,
    ):

        if self.has_rnn:
            # TESTING ONLY THE RNN
            for testing_mode in self.getRNNTestingModes():
                # Check invalid combinations
                self.checkTestingMode(testing_mode)
                self.testOnMode(data_loader, dt, set_, testing_mode)

        elif self.has_autoencoder:
            # if self.has_autoencoder:
            # TESTING AUTOENCODER
            for testing_mode in self.getAutoencoderTestingModes():
                self.testOnMode(data_loader, dt, set_, testing_mode)

        return 0

    def testOnSet(self, set_="train"):
        print("#####     Testing on set: {:}     ######".format(set_))

        dt = self.data_info_dict["dt"]
        loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 0,
        }

        data_path = self.getDataPath(set_)

        data_loader = self.getDataLoader(data_path, self.data_info_dict,
                                         loader_params)
        self.testingRoutine(data_loader, dt, set_)

        return 0

    def getDataPath(self, set_):
        if set_ == "test":
            data_path = self.data_path_test
        elif set_ == "train":
            data_path = self.data_path_train
        elif set_ == "val":
            data_path = self.data_path_val
        else:
            raise ValueError("Invalid set {:}.".format(set_))
        return data_path

    def checkTestingMode(self, testing_mode):
        if "iterative" in testing_mode and not (
                self.iterative_state_forecasting
                or self.iterative_latent_forecasting):
            raise ValueError
        if "multiscale" in testing_mode and not self.multiscale_forecasting:
            raise ValueError
        if "teacher" in testing_mode and not self.teacher_forcing_forecasting:
            raise ValueError
        return 0

    def testOnMode(self, data_loader, dt, set_, testing_mode):
        assert (testing_mode in self.getTestingModes())
        assert (set_ == "train" or set_ == "test" or set_ == "val")
        print("---- Testing on Mode {:} ----".format(testing_mode))
        # CASE: Testing autoencoder
        if testing_mode == "autoencoder_testing" and self.has_autoencoder:
            if self.has_autoencoder:
                results = self.testAutoencoder(data_loader, dt, set_,
                                               testing_mode)
                data_path = self.getResultsDir() + "/results_{:}_{:}".format(
                    testing_mode, set_)
                utils.saveData(results, data_path, self.save_format)
        # CASE: Testing RNN
        elif self.has_rnn and testing_mode != "autoencoder_testing":
            if self.num_test_ICS > 0:
                results = self.predictIndexes(data_loader, dt, set_,
                                              testing_mode)
                data_path = self.getResultsDir() + "/results_{:}_{:}".format(
                    testing_mode, set_)
                utils.saveData(results, data_path, self.save_format)
            else:
                print(
                    "Model has RNN but no initial conditions set to test num_test_ICS={:}."
                    .format(self.num_test_ICS))
        else:
            raise ValueError(
                "Invalid testing mode {:}/network ({:},{:},{:}) combination. {:}"
                .format(testing_mode, self.has_autoencoder, self.has_rnn))
        return 0

    def getRNNTestingModes(self):
        modes = []
        if self.iterative_state_forecasting:
            modes.append("iterative_state_forecasting")
        if self.iterative_latent_forecasting:
            modes.append("iterative_latent_forecasting")
        if self.teacher_forcing_forecasting:
            modes.append("teacher_forcing_forecasting")
        if self.multiscale_forecasting:
            for micro_steps in self.multiscale_micro_steps_list:
                for macro_steps in self.multiscale_macro_steps_list:
                    mode = "multiscale_forecasting_micro_{:}_macro_{:}".format(
                        int(micro_steps), int(macro_steps))
                    modes.append(mode)
        return modes

    def getRNNMacroZeroModes(self):
        modes = []
        for micro_steps in self.multiscale_micro_steps_list:
            for macro_steps in [0]:
                mode = "multiscale_forecasting_micro_{:}_macro_{:}".format(
                    int(micro_steps), int(macro_steps))
                modes.append(mode)
        return modes

    def getMultiscaleParams(self, testing_mode):
        temp = testing_mode.split("_")
        multiscale_macro_steps = int(float(temp[-1]))
        multiscale_micro_steps = int(float(temp[-3]))
        macro_steps_per_round = []
        micro_steps_per_round = []
        steps = 0
        while (steps < self.prediction_horizon):
            steps_to_go = self.prediction_horizon - steps
            if steps_to_go >= multiscale_macro_steps:
                macro_steps_per_round.append(multiscale_macro_steps)
                steps += multiscale_macro_steps
            elif steps_to_go != 0:
                macro_steps_per_round.append(steps_to_go)
                steps += steps_to_go
            else:
                raise ValueError("This was not supposed to happen.")
            steps_to_go = self.prediction_horizon - steps
            if steps_to_go >= multiscale_micro_steps:
                micro_steps_per_round.append(multiscale_micro_steps)
                steps += multiscale_micro_steps
            elif steps_to_go != 0:
                micro_steps_per_round.append(steps_to_go)
                steps += steps_to_go

        print("macro_steps_per_round: \n{:}".format(macro_steps_per_round))
        print("micro_steps_per_round: \n{:}".format(micro_steps_per_round))
        multiscale_rounds = np.max(
            [len(micro_steps_per_round),
             len(macro_steps_per_round)])
        print("multiscale_rounds: \n{:}".format(multiscale_rounds))
        return multiscale_rounds, macro_steps_per_round, micro_steps_per_round

    def getAutoencoderTestingModes(self):
        return [
            "autoencoder_testing",
        ]

    def getTestingModes(self):
        modes = self.getAutoencoderTestingModes() + self.getRNNTestingModes()
        return modes

    def testAutoencoder(self, data_loader, dt, set_name, testing_mode):

        num_test_ICS = self.num_test_ICS
        if num_test_ICS > len(data_loader):
            print(
                "# NOTE: Not enough ICs in the dataset {:}. The dataset has {:} / {:}."
                .format(set_name, len(data_loader), num_test_ICS))
            num_test_ICS = len(data_loader)

        print("# Testing autoencoder on {:}/{:} initial conditions.".format(
            num_test_ICS, len(data_loader)))
        # Saving only some of the data while discarding others
        latent_states_all = []
        input_decoded_all = []
        input_sequence_all = []
        num_seqs_tested_on = 1

        # Dictionary of error lists
        error_dict = utils.getErrorLabelsDict(self)

        for input_sequence in data_loader:
            if num_seqs_tested_on > num_test_ICS: break
            if self.display_output:
                print("IC {:}/{:}, {:2.3f}%".format(
                    num_seqs_tested_on, num_test_ICS,
                    num_seqs_tested_on / num_test_ICS * 100))

            input_sequence = input_sequence[:, :self.prediction_horizon]

            assert np.shape(input_sequence)[0] == 1
            input_sequence = input_sequence[0]

            initial_hidden_states = self.getZeroRnnHiddenState(1)
            input_sequence = input_sequence[np.newaxis, :]
            input_sequence = self.torch_dtype(input_sequence)
            if self.gpu:
                # SENDING THE TENSORS TO CUDA
                input_sequence = input_sequence.cuda()

            _, _, latent_states, _, _, input_decoded, _ = self.model.forward(
                input_sequence, initial_hidden_states, is_train=False)

            if self.MDN_bool:
                if self.channels == 1:
                    K, T, N_O, Dx = input_sequence.size()
                elif self.channels == 0:
                    K, T, N_O = input_sequence.size()
                    Dx = 1
                else:
                    raise ValueError("Invalid channel size.")

                # print("MDN LOSS")
                pi, MDN_var1, MDN_var2, MDN_var3, MDN_var4 = input_decoded

                if self.MDN_bool:
                    target, pi, MDN_var1, MDN_var2, MDN_var3, MDN_var4 = self.parseMDNOutput(
                        input_decoded,
                        input_sequence,
                        self.MDN_multivariate,
                        self.MDN_kernels,
                        self.AE_perm_invariant_latent_dim,
                        self.MDN_distribution,
                    )

                input_decoded = self.model.DECODER[-1].sampleFromOutput(
                    pi, MDN_var1, MDN_var2, MDN_var3, MDN_var4)
                # print(np.shape(input_decoded))
                # print(ark)

                if self.channels == 0:
                    input_decoded = np.reshape(input_decoded, (K, T, N_O))
                elif self.channels == 1:
                    input_decoded = np.reshape(input_decoded, (K, T, N_O, Dx))
                else:
                    raise ValueError("Not implemented.")
                input_decoded = input_decoded[0]
            else:
                input_decoded = input_decoded.cpu().detach().numpy()[0]

            input_sequence = input_sequence.cpu().detach().numpy()[0]
            latent_states = latent_states.cpu().detach().numpy()[0]

            input_sequence = self.data_info_dict["scaler"].descaleData(
                input_sequence, single_sequence=True)
            input_decoded = self.data_info_dict["scaler"].descaleData(
                input_decoded, single_sequence=True)

            errors = utils.computeErrors(
                input_sequence,
                input_decoded,
                compute_r2=self.data_info_dict["compute_r2"])
            # Updating the error
            for error in errors:
                error_dict[error].append(errors[error])

            latent_states_all.append(latent_states)
            input_decoded_all.append(input_decoded)
            input_sequence_all.append(input_sequence)
            num_seqs_tested_on += 1

        # print(latent_states_all)
        # print(ark)
        input_sequence_all = np.array(input_sequence_all)
        input_decoded_all = np.array(input_decoded_all)

        print(np.shape(input_sequence_all))
        print(np.shape(input_decoded_all))

        # Computing the average over time and initial conditions
        error_dict_avg = {}
        for key in error_dict:
            print(np.shape(error_dict[key]))
            # Average over time
            temp = np.mean(error_dict[key], axis=1)
            # Average over initial conditions
            temp = np.mean(temp, axis=0)
            error_dict_avg[key + "_avg"] = temp
        utils.printErrors(error_dict_avg)

        # Computing additional errors based on all predictions (e.g. frequency spectra)
        additional_results_dict, additional_errors_dict = utils.computeAdditionalResults(
            self, input_sequence_all, input_decoded_all, dt)
        error_dict_avg = {**error_dict_avg, **additional_errors_dict}

        state_statistics = utils.computeStateDistributionStatistics(
            input_sequence_all,
            input_decoded_all,
            statistics_per_state=self.data_info_dict["statistics_per_state"],
            statistics_cummulative=self.
            data_info_dict["statistics_cummulative"],
            statistics_per_timestep=self.
            data_info_dict["statistics_per_timestep"],
        )
        state_statistics = systems.computeStateDistributionStatisticsSystem(
            self, state_statistics, input_sequence_all, input_decoded_all)

        fields_2_save_2_logfile = [
            "state_dist_L1_hist_error",
            "state_dist_wasserstein_distance",
        ]

        fields_2_save_2_logfile += list(error_dict_avg.keys())

        results_autoencoder = {
            "dt": dt,
            "latent_states_all": latent_states_all,
            "input_decoded_all": input_decoded_all,
            "input_sequence_all": input_sequence_all,
            "fields_2_save_2_logfile": fields_2_save_2_logfile,
        }
        results_autoencoder = {
            **results_autoencoder,
            **additional_results_dict,
            **error_dict,
            **error_dict_avg,
            **state_statistics
        }
        results_autoencoder = systems.addResultsSystem(
            self,
            results_autoencoder,
            state_statistics,
            testing_mode="autoencoder_testing")
        return results_autoencoder

    def predictIndexes(self, data_loader, dt, set_name, testing_mode):
        print("# predictIndexes() #")
        assert (testing_mode in self.getTestingModes())
        if testing_mode not in self.getRNNTestingModes():
            raise ValueError("Mode {:} is invalid".format(testing_mode))

        num_test_ICS = self.num_test_ICS

        predictions_all = []
        targets_all = []
        latent_states_all = []

        predictions_augmented_all = []
        targets_augmented_all = []
        latent_states_augmented_all = []

        rmse_all = []
        mnad_all = []
        correlation_all = []
        num_accurate_pred_005_all = []
        num_accurate_pred_050_all = []
        time_total_per_iter_all = []

        # Dictionary of error lists
        error_dict = utils.getErrorLabelsDict(self)

        # if num_test_ICS > len(data_loader): raise ValueError("Not enough ICs in the dataset {:}. The dataset has {:} / {:}.".format(set_name, len(data_loader), num_test_ICS))
        if num_test_ICS > len(data_loader):
            print(
                "# NOTE: Not enough ICs in the dataset {:}. The dataset has {:} / {:}."
                .format(set_name, len(data_loader), num_test_ICS))
            num_test_ICS = len(data_loader)

        ic_num = 1
        ic_indexes = []
        for sequence in data_loader:
            if ic_num > num_test_ICS: break
            if self.display_output:
                print("IC {:}/{:}, {:2.3f}%".format(
                    ic_num, num_test_ICS, ic_num / num_test_ICS * 100))
            sequence = sequence[0]

            # STARTING TO PREDICT THE SEQUENCE IN self.predict_on=self.sequence_length
            # Warming-up with sequence_length
            self.predict_on = self.sequence_length
            if self.predict_on + self.prediction_horizon > np.shape(
                    sequence)[0]:
                raise ValueError(
                    "self.predict_on {:} + self.prediction_horizon {:} > np.shape(sequence)[0] {:}. Reduce the prediction horizon, or the sequence length."
                    .format(self.predict_on, self.prediction_horizon,
                            np.shape(sequence)[0]))
            assert (self.predict_on + self.prediction_horizon <=
                    np.shape(sequence)[0])
            assert (self.predict_on - self.n_warmup >= 0)

            sequence = sequence[self.predict_on -
                                self.n_warmup:self.predict_on +
                                self.prediction_horizon]

            prediction, target, prediction_augment, target_augment, latent_states, latent_states_augmented, time_total_per_iter = self.predictSequence(
                sequence, testing_mode, dt, ic_num, set_name)

            prediction = self.data_info_dict["scaler"].descaleData(
                prediction, single_sequence=True)
            target = self.data_info_dict["scaler"].descaleData(
                target, single_sequence=True)

            prediction_augment = self.data_info_dict["scaler"].descaleData(
                prediction_augment, single_sequence=True)
            target_augment = self.data_info_dict["scaler"].descaleData(
                target_augment, single_sequence=True)

            errors = utils.computeErrors(
                target,
                prediction,
                compute_r2=self.data_info_dict["compute_r2"])
            # Updating the error
            for error in errors:
                error_dict[error].append(errors[error])

            latent_states_all.append(latent_states)
            predictions_all.append(prediction)
            targets_all.append(target)

            latent_states_augmented_all.append(latent_states_augmented)
            predictions_augmented_all.append(prediction_augment)
            targets_augmented_all.append(target_augment)

            time_total_per_iter_all.append(time_total_per_iter)
            ic_indexes.append(ic_num)
            ic_num += 1

        time_total_per_iter_all = np.array(time_total_per_iter_all)
        time_total_per_iter = np.mean(time_total_per_iter_all)

        predictions_all = np.array(predictions_all)
        targets_all = np.array(targets_all)
        latent_states_all = np.array(latent_states_all)

        predictions_augmented_all = np.array(predictions_augmented_all)
        targets_augmented_all = np.array(targets_augmented_all)
        latent_states_augmented_all = np.array(latent_states_augmented_all)

        print("TRAJECTORIES SHAPES:")
        print(np.shape(targets_all))
        print(np.shape(predictions_all))

        # Computing the average over time
        error_dict_avg = {}
        for key in error_dict:
            error_dict_avg[key + "_avg"] = np.mean(error_dict[key])
        utils.printErrors(error_dict_avg)

        # Computing additional errors based on all predictions (e.g. frequency spectra)
        additional_results_dict, additional_errors_dict = utils.computeAdditionalResults(
            self, targets_all, predictions_all, dt)
        error_dict_avg = {**error_dict_avg, **additional_errors_dict}

        state_statistics = utils.computeStateDistributionStatistics(
            targets_all,
            predictions_all,
            statistics_per_state=self.data_info_dict["statistics_per_state"],
            statistics_cummulative=self.
            data_info_dict["statistics_cummulative"],
            statistics_per_timestep=self.
            data_info_dict["statistics_per_timestep"],
        )
        state_statistics = systems.computeStateDistributionStatisticsSystem(
            self, state_statistics, targets_all, predictions_all)

        fields_2_save_2_logfile = [
            "state_dist_wasserstein_distance",
            "state_dist_L1_hist_error",
            "time_total_per_iter",
        ]
        fields_2_save_2_logfile += list(error_dict_avg.keys())

        results = {
            "fields_2_save_2_logfile": fields_2_save_2_logfile,
            "predictions_all": predictions_all,
            "targets_all": targets_all,
            "latent_states_all": latent_states_all,
            "predictions_augmented_all": predictions_augmented_all,
            "targets_augmented_all": targets_augmented_all,
            "latent_states_augmented_all": latent_states_augmented_all,
            "n_warmup": self.n_warmup,
            "testing_mode": testing_mode,
            "dt": dt,
            "time_total_per_iter": time_total_per_iter,
            "ic_indexes": ic_indexes,
        }
        results = {
            **results,
            **additional_results_dict,
            **error_dict,
            **error_dict_avg,
            **state_statistics
        }
        # According to the system under study, adding additional results
        results = systems.addResultsSystem(self, results, state_statistics,
                                           testing_mode)
        return results

    def predictSequence(self,
                        input_sequence,
                        testing_mode=None,
                        dt=1,
                        ic_idx=None,
                        set_name=None):
        print("# predictSequence() #")
        print(np.shape(input_sequence))

        N = np.shape(input_sequence)[0]
        # PREDICTION LENGTH
        if N - self.n_warmup != self.prediction_horizon:
            raise ValueError(
                "Error! N ({:}) - self.n_warmup ({:}) != prediction_horizon ({:})"
                .format(N, self.n_warmup, self.prediction_horizon))
        # PREPARING THE HIDDEN STATES
        initial_hidden_states = self.getZeroRnnHiddenState(1)
        assert self.n_warmup > 1, "Warm up steps cannot be <= 1. Increase the iterative prediction length."

        warmup_data_input = input_sequence[:self.n_warmup - 1]
        warmup_data_input = warmup_data_input[np.newaxis, :]

        warmup_data_target = input_sequence[1:self.n_warmup]
        warmup_data_target = warmup_data_target[np.newaxis, :]

        if testing_mode in self.getRNNTestingModes():
            target = input_sequence[self.n_warmup:self.n_warmup +
                                    self.prediction_horizon]
        else:
            raise ValueError(
                "Testing mode {:} not recognized.".format(testing_mode))

        warmup_data_input = self.torch_dtype(warmup_data_input)

        if self.gpu:
            # SENDING THE TENSORS TO CUDA
            warmup_data_input = warmup_data_input.cuda()
            initial_hidden_states = self.sendHiddenStateToGPU(
                initial_hidden_states)

        # print(initial_hidden_states)
        # print(warmup_data_input[-1])

        warmup_data_output, last_hidden_state, warmup_latent_states, latent_states_pred, _, _, _ = self.model.forward(
            warmup_data_input,
            initial_hidden_states,
            is_train=False,
            sample_mixture=True)

        # print(latent_states_pred)

        prediction = []

        if ("iterative_latent" in testing_mode) or ("multiscale"
                                                    in testing_mode):
            iterative_propagation_is_latent = 1
            # GETTING THE LAST LATENT STATE (K, T, LD)
            # In iterative latent forecasting, the input is the latent state
            input_latent = latent_states_pred[:, -1, :]
            input_latent.unsqueeze_(0)
            input_t = input_latent
        elif ("iterative_state" in testing_mode):
            iterative_propagation_is_latent = 0
            # LATTENT PROPAGATION
            input_t = input_sequence[self.n_warmup - 1]
            input_t = input_t[np.newaxis, np.newaxis, :]
        elif "teacher_forcing" in testing_mode:
            iterative_propagation_is_latent = 0
            temp = input_sequence[self.n_warmup - 1:-1]
            temp = temp.cpu().detach().numpy()
            if self.channels == 0:
                input_t = np.reshape(temp, (1, -1, self.input_dim))
            elif self.channels == 1:
                input_t = np.reshape(temp, (1, -1, self.input_dim, self.Dx))
            elif self.channels == 2:
                input_t = np.reshape(temp,
                                     (1, -1, self.input_dim, self.Dx, self.Dy))
            else:
                raise ValueError(
                    "Invalid number of channels: not implemented.")
        else:
            raise ValueError(
                "I do not know how to initialize the state for {:}.".format(
                    testing_mode))

        input_t = self.torch_dtype(input_t)

        if self.gpu:
            input_t = input_t.cuda()
            last_hidden_state = self.sendHiddenStateToGPU(last_hidden_state)

        # In case of multiscale modeling, prepare the micro_solver
        if "multiscale" in testing_mode:
            micro_solver = systems.prepareMicroSolver(self, ic_idx, dt,
                                                      set_name)

        time_start = time.time()
        if "teacher_forcing" in testing_mode:
            input_t = self.torch_dtype(input_t)
            prediction, last_hidden_state, latent_states, latent_states_pred, RNN_outputs, input_decoded, time_latent_prop = self.model.forward(
                input_t,
                last_hidden_state,
                is_iterative_forecasting=False,
                horizon=self.prediction_horizon,
                is_train=False,
                iterative_propagation_is_latent=iterative_propagation_is_latent,
                input_is_latent=False,
                sample_mixture=True)
        elif "iterative_latent" in testing_mode:
            # LATENT/ORIGINAL DYNAMICS PROPAGATION
            input_t = self.torch_dtype(input_t)
            prediction, last_hidden_state, latent_states, latent_states_pred, RNN_outputs, input_decoded, time_latent_prop = self.model.forward(
                input_t,
                last_hidden_state,
                is_iterative_forecasting=True,
                iterative_forecasting_prob=1.0,
                horizon=self.prediction_horizon,
                is_train=False,
                iterative_propagation_is_latent=iterative_propagation_is_latent,
                input_is_latent=iterative_propagation_is_latent,
                sample_mixture=True,
            )
        elif "iterative_state" in testing_mode:
            # LATENT/ORIGINAL DYNAMICS PROPAGATION
            prediction, last_hidden_state, latent_states, latent_states_pred, RNN_outputs, input_decoded, time_latent_prop = self.model.forward(
                input_t,
                last_hidden_state,
                is_iterative_forecasting=True,
                iterative_forecasting_prob=1.0,
                horizon=self.prediction_horizon,
                is_train=False,
                iterative_propagation_is_latent=iterative_propagation_is_latent,
                input_is_latent=iterative_propagation_is_latent,
                sample_mixture=True)

        elif "multiscale" in testing_mode:
            # LATENT DYNAMICS PROPAGATION
            multiscale_rounds, macro_steps_per_round, micro_steps_per_round = self.getMultiscaleParams(
                testing_mode)
            # print("# Multiscale: Macroscale timesteps: {:}, Microscale timesteps: {:}".format(multiscale_macro_steps, multiscale_micro_steps))
            # iterative_predictions_per_round = multiscale_macro_steps + multiscale_micro_steps
            self.multiscale_rounds = multiscale_rounds

            # print("PER ROUND: Model prediction length: {:}, Dynamics prediction length: {:}".format(multiscale_macro_steps, multiscale_micro_steps))
            # self.multiscale_rounds = int(self.prediction_horizon / (multiscale_macro_steps+multiscale_micro_steps))

            # if self.prediction_horizon % self.multiscale_rounds != 0:
            #     raise ValueError("prediction_horizon ({:}) % multiscale_rounds ({:}) != 0.".format(self.prediction_horizon,self.multiscale_rounds))

            # print("# multiscale rounds: {:}, total timesteps: {:}".format(self.multiscale_rounds, self.prediction_horizon))

            time_dynamics = 0.0
            time_latent_prop = 0.0
            for round_ in range(self.multiscale_rounds):
                multiscale_macro_steps = macro_steps_per_round[round_]
                if multiscale_macro_steps > 0:
                    prediction_model_dyn, last_hidden_state, latent_states_, latent_states_pred, RNN_outputs_, input_decoded, time_latent_prop_t = self.model.forward(
                        input_t,
                        last_hidden_state,
                        is_iterative_forecasting=True,
                        iterative_forecasting_prob=1.0,
                        horizon=multiscale_macro_steps,
                        is_train=False,
                        iterative_propagation_is_latent=
                        iterative_propagation_is_latent,
                        input_is_latent=iterative_propagation_is_latent,
                        sample_mixture=True)
                    time_latent_prop += time_latent_prop_t
                    if round_ == 0:
                        prediction = prediction_model_dyn
                        RNN_outputs = RNN_outputs_
                        latent_states = latent_states_
                    else:
                        prediction = torch.cat(
                            (prediction, prediction_model_dyn), 1)
                        RNN_outputs = torch.cat((RNN_outputs, RNN_outputs_), 1)
                        latent_states = torch.cat(
                            (latent_states, latent_states_), 1)

                    init_state = prediction_model_dyn.cpu().detach().numpy(
                    )[0][-1]
                elif round_ == 0:
                    init_state = input_sequence[self.n_warmup - 1]
                    init_state = init_state.cpu().detach().numpy()
                else:
                    # Correcting the shape
                    init_state = prediction_sys_dyn[:, -1:, :]
                    init_state = init_state[0, 0]

                if round_ < len(micro_steps_per_round):
                    multiscale_micro_steps = micro_steps_per_round[round_]

                    init_state = np.reshape(init_state,
                                            (1, *np.shape(init_state)))
                    # print(np.shape(init_state))
                    # print(ark)
                    # Time covered in multiscale dynamics - needed by multiscale applications with time-dependent forcing
                    # t_jump = dt * (round_ *
                    #     (multiscale_macro_steps + multiscale_micro_steps) +
                    #     multiscale_macro_steps)

                    t_jump = dt * multiscale_macro_steps

                    # How much time to cover with the full scale dynamics
                    total_time = multiscale_micro_steps * dt
                    # Upsampling to the original space and using the PDE to evolve in time
                    if total_time > 0.0:

                        init_state = self.data_info_dict["scaler"].descaleData(
                            init_state, single_sequence=True)

                        time_dynamics_start = time.time()

                        prediction_sys_dyn = systems.evolveSystem(
                            self, micro_solver, init_state, total_time, dt,
                            t_jump)

                        prediction_sys_dyn = self.data_info_dict[
                            "scaler"].scaleData(
                                prediction_sys_dyn,
                                single_sequence=True,
                            )

                        init_state = self.data_info_dict["scaler"].scaleData(
                            init_state, single_sequence=True)
                        time_dynamics_end = time.time()
                        time_dynamics_round = time_dynamics_end - time_dynamics_start
                        time_dynamics += time_dynamics_round

                        prediction_sys_dyn = prediction_sys_dyn[np.newaxis]
                        prediction_sys_dyn_tensor = self.torch_dtype(
                            prediction_sys_dyn)

                        prediction = prediction_sys_dyn_tensor if (
                            multiscale_macro_steps == 0
                            and round_ == 0) else torch.cat(
                                (prediction, prediction_sys_dyn_tensor), 1)

                        # print(np.shape(init_state))
                        # print(ark)
                        init_state = np.reshape(init_state,
                                                (1, *np.shape(init_state)))

                        # In order to update the last hidden state, we need to feed the dynamics up to input_t
                        # to the network
                        idle_dynamics = prediction_sys_dyn[:, :-1, :]
                        idle_dynamics = np.concatenate(
                            (init_state, idle_dynamics), axis=1)
                        idle_dynamics = self.torch_dtype(idle_dynamics)

                        _, last_hidden_state, latent_states_, latent_states_pred, RNN_outputs_, _, time_latent_prop_t = self.model.forward(
                            idle_dynamics,
                            last_hidden_state,
                            is_train=False,
                            sample_mixture=True)
                        time_latent_prop += time_latent_prop_t

                        RNN_outputs = RNN_outputs_ if (
                            multiscale_macro_steps == 0
                            and round_ == 0) else torch.cat(
                                (RNN_outputs, RNN_outputs_), 1)

                        latent_states = latent_states_ if (
                            multiscale_macro_steps == 0
                            and round_ == 0) else torch.cat(
                                (latent_states, latent_states_), 1)

                    else:
                        prediction_sys_dyn = prediction_model_dyn

                    if iterative_propagation_is_latent:
                        input_t = latent_states_pred[:, -1:, :]
                    else:
                        # Next to feed, the last predicted state (from the dynamics)
                        input_t = prediction_sys_dyn[:, -1:, :]
                        input_t = self.torch_dtype(input_t)
        else:
            raise ValueError(
                "Testing mode {:} not recognized.".format(testing_mode))
        time_end = time.time()
        time_total = time_end - time_start

        # Correcting the time-measurement in case of evolution of the original system (in this case, we do not need to internally propagate the latent space of the RNN)
        if "multiscale" in testing_mode:
            if multiscale_macro_steps == 0:
                print("Tracking the time when using the original dynamics...")
                time_total = time_dynamics
            else:
                time_total = time_latent_prop + time_dynamics
        else:
            time_total = time_latent_prop

        time_total_per_iter = time_total / self.prediction_horizon

        prediction = prediction[0]
        RNN_outputs = RNN_outputs[0]
        latent_states = latent_states[0]

        prediction = prediction.cpu().detach().numpy()
        latent_states = latent_states.cpu().detach().numpy()
        RNN_outputs = RNN_outputs.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        prediction = np.array(prediction)
        latent_states = np.array(latent_states)
        RNN_outputs = np.array(RNN_outputs)
        target = np.array(target)

        print("Shapes of prediction/target/latent_states")
        print(np.shape(prediction))
        print(np.shape(target))
        print(np.shape(latent_states))

        # print("Min/Max")
        # print("Target:")
        # print(np.max(target[:,0]))
        # print(np.min(target[:,0]))
        # print("Prediction:")
        # print(np.max(prediction[:,0]))
        # print(np.min(prediction[:,0]))
        # # print(ark)
        #
        warmup_data_target = warmup_data_target.cpu().detach().numpy()
        warmup_data_output = warmup_data_output.cpu().detach().numpy()
        warmup_latent_states = warmup_latent_states.cpu().detach().numpy()

        target_augment = np.concatenate((warmup_data_target[0], target),
                                        axis=0)
        prediction_augment = np.concatenate(
            (warmup_data_output[0], prediction), axis=0)
        latent_states_augmented = np.concatenate(
            (warmup_latent_states[0], latent_states), axis=0)

        return prediction, target, prediction_augment, target_augment, latent_states, latent_states_augmented, time_total_per_iter

    def plot(self):
        print("# plot() #")

        sets_to_test = []
        if self.params["test_on_train"]: sets_to_test.append("train")
        if self.params["test_on_val"]: sets_to_test.append("val")
        if self.params["test_on_test"]: sets_to_test.append("test")

        for set_name in sets_to_test:

            # Autoencoder testing
            if self.has_autoencoder and not (self.has_rnn):

                testing_mode = "autoencoder_testing"

                data_path = self.getResultsDir() + "/results_{:}_{:}".format(
                    testing_mode, set_name)
                results = utils.loadData(data_path, self.save_format)

                # Plot the distribution on the state
                if self.params["plot_state_distributions"]:
                    utils.plotStateDistributions(self, results, set_name,
                                                 testing_mode)

                # Plot distributions specific to a system
                if self.params["plot_state_distributions_system"]:
                    systems.plotStateDistributionsSystem(
                        self, results, set_name, testing_mode)

                # Plot transition times specific to a system
                if self.params["plot_system"]:
                    systems.plotSystem(self, results, set_name, testing_mode)

                # Plot examples of testing initial conditions
                if self.params["plot_testing_ics_examples"]:
                    # Postprocessing to add statistics on single time instants (for plotting of stochastic systems)
                    results = utils.postprocessComputeStateDensitiesSingleTimeInstant(
                        self, results)

                # Writing to a log-file
                if self.write_to_log:
                    logfile = self.getLogFileDir(
                    ) + "/results_{:}_{:}.txt".format(testing_mode, set_name)
                    utils.writeToLogFile(self, logfile, results,
                                         results["fields_2_save_2_logfile"])

                # Plotting examples of testing initial conditions
                if self.params["plot_testing_ics_examples"]:
                    # Maximum plotting of two examples
                    ic_plot = np.min([1, len(results["input_sequence_all"])])
                    for ic in range(ic_plot):
                        print("IC {:}".format(ic))
                        utils.plotTestingContours(
                            self,
                            results["input_sequence_all"][ic],
                            results["input_decoded_all"][ic],
                            results["dt"],
                            ic,
                            set_name,
                            latent_states=results["latent_states_all"][ic],
                            testing_mode=testing_mode,
                            hist_data=results["hist_data"],
                            wasserstein_distance_data=results[
                                "wasserstein_distance_data"],
                        )

                        # Plotting the latent dynamics for these examples
                        if self.params["plot_latent_dynamics_single_traj"]:
                            utils.plotLatentDynamicsOfSingleTrajectory(
                                self,
                                set_name,
                                results["latent_states_all"][ic],
                                ic,
                                testing_mode,
                                dt=results["dt"])

            # Postprocessing of RNN testing results
            elif self.has_rnn:
                # fields_to_compare = [
                # "time_total_per_iter",
                # # "rmnse_avg_over_ics",
                # "rmnse_avg",
                # # "num_accurate_pred_050_avg",
                # # "error_freq",
                # ]
                # fields_to_compare = systems.addFieldsToCompare(self, fields_to_compare)

                if self.multiscale_forecasting:
                    fields_to_compare = utils.getFieldsToCompare(self)
                    fields_to_compare = systems.addFieldsToCompare(
                        self, fields_to_compare)
                else:
                    fields_to_compare = []

                dicts_to_compare = {}
                latent_states_dict = {}

                for testing_mode in self.getRNNTestingModes():

                    # Loading the results
                    data_path = self.getResultsDir(
                    ) + "/results_{:}_{:}".format(testing_mode, set_name)
                    results = utils.loadData(data_path, self.save_format)

                    # Plotting the state distributions specific to a system
                    if self.params["plot_state_distributions_system"]:
                        systems.plotStateDistributionsSystem(
                            self, results, set_name, testing_mode)

                    # Plotting the state distributions
                    if self.params["plot_state_distributions"]:
                        utils.plotStateDistributions(self, results, set_name,
                                                     testing_mode)

                    # Computing the spectrum
                    if self.params["compute_spectrum"]:
                        utils.plotSpectrum(self, results, set_name,
                                           testing_mode)

                    # Plot the transition times in a system
                    if self.params["plot_system"]:
                        systems.plotSystem(self, results, set_name,
                                           testing_mode)

                    if self.write_to_log:
                        logfile = self.getLogFileDir(
                        ) + "/results_{:}_{:}.txt".format(
                            testing_mode, set_name)
                        utils.writeToLogFile(
                            self, logfile, results,
                            results["fields_2_save_2_logfile"])

                    ic_indexes = results["ic_indexes"]
                    dt = results["dt"]
                    n_warmup = results["n_warmup"]

                    predictions_augmented_all = results[
                        "predictions_augmented_all"]
                    targets_augmented_all = results["targets_augmented_all"]

                    predictions_all = results["predictions_all"]
                    targets_all = results["targets_all"]
                    latent_states_all = results["latent_states_all"]

                    latent_states_dict[testing_mode] = latent_states_all

                    results_dict = {}
                    for field in fields_to_compare:
                        results_dict[field] = results[field]
                    dicts_to_compare[testing_mode] = results_dict

                    if self.params["plot_testing_ics_examples"]:
                        # max_index = np.min([1, np.shape(results["targets_all"])[0]])
                        max_index = np.min(
                            [3, np.shape(results["targets_all"])[0]])
                        # max_index = np.min([10, np.shape(results["targets_all"])[0]])
                        for idx in range(max_index):

                            results_idx = {
                                "target": targets_all[idx],
                                "prediction": predictions_all[idx],
                                "latent_states": latent_states_all[idx],
                                "fields_2_save_2_logfile": [],
                            }

                            results_idx = utils.postprocessComputeStateDensitiesSingleTimeInstant(
                                self, results_idx, autoencoder=False)
                            utils.createIterativePredictionPlots(self, \
                                targets_all[idx], \
                                predictions_all[idx], \
                                dt, ic_indexes[idx], set_name, \
                                testing_mode=testing_mode, \
                                latent_states=latent_states_all[idx], \
                                hist_data=results_idx["hist_data"], \
                                wasserstein_distance_data=results_idx["wasserstein_distance_data"], \
                                warm_up=n_warmup, \
                                target_augment=targets_augmented_all[idx], \
                                prediction_augment=predictions_augmented_all[idx], \
                                )

                        # Plotting the latent dynamics for these examples
                        if self.params["plot_latent_dynamics_single_traj"]:
                            utils.plotLatentDynamicsOfSingleTrajectory(
                                self,
                                set_name,
                                results["latent_states_all"][idx],
                                idx,
                                testing_mode,
                                dt=results["dt"])

                if self.params["plot_latent_dynamics_comparison_system"]:
                    systems.plotLatentDynamicsComparisonSystem(self, set_name)
                if self.params["plot_multiscale_results_comparison"]:
                    utils.plotMultiscaleResultsComparison(
                        self,
                        dicts_to_compare,
                        set_name,
                        fields_to_compare,
                        results["dt"],
                    )
                # utils.plotLatentDynamicsComparison(self, latent_states_dict, set_name)

    def debug(self):
        print("# debug() #")

        sets_to_test = []
        if self.params["test_on_train"]: sets_to_test.append("train")
        if self.params["test_on_val"]: sets_to_test.append("val")
        if self.params["test_on_test"]: sets_to_test.append("test")

        for set_name in sets_to_test:

            # # Autoencoder testing
            # # if self.has_autoencoder and not (self.has_rnn):
            # if self.has_autoencoder:
            #     testing_mode = "autoencoder_testing"
            #     data_path = self.getResultsDir() + "/results_{:}_{:}".format(testing_mode, set_name)
            #     results_autoencoder = utils.loadData(data_path, self.save_format)
            #     state_statistics=None
            #     results_autoencoder = systems.addResultsSystem(self, results_autoencoder, state_statistics, testing_mode="autoencoder_testing")
            #     # utils.saveData(results_autoencoder, data_path, self.save_format)
            #     systems.plotSystem(self, results_autoencoder, set_name, testing_mode)

            if self.has_rnn:

                # for testing_mode in self.getRNNTestingModes():
                for testing_mode in ["iterative_latent_forecasting"]:

                    # Loading the results
                    data_path = self.getResultsDir(
                    ) + "/results_{:}_{:}".format(testing_mode, set_name)
                    results = utils.loadData(data_path, self.save_format)

                    # results["predictions_all"] = results["predictions_all"][:10]
                    # results["targets_all"] = results["targets_all"][:10]
                    # results["latent_states_all"] = results["latent_states_all"][:10]

                    # results = systems.addResultsSystem(self, results, None, testing_mode, set_name)
                    # utils.saveData(results, data_path, self.save_format)
                    
                    # Plot distributions specific to a system
                    systems.plotStateDistributionsSystem(self, results, set_name, testing_mode)
                    # systems.plotSystem(self, results, set_name, testing_mode)

                # systems.plotLatentDynamicsComparisonSystem(self, set_name)

                # systems.plotLatentDynamicsComparisonSystem(self, set_name)





