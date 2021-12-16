#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python

# TORCH
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.distributions as tdist

from . import zoneoutlayer
from . import permutation_invariance
from . import mixture_density
from . import activations

# PRINTING
from functools import partial
print = partial(print, flush=True)

import time


class md_arnn_model(nn.Module):
    def __init__(self, params, model):
        super(md_arnn_model, self).__init__()
        self.parent = model

        # Determining cell type
        if self.parent.params["RNN_cell_type"] == "lstm":
            self.RNN_cell = nn.LSTMCell
        elif self.parent.params["RNN_cell_type"] == "gru":
            self.RNN_cell = nn.GRUCell
        elif self.parent.params["RNN_cell_type"] == "plain":
            self.RNN_cell = nn.RNNCell
        else:
            raise ValueError("Invalid RNN_cell_type {:}".format(
                params["RNN_cell_type"]))

        # Determining autoencoder non-linearity
        self.activation_str_general = params["activation_str_general"]
        if self.activation_str_general not in [
                "relu", "celu", "elu", "selu", "tanh"
        ]:  # WITH AUTOENCODER
            raise ValueError("Invalid general activation {:}".format(
                self.activation_str_general))
        self.activation_general = activations.getActivation(
            params["activation_str_general"])
        self.softmax = nn.Softmax(dim=-1)
        self.buildNetwork()

    def ifAnyIn(self, list_, name):
        for element in list_:
            if element in name:
                return True
        return False

    def initializeWeights(self):
        print("Initializing parameters...\n")
        # for modules in [self.ENCODER, self.DECODER, self.RNN, self.RNN_OUTPUT]:
        for modules in self.module_list:
            for module in modules:
                for name, param in module.named_parameters():
                    # print(name)
                    # INITIALIZING RNN, GRU CELLS
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)

                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)

                    elif self.ifAnyIn([
                            "Wxi.weight", "Wxf.weight", "Wxc.weight",
                            "Wxo.weight"
                    ], name):
                        torch.nn.init.xavier_uniform_(param.data)

                    elif self.ifAnyIn([
                            "Wco", "Wcf", "Wci", "Whi.weight", "Whf.weight",
                            "Whc.weight", "Who.weight"
                    ], name):
                        torch.nn.init.orthogonal_(param.data)

                    elif self.ifAnyIn([
                            "Whi.bias", "Wxi.bias", "Wxf.bias", "Whf.bias",
                            "Wxc.bias", "Whc.weight", "Wxo.bias", "Who.bias"
                    ], name):
                        param.data.fill_(0)

                    elif 'weight' in name:
                        torch.nn.init.xavier_uniform_(param.data)

                    elif 'bias' in name:
                        param.data.fill_(0)
                    else:
                        raise ValueError("NAME {:} NOT FOUND!".format(name))
                        # print("NAME {:} NOT FOUND!".format(name))
        print("Parameteres initialized!")
        return 0

    def sendModelToCPU(self):
        print("SENDING MODEL TO CPU")
        for modules in self.module_list:
            for model in modules:
                model.cpu()
        return 0

    def sendModelToCuda(self):
        print("SENDING MODEL TO CUDA")
        for modules in self.module_list:
            for model in modules:
                model.cuda()
        return 0

    def buildNetwork(self):
        self.DROPOUT = nn.ModuleList()
        self.DROPOUT.append(nn.Dropout(p=1 - self.parent.dropout_keep_prob))

        # Translation invariant layer
        self.PERM_INV = nn.ModuleList()
        for ln in range(len(self.parent.layers_perm_inv_aug) - 1):
            self.PERM_INV.append(
                nn.Linear(self.parent.layers_perm_inv_aug[ln],
                          self.parent.layers_perm_inv_aug[ln + 1],
                          bias=True))

        if self.parent.has_autoencoder:
            # Building the layers of the encoder
            if not self.parent.AE_convolutional:
                self.ENCODER = nn.ModuleList()
                for ln in range(len(self.parent.layers_encoder_aug) - 1):
                    self.ENCODER.append(
                        nn.Linear(self.parent.layers_encoder_aug[ln],
                                  self.parent.layers_encoder_aug[ln + 1],
                                  bias=True))
            else:
                raise ValueError("Not implemented.")
                encoder = conv_models.getEncoderModel(self.parent)
                self.ENCODER = encoder.layers
        else:
            self.ENCODER = nn.ModuleList()

        self.RNN = nn.ModuleList()
        if self.parent.has_rnn:
            # Parsing the layers of the RNN
            # The input to the RNN is an embedding (effective dynamics) vector, or latent vector
            input_size = self.parent.params["RNN_state_dim"]
            for ln in range(len(self.parent.layers_rnn)):
                self.RNN.append(
                    zoneoutlayer.ZoneoutLayer(
                        self.RNN_cell(input_size=input_size,
                                      hidden_size=self.parent.layers_rnn[ln]),
                        self.parent.zoneout_keep_prob))
                input_size = self.parent.layers_rnn[ln]

        # Output MLP of the RNN
        self.RNN_OUTPUT = nn.ModuleList()
        if (self.parent.has_rnn):
            if (self.parent.RNN_MDN_bool):
                print("Adding RNN Mixture-Density-Network Layer")
                # Deciding on the output dimension of the MDN, depending on multivariate/independent modeling
                self.RNN_MDN_output_dim = self.parent.params["RNN_state_dim"]
                self.RNN_MDN_input_dim = self.parent.layers_rnn[-1]
                # Deciding on the scaler used in the RNN-MDN
                if self.parent.scaler == "MinMaxZeroOne" and not self.parent.has_autoencoder:
                    # No autoencoder, so targets of the RNN are
                    self.RNN_MDN_bounded = True
                else:
                    self.RNN_MDN_bounded = False

                self.RNN_OUTPUT.extend([
                    mixture_density.MixtureDensityNetwork(
                        input_dim=self.RNN_MDN_input_dim,
                        output_dim=self.RNN_MDN_output_dim,
                        n_kernels=self.parent.RNN_MDN_kernels,
                        n_hidden=self.parent.RNN_MDN_hidden_units,
                        weight_sharing=0,
                        multivariate=self.parent.RNN_MDN_multivariate,
                        bounded=self.RNN_MDN_bounded,
                        activation=self.parent.
                        params["activation_str_general"],
                        sigma_max=self.parent.RNN_MDN_sigma_max,
                        fixed_kernels=self.parent.RNN_MDN_fixed_kernels,
                        train_kernels=self.parent.RNN_MDN_train_kernels,
                        multivariate_covariance_layer=self.parent.
                        RNN_MDN_multivariate_covariance_layer,
                        MDN_multivariate_pretrain_diagonal=self.parent.
                        RNN_MDN_multivariate_pretrain_diagonal,
                        dist=self.parent.RNN_MDN_distribution)
                ])
            else:
                self.RNN_OUTPUT.extend([
                    nn.Linear(self.parent.layers_rnn[-1],
                              self.parent.params["RNN_state_dim"],
                              bias=True)
                ])

        # Building the layers of the decoder (additional input is the latent noise)
        self.DECODER = nn.ModuleList()
        for ln in range(len(self.parent.layers_decoder_aug) - 1):
            if self.parent.MDN_bool and ln == len(
                    self.parent.layers_decoder_aug) - 2:  # In the last layer
                print("Adding DECODING Mixture-Density-Network Layer")
                # Deciding on the output dimension of the MDN, depending on multivariate/independent modeling
                if self.parent.AE_perm_invariant_latent_dim:
                    if not self.parent.MDN_multivariate:
                        self.MDN_output_dim = self.parent.Dx * self.parent.layers_decoder_aug[
                            ln + 1]
                    elif self.parent.MDN_multivariate:
                        self.MDN_output_dim = self.parent.Dx
                else:
                    self.MDN_output_dim = self.parent.input_dim
                self.MDN_input_dim = self.parent.layers_decoder_aug[ln]

                # Deciding on the scaler used in the MDN
                if self.parent.scaler == "MinMaxZeroOne":
                    self.MDN_bounded = True
                elif self.parent.scaler in ["Standard", "standard"]:
                    self.MDN_bounded = False
                elif self.parent.scaler in ["no"]:
                    self.MDN_bounded = False
                elif self.parent.scaler == "MinMaxMinusOneOne":
                    raise ValueError(
                        "Not implemented! MDN with bounded output in (-1,1) is not implemented."
                    )

                self.DECODER.append(
                    mixture_density.MixtureDensityNetwork(
                        input_dim=self.MDN_input_dim,
                        output_dim=self.MDN_output_dim,
                        n_kernels=self.parent.MDN_kernels,
                        n_hidden=self.parent.MDN_hidden_units,
                        weight_sharing=self.parent.MDN_weight_sharing,
                        multivariate=self.parent.MDN_multivariate,
                        bounded=self.MDN_bounded,
                        activation=self.parent.
                        params["activation_str_general"],
                        sigma_max=self.parent.MDN_sigma_max,
                        fixed_kernels=self.parent.MDN_fixed_kernels,
                        train_kernels=self.parent.MDN_train_kernels,
                        multivariate_covariance_layer=self.parent.
                        MDN_multivariate_covariance_layer,
                        MDN_multivariate_pretrain_diagonal=self.parent.
                        MDN_multivariate_pretrain_diagonal,
                        dist=self.parent.MDN_distribution))
            else:
                self.DECODER.append(
                    nn.Linear(self.parent.layers_decoder_aug[ln],
                              self.parent.layers_decoder_aug[ln + 1],
                              bias=True))

        self.module_list = [
            self.DROPOUT, self.PERM_INV, self.ENCODER, self.DECODER, self.RNN,
            self.RNN_OUTPUT
        ]
        return 0

    def countTrainableParams(self):
        temp = 0
        for layers in self.module_list:
            for layer in layers:
                temp += sum(p.numel() for p in layer.parameters()
                            if p.requires_grad)
        return temp

    def countParams(self):
        temp = 0
        for layers in self.module_list:
            for layer in layers:
                temp += sum(p.numel() for p in layer.parameters())
        return temp

    def getParams(self):
        params = list()
        for layers in self.module_list:
            for layer in layers:
                params += layer.parameters()
        return params

    def getAutoencoderParams(self):
        params = list()
        for layers in [self.PERM_INV, self.ENCODER, self.DECODER]:
            for layer in layers:
                params += layer.parameters()
        return params

    def getRNNParams(self):
        params = list()
        for layers in [self.RNN, self.RNN_OUTPUT]:
            for layer in layers:
                params += layer.parameters()
        return params

    def printModuleList(self):
        print(self.module_list)
        return 0

    def eval(self):
        for modules in [
                self.DROPOUT, self.PERM_INV, self.ENCODER, self.DECODER,
                self.RNN, self.RNN_OUTPUT
        ]:
            for layer in modules:
                layer.eval()
        return 0

    def train(self):
        for modules in [
                self.DROPOUT, self.PERM_INV, self.ENCODER, self.DECODER,
                self.RNN, self.RNN_OUTPUT
        ]:
            for layer in modules:
                layer.train()
        return 0

    def forecast(self,
                 inputs,
                 init_hidden_state,
                 horizon=None,
                 is_train=False,
                 iterative_forecasting_prob=0,
                 iterative_forecasting_gradient=1,
                 sample_mixture=False,
                 iterative_propagation_is_latent=False,
                 input_is_latent=False):
        if is_train:
            self.train()
        else:
            self.eval()

        if input_is_latent and not iterative_propagation_is_latent:
            raise ValueError(
                "input_is_latent and not iterative_propagation_is_latent Not implemented."
            )

        with torch.set_grad_enabled(is_train):
            # inputs is either the inputs of the encoder or the latent state when input_is_latent=True
            # In mixture density network (MDN), the output is either pi/MDN_var1/MDN_var2
            if self.parent.MDN_bool:
                inputs_decoded_MDN_var1 = []
                inputs_decoded_MDN_var2 = []
                inputs_decoded_MDN_var3 = []
                inputs_decoded_MDN_var4 = []
                inputs_decoded_pi = []
                if sample_mixture:
                    outputs = []
                else:
                    outputs_MDN_var1 = []
                    outputs_MDN_var2 = []
                    outputs_MDN_var3 = []
                    outputs_MDN_var4 = []
                    outputs_pi = []
            else:
                outputs = []
                inputs_decoded = []

            latent_states = []
            latent_states_pred = []
            RNN_internal_states = []
            RNN_outputs = []

            if self.parent.channels == 0 or input_is_latent:
                K, T, D = inputs.size()
            elif self.parent.channels == 1:
                K, T, D, Dc = inputs.size()
            elif self.parent.channels == 2:
                K, T, D, Dy, Dx = inputs.size()
            else:
                raise ValueError("Invalid number of channels {:}.".format(
                    self.parent.channels))

            # print("inputs.size()")
            # print(inputs.size())

            if (horizon is not None):
                if (not (K == 1)) or (not (T == 1)):
                    raise ValueError(
                        "Forward iterative called with K!=1 or T!=1 and a horizon. This is not allowed! K={:}, T={:}, D={:}"
                        .format(K, T, D))
                else:
                    # Horizon is not None and T=1, so forecast called in the testing phase
                    pass
            else:
                horizon = T

            if iterative_forecasting_prob == 0:
                assert T == horizon, "If iterative forecasting, with iterative_forecasting_prob={:}>0, the provided time-steps T cannot be {:}, but have to be horizon={:}.".format(
                    iterative_forecasting_prob, T, horizon)

            # When T>1, only inputs[:,0,:] is taken into account. The network is propagating its own predictions.
            input_t = inputs[:, 0].view(K, 1, *inputs.size()[2:])
            # print("start")
            # print("horizon")
            # print(horizon)
            # print(T)
            assert (T > 0)
            assert (horizon > 0)
            time_latent_prop = 0.0
            for t in range(horizon):
                # print("t")
                # print(t)
                # print(input_is_latent)
                # print(input_t.size())
                # BE CAREFULL: input may be the latent input!
                output, next_hidden_state, latent_state, latent_state_pred, RNN_output, input_decoded, time_latent_prop_t = self.forward_(
                    input_t,
                    init_hidden_state,
                    is_train=is_train,
                    is_latent=input_is_latent,
                    sample_mixture=sample_mixture)
                # assert(output[0,0,0]>0)
                # assert(output[0,0,0]<1)

                time_latent_prop += time_latent_prop_t

                # Settting the next input if t < horizon - 1
                if t < horizon - 1:

                    if iterative_forecasting_prob > 0.0:
                        # Iterative forecasting:
                        # with probability iterative_forecasting_prob propagate the state
                        # with probability (1-iterative_forecasting_prob) propagate the data
                        temp = torch.rand(1).data[0].item()
                    else:
                        temp = 0.0

                    if temp < (1 - iterative_forecasting_prob):
                        # with probability (1-iterative_forecasting_prob) propagate the data
                        input_t = inputs[:,
                                         t + 1].view(K, 1,
                                                     *inputs.size()[2:])
                    else:
                        # with probability iterative_forecasting_prob propagate the state
                        if iterative_propagation_is_latent:
                            # Changing the propagation to latent
                            input_is_latent = True
                            # input_t = latent_state_pred

                            if iterative_forecasting_gradient:
                                # Forecasting the prediction as a tensor in graph
                                input_t = latent_state_pred
                            else:
                                # Deatching, and propagating the prediction as data
                                input_t = latent_state_pred.detach()

                        else:
                            if self.parent.MDN_bool:
                                if sample_mixture:
                                    input_t = output
                                else:
                                    # Forecasting by propagating the mean (training)
                                    output_pi, output_MDN_var1, output_MDN_var2, output_MDN_var3, output_MDN_var4 = output
                                    output_pi = output_pi[:, 0]
                                    output_MDN_var1 = output_MDN_var1[:, 0]
                                    output_MDN_var2 = output_MDN_var2[:, 0]
                                    if self.parent.MDN_distribution in [
                                            "alanine", "trp"
                                    ]:
                                        output_MDN_var3 = output_MDN_var3[:, 0]
                                    if self.parent.MDN_distribution in [
                                            "alanine", "trp"
                                    ]:
                                        output_MDN_var4 = output_MDN_var4[:, 0]
                                    raise ValueError("Not implemented.")
                                    # outputMean = self.DECODER[-1].getPropagationMean(output_pi, output_MDN_var1)
                                    input_t = outputMean[:, None, :]
                            else:
                                if iterative_forecasting_gradient:
                                    # Forecasting the prediction as a tensor in graph
                                    input_t = output
                                else:
                                    # Deatching, and propagating the prediction as data
                                    # input_t = output.detach()
                                    input_t = Variable(output.data)

                if self.parent.MDN_bool:
                    input_decoded_pi, input_decoded_MDN_var1, input_decoded_MDN_var2, input_decoded_MDN_var3, input_decoded_MDN_var4 = input_decoded
                    inputs_decoded_MDN_var1.append(input_decoded_MDN_var1[:,
                                                                          0])
                    inputs_decoded_pi.append(input_decoded_pi[:, 0])
                    inputs_decoded_MDN_var2.append(input_decoded_MDN_var2[:,
                                                                          0])
                    if self.parent.MDN_distribution in ["alanine", "trp"]:
                        inputs_decoded_MDN_var3.append(
                            input_decoded_MDN_var3[:, 0])
                    if self.parent.MDN_distribution in ["alanine", "trp"]:
                        inputs_decoded_MDN_var4.append(
                            input_decoded_MDN_var4[:, 0])

                    if sample_mixture:
                        outputs.append(output[:, 0])
                    else:
                        # Propagating the mean prediction
                        output_pi, output_MDN_var1, output_MDN_var2, output_MDN_var3, output_MDN_var4 = output
                        outputs_MDN_var1.append(output_MDN_var1[:, 0])
                        outputs_pi.append(output_pi[:, 0])
                        outputs_MDN_var2.append(output_MDN_var2[:, 0])
                else:
                    outputs.append(output[:, 0])
                    inputs_decoded.append(input_decoded[:, 0])

                latent_states.append(latent_state[:, 0])
                latent_states_pred.append(latent_state_pred[:, 0])
                RNN_internal_states.append(next_hidden_state)
                RNN_outputs.append(RNN_output[:, 0])
                init_hidden_state = next_hidden_state

            if self.parent.MDN_bool:
                inputs_decoded_MDN_var1 = torch.stack(
                    inputs_decoded_MDN_var1).transpose(1, 0).contiguous()
                inputs_decoded_pi = torch.stack(inputs_decoded_pi).transpose(
                    1, 0).contiguous()
                inputs_decoded_MDN_var2 = torch.stack(
                    inputs_decoded_MDN_var2).transpose(1, 0).contiguous()
                if self.parent.MDN_distribution in ["alanine", "trp"]:
                    inputs_decoded_MDN_var3 = torch.stack(
                        inputs_decoded_MDN_var3).transpose(1, 0).contiguous()
                if self.parent.MDN_distribution in ["alanine", "trp"]:
                    inputs_decoded_MDN_var4 = torch.stack(
                        inputs_decoded_MDN_var4).transpose(1, 0).contiguous()

                if sample_mixture:
                    outputs = torch.stack(outputs)
                    outputs = outputs.transpose(1, 0)
                else:
                    outputs_pi = torch.stack(outputs_pi).transpose(
                        1, 0).contiguous()
                    outputs_MDN_var1 = torch.stack(outputs_MDN_var1).transpose(
                        1, 0).contiguous()
                    outputs_MDN_var2 = torch.stack(outputs_MDN_var2).transpose(
                        1, 0).contiguous()
                    if self.parent.MDN_distribution in ["alanine", "trp"]:
                        outputs_MDN_var3 = torch.stack(
                            outputs_MDN_var3).transpose(1, 0).contiguous()
                    if self.parent.MDN_distribution in ["alanine", "trp"]:
                        outputs_MDN_var4 = torch.stack(
                            outputs_MDN_var4).transpose(1, 0).contiguous()
                    outputs = [
                        outputs_pi, outputs_MDN_var1, outputs_MDN_var2,
                        outputs_MDN_var3, outputs_MDN_var4
                    ]
                inputs_decoded = [
                    inputs_decoded_pi, inputs_decoded_MDN_var1,
                    inputs_decoded_MDN_var2, inputs_decoded_MDN_var3,
                    inputs_decoded_MDN_var4
                ]
            else:
                outputs = torch.stack(outputs)
                outputs = outputs.transpose(1, 0)
                inputs_decoded = torch.stack(inputs_decoded)
                inputs_decoded = inputs_decoded.transpose(1, 0)

            latent_states = torch.stack(latent_states)
            latent_states_pred = torch.stack(latent_states_pred)
            RNN_outputs = torch.stack(RNN_outputs)

            latent_states = latent_states.transpose(1, 0)
            latent_states_pred = latent_states_pred.transpose(1, 0)
            RNN_outputs = RNN_outputs.transpose(1, 0)

        return outputs, next_hidden_state, latent_states, latent_states_pred, RNN_outputs, inputs_decoded, time_latent_prop

    def transposeHiddenState(self, hidden_state):
        # Transpose hidden state from batch_first to Layer first
        # (gru)  [K, L, H]    -> [L, K, H]
        # (lstm) [K, 2, L, H] -> [L, 2, K, H]
        if self.parent.params["RNN_cell_type"] in ["gru", "plain"]:
            hidden_state = hidden_state.transpose(0, 1)  #
        elif self.parent.params["RNN_cell_type"] == "lstm":
            hidden_state = hidden_state.transpose(0, 2)  # (lstm)
        else:
            raise ValueError("RNN_cell_type {:} not recognized".format(
                self.parent.params["RNN_cell_type"]))
        return hidden_state

    def forward(
        self,
        inputs,
        init_hidden_state,
        is_train=False,
        is_iterative_forecasting=False,
        iterative_forecasting_prob=0,
        iterative_forecasting_gradient=1,
        horizon=None,
        sample_mixture=False,
        input_is_latent=False,
        iterative_propagation_is_latent=False,
    ):
        # ARGUMENTS:
        #   inputs:                     The input to the RNN
        #   init_hidden_state:          The initial hidden state
        #   is_train:                   Whether it is training or evaluation
        #   is_latent:                  Whether we are forwarding the
        #                               latent dimension
        #   is_iterative_forecasting:   Whether to feed the predicted output
        #                               back in the input (iteratively)

        # ONLY RELEVANT FOR ITERATIVE FORECASTING
        #   horizon:            The iterative prediction horizon

        # print(inputs.size())

        if is_iterative_forecasting:
            return self.forecast(
                inputs,
                init_hidden_state,
                horizon,
                is_train=is_train,
                iterative_forecasting_prob=iterative_forecasting_prob,
                iterative_forecasting_gradient=iterative_forecasting_gradient,
                sample_mixture=sample_mixture,
                input_is_latent=input_is_latent,
                iterative_propagation_is_latent=iterative_propagation_is_latent
            )
        else:
            assert (input_is_latent == iterative_propagation_is_latent)
            return self.forward_(inputs,
                                 init_hidden_state,
                                 is_train=is_train,
                                 is_latent=input_is_latent,
                                 sample_mixture=sample_mixture)

    def forward_(self,
                 inputs,
                 init_hidden_state,
                 is_train=True,
                 is_latent=False,
                 sample_mixture=False):
        # TRANSPOSE FROM BATCH FIRST TO LAYER FIRST
        if self.parent.has_rnn:
            init_hidden_state = self.transposeHiddenState(init_hidden_state)

        if is_train:
            self.train()
        else:
            self.eval()

        # Time spent in propagation of the latent state
        time_latent_prop = 0.0
        with torch.set_grad_enabled(is_train):
            if self.parent.channels == 0 or is_latent:
                K, T, D = inputs.size()
            elif self.parent.channels == 1:
                K, T, D, Dc = inputs.size()
            elif self.parent.channels == 2:
                K, T, D, Dy, Dx = inputs.size()
            else:
                raise ValueError("Invalid number of channels {:}.".format(
                    self.parent.channels))

            # Swapping the inputs to RNN [K,T,LD]->[T, K, LD] (time first) LD=latent dimension
            inputs = inputs.transpose(1, 0)

            if (K != self.parent.batch_size and is_train == True
                    and (not self.parent.device_count > 1)):
                raise ValueError(
                    "Batch size {:d} does not match {:d} and model not in multiple GPUs."
                    .format(K, self.parent.batch_size))

            if self.parent.has_autoencoder and not is_latent:
                if D != self.parent.input_dim:
                    raise ValueError(
                        "Input dimension {:d} does not match {:d}.".format(
                            D, self.parent.input_dim))
                # Forward the encoder only in the original space
                encoder_output = self.forwardEncoder(inputs)
            else:
                encoder_output = inputs

            # print(" --- ")
            # print(encoder_output.size())

            decoder_input = encoder_output

            latent_states = encoder_output

            # print(decoder_input.size())
            # decoder_input = 0.0 * torch.ones_like(decoder_input)
            # print(ark)
            if self.parent.has_autoencoder:
                inputs_decoded = self.forwardDecoder(decoder_input)
                if self.parent.MDN_bool:
                    inputs_decoded_pi, inputs_decoded_MDN_var1, inputs_decoded_MDN_var2, inputs_decoded_MDN_var3, inputs_decoded_MDN_var4 = inputs_decoded
            else:
                inputs_decoded = decoder_input

            if self.parent.has_rnn:
                time0 = time.time()

                # Latent states are the autoencoded states BEFORE being past through the RNN
                RNN_outputs, next_hidden_state = self.forwardRNN(
                    encoder_output, init_hidden_state, is_train)

                # for name, param in self.RNN_OUTPUT[0].named_parameters():
                #     print("DEVICE {:} - PARAM {:} DEVICE {:}".format(init_hidden_state.get_device(), name, param.get_device()))

                # Output of the RNN passed through MLP and then softmaxed
                # print(RNN_outputs)
                # outputs = self.RNN_OUTPUT[0](RNN_outputs)
                # Forwarding the self.RNN_OUTPUT[0] depending if it is a MDN or a linear layer
                outputs = self.forwardRNNOutput(RNN_outputs)

                time1 = time.time()
                time_latent_prop += (time1 - time0)

                if self.parent.RNN_MDN_bool:
                    outputs = self.processOutputOfMDN(
                        outputs,
                        sample_mixture=sample_mixture,
                        multivariate=self.parent.RNN_MDN_multivariate,
                        distribution=self.parent.RNN_MDN_distribution,
                        MDN_output_dim=self.RNN_MDN_output_dim,
                        MDN_kernels=self.parent.RNN_MDN_kernels,
                        MDN_model=self.RNN_OUTPUT[-1],
                        perm_inv=False,
                        T=T,
                        K=K)
                    latent_states_pred = outputs
                else:
                    latent_states_pred = outputs
                    latent_states_pred = latent_states_pred.transpose(1, 0)

                # TRANSPOSING BATCH_SIZE WITH TIME
                RNN_outputs = RNN_outputs.transpose(1, 0)

                # print(outputs.size())
                # print("#####")
                # The predicted latent states are the autoencoded states AFTER being past through the RNN, before beeing decoded
                decoder_input_pred = outputs

                if self.parent.AE_convolutional:
                    latent_states_pred = latent_states_pred.view(
                        sl, bs, lat_dim, dx, dy)

                # TRANSPOSE BACK FROM LAYER FIRST TO BATCH FIRST
                next_hidden_state = self.transposeHiddenState(
                    next_hidden_state)

                # Output of the RNN (after the MLP) has dimension
                # [T, K, latend_dim]

                # print("#"*20)
                # print(len(decoder_input_pred))
                # print(decoder_input_pred[0].size())
                # print(decoder_input_pred[1].size())
                # print(decoder_input_pred[2].size())
                # print(ark)
                if self.parent.has_autoencoder:
                    if self.parent.RNN_MDN_bool and not sample_mixture:
                        # In case of:
                        # A. MD-RNN and a decoder
                        # B. We did not sample the MD-RNN
                        # THEN: Sample it before passing it through the decoder
                        decoder_input_pred = self.processOutputOfMDN(
                            decoder_input_pred,
                            sample_mixture=True,
                            multivariate=self.parent.RNN_MDN_multivariate,
                            MDN_output_dim=self.RNN_MDN_output_dim,
                            distribution=self.parent.RNN_MDN_distribution,
                            MDN_kernels=self.parent.RNN_MDN_kernels,
                            MDN_model=self.RNN_OUTPUT[0],
                            perm_inv=False,
                            num_particles=None,
                            Dx=None,
                            channels=0,
                            T=T,
                            K=K,
                        )
                        decoder_input_pred = decoder_input_pred.transpose(
                            1, 0).contiguous()
                    # print("decoder_input_pred.size():")
                    # print(decoder_input_pred.size())
                    outputs = self.forwardDecoder(decoder_input_pred)
                    if self.parent.MDN_bool:
                        outputs = self.processOutputOfMDN(
                            outputs,
                            sample_mixture=sample_mixture,
                            multivariate=self.parent.MDN_multivariate,
                            MDN_output_dim=self.MDN_output_dim,
                            distribution=self.parent.MDN_distribution,
                            MDN_kernels=self.parent.MDN_kernels,
                            MDN_model=self.DECODER[-1],
                            perm_inv=self.parent.AE_perm_invariant_latent_dim,
                            num_particles=self.parent.input_dim,
                            Dx=self.parent.Dx,
                            channels=self.parent.channels,
                            T=T,
                            K=K,
                        )
                    else:
                        # print(outputs.size())
                        outputs = outputs.transpose(1, 0).contiguous()
                        # print(outputs.size())
                elif (not self.parent.RNN_MDN_bool):
                    # Transposing the output of the RNN
                    outputs = outputs.transpose(1, 0).contiguous()

            else:
                outputs = []
                RNN_outputs = []
                latent_states_pred = []
                next_hidden_state = []

            latent_states = latent_states.transpose(1, 0).contiguous()
            if self.parent.MDN_bool:
                inputs_decoded_pi = inputs_decoded_pi.transpose(
                    1, 0).contiguous()
                inputs_decoded_MDN_var1 = inputs_decoded_MDN_var1.transpose(
                    1, 0).contiguous()
                inputs_decoded_MDN_var2 = inputs_decoded_MDN_var2.transpose(
                    1, 0).contiguous()
                if self.parent.MDN_distribution in ["alanine", "trp"]:
                    inputs_decoded_MDN_var3 = inputs_decoded_MDN_var3.transpose(
                        1, 0).contiguous()
                if self.parent.MDN_distribution in ["alanine", "trp"]:
                    inputs_decoded_MDN_var4 = inputs_decoded_MDN_var4.transpose(
                        1, 0).contiguous()
                inputs_decoded = [
                    inputs_decoded_pi, inputs_decoded_MDN_var1,
                    inputs_decoded_MDN_var2, inputs_decoded_MDN_var3,
                    inputs_decoded_MDN_var4
                ]
            else:
                inputs_decoded = inputs_decoded.transpose(1, 0).contiguous()
        return outputs, next_hidden_state, latent_states, latent_states_pred, RNN_outputs, inputs_decoded, time_latent_prop
        # return outputs

    def processOutputOfMDN(
        self,
        output,
        sample_mixture=False,
        multivariate=False,
        num_particles=None,
        MDN_kernels=None,
        Dx=None,
        MDN_output_dim=None,
        distribution=None,
        channels=0,
        MDN_model=None,
        perm_inv=False,
        T=None,
        K=None,
    ):
        outputs_pi, outputs_MDN_var1, outputs_MDN_var2, outputs_MDN_var3, outputs_MDN_var4 = output
        if sample_mixture:
            if multivariate:
                outputs_pi = torch.reshape(outputs_pi, (T * K, MDN_kernels))
                outputs_MDN_var1 = torch.reshape(
                    outputs_MDN_var1, (T * K, MDN_kernels, MDN_output_dim))
                outputs_MDN_var2 = torch.reshape(
                    outputs_MDN_var2,
                    (T * K, MDN_kernels, MDN_output_dim, MDN_output_dim))

                if perm_inv:
                    outputs_MDN_var1 = MDN_model.repeatAlongDim(
                        var=outputs_MDN_var1,
                        axis=0,
                        repeat_times=num_particles,
                        interleave=True)
                    outputs_MDN_var2 = MDN_model.repeatAlongDim(
                        var=outputs_MDN_var2,
                        axis=0,
                        repeat_times=num_particles,
                        interleave=True)
                    outputs_pi = MDN_model.repeatAlongDim(
                        var=outputs_pi,
                        axis=0,
                        repeat_times=num_particles,
                        interleave=True)
            elif distribution in ["alanine", "trp"]:
                MDN_output_dim_normal = self.DECODER[-1].output_dim_normal
                MDN_output_dim_von_mishes = self.DECODER[
                    -1].output_dim_von_mishes
                outputs_pi = torch.reshape(
                    outputs_pi, (T * K, MDN_output_dim, MDN_kernels))
                outputs_MDN_var1 = torch.reshape(
                    outputs_MDN_var1,
                    (T * K, MDN_output_dim_normal, MDN_kernels))
                outputs_MDN_var2 = torch.reshape(
                    outputs_MDN_var2,
                    (T * K, MDN_output_dim_normal, MDN_kernels))
                outputs_MDN_var3 = torch.reshape(
                    outputs_MDN_var3,
                    (T * K, MDN_output_dim_von_mishes, MDN_kernels))
                outputs_MDN_var4 = torch.reshape(
                    outputs_MDN_var4,
                    (T * K, MDN_output_dim_von_mishes, MDN_kernels))
            else:
                outputs_pi = torch.reshape(
                    outputs_pi, (T * K, MDN_output_dim, MDN_kernels))
                outputs_MDN_var1 = torch.reshape(
                    outputs_MDN_var1, (T * K, MDN_output_dim, MDN_kernels))
                outputs_MDN_var2 = torch.reshape(
                    outputs_MDN_var2, (T * K, MDN_output_dim, MDN_kernels))

            sample = MDN_model.sampleFromOutput(outputs_pi, outputs_MDN_var1,
                                                outputs_MDN_var2,
                                                outputs_MDN_var3,
                                                outputs_MDN_var4)
            outputs = self.parent.torch_dtype(sample)
            if channels == 0:
                outputs = torch.reshape(outputs, (T, K, MDN_output_dim))
            elif channels == 1:
                outputs = torch.reshape(outputs, (T, K, num_particles, Dx))
            else:
                raise ValueError("Not implemented.")
            outputs = outputs.transpose(1, 0).contiguous()
        else:
            outputs_pi = outputs_pi.transpose(1, 0).contiguous()
            outputs_MDN_var1 = outputs_MDN_var1.transpose(1, 0).contiguous()
            outputs_MDN_var2 = outputs_MDN_var2.transpose(1, 0).contiguous()
            if distribution in ["alanine", "trp"]:
                outputs_MDN_var3 = outputs_MDN_var3.transpose(1,
                                                              0).contiguous()
            if distribution in ["alanine", "trp"]:
                outputs_MDN_var4 = outputs_MDN_var4.transpose(1,
                                                              0).contiguous()
            outputs = [
                outputs_pi, outputs_MDN_var1, outputs_MDN_var2,
                outputs_MDN_var3, outputs_MDN_var4
            ]
        return outputs

    def forwardRNN(self, inputs, init_hidden_state, is_train):

        # The inputs are the latent_states
        T = inputs.size()[0]
        RNN_outputs = []
        # RNN_internal_states = []
        for t in range(T):
            input_t = inputs[t]
            next_hidden_state = []
            for ln in range(len(self.RNN)):
                hidden_state = init_hidden_state[ln]

                # TRANSFORMING THE HIDEN STATE TO TUPLE FOR LSTM
                # if not isinstance(hidden_state, tuple):
                # if len(hidden_state.size())==3: # [2, K, H]
                if self.parent.params["RNN_cell_type"] == "lstm":
                    hx, cx = init_hidden_state[ln]
                    hidden_state = tuple([hx, cx])

                RNN_output, next_hidden_state_layer = self.RNN[ln].forward(
                    input_t, hidden_state, is_train=is_train)

                # TRANSFORMING FROM TUPLE TO TORCH TENSOR
                # if len(next_hidden_state_layer.size())==3: # [2, K, H]
                # if isinstance(next_hidden_state_layer, tuple):
                if self.parent.params["RNN_cell_type"] == "lstm":
                    hx, cx = next_hidden_state_layer
                    next_hidden_state_layer = torch.stack([hx, cx])

                # RNN_output, next_hidden_state_layer = self.RNN[ln].forward(input_t, init_hidden_state[ln], is_train=is_train)
                next_hidden_state.append(next_hidden_state_layer)
                input_t = RNN_output

            init_hidden_state = next_hidden_state
            # RNN_internal_states.append(next_hidden_state)
            RNN_outputs.append(RNN_output)

        RNN_outputs = torch.stack(RNN_outputs)
        next_hidden_state = torch.stack(next_hidden_state)

        # print(RNN_outputs.size())
        # RNN_internal_states size is [T x NL x (2 x) BS x SL]
        # print(len(RNN_internal_states))
        # print(len(RNN_internal_states[0]))
        # print(RNN_internal_states[0][0][0].size())
        # print(RNN_outputs.size())
        return RNN_outputs, next_hidden_state

    def constructPermInvFeatures(self, data):
        if self.parent.AE_perm_invariant_feature == "mean":
            data = torch.mean(data, 1)
        elif self.parent.AE_perm_invariant_feature == "min":
            data, _ = torch.min(data, 1, keepdim=False)
        elif self.parent.AE_perm_invariant_feature == "max":
            data, _ = torch.max(data, 1, keepdim=False)
        else:
            raise ValueError("Invalid AE_perm_invariant_feature : {:}".format(
                elf.parent.AE_perm_invariant_feature))
        return data

    def forwardEncoder(self, inputs):
        # print("PROPAGATING THROUGH THE ENCODER")
        # PROPAGATING THROUGH THE ENCODER TO GET THE LATENT STATE
        outputs = []
        for input_t in inputs:
            output = input_t
            if self.parent.AE_perm_invariant_latent_dim:
                # In case there are no channels, adding the dummy channel dimension (1)
                if self.parent.channels == 0 and len(output.size()) == 2:
                    output = output.unsqueeze(2)
                for l in range(len(self.PERM_INV)):
                    output = self.PERM_INV[l](output)
                    if l < len(self.PERM_INV) - 1:
                        output = self.activation_general(output)
                    output = self.DROPOUT[0](output)
                output = self.constructPermInvFeatures(output)

            for l in range(len(self.ENCODER)):
                # In convolutional autoencoders, add the residual connections 0-2, 2-4, 4-6 (the dimension remains the same)
                # In normal autoencoder, add the residual connections 1-3, 3-5, 5-7
                if (self.parent.AE_convolutional and
                    (l > 0
                     and l % 2 == 0)) or (not self.parent.AE_convolutional and
                                          (l > 2 and l % 2 == 1)):
                    if self.parent.params["AE_residual"]:
                        # # RESIDUAL CONNECTION
                        # print("output.size()")
                        # print(output.size())
                        # print("output_res_prev.size()")
                        # print(output_res_prev.size())
                        # print("RESIDUAL connection between layers {:}-{:}".format(layer_prev, l))
                        output = output + output_res_prev
                if (self.parent.AE_convolutional
                        and l % 2 == 0) or (not self.parent.AE_convolutional
                                            and l % 2 == 1):
                    # print("output_res_prev.size()")
                    output_res_prev = output
                    # print(output_res_prev.size())
                    layer_prev = l
                if self.parent.AE_convolutional:
                    output = pad_circular(output, self.padding)
                output = self.ENCODER[l](output)

                if l < len(self.ENCODER) - 1:
                    output = self.activation_general(output)
                output = self.DROPOUT[0](output)

            outputs.append(output)
        outputs = torch.stack(outputs)
        return outputs

    def forwardDecoder(self, inputs):
        # Dimension of inputs: [T, K, latend_dim + noise_dim]
        if self.parent.MDN_bool:
            pi_ = []
            MDN_var1_ = []
            MDN_var2_ = []
            MDN_var3_ = []
            MDN_var4_ = []
        else:
            outputs = []
        # print("inputs.size()")
        # print(inputs.size())
        for input_t in inputs:
            output = input_t
            # print("input_t.size()")
            # print(input_t.size())
            # print(ark)
            # print(len(self.DECODER))
            for l in range(len(self.DECODER)):
                # print(output.size())
                if (self.parent.AE_convolutional and
                    (l > 0
                     and l % 2 == 0)) or (not self.parent.AE_convolutional and
                                          (l > 2 and l % 2 == 1)):
                    # RESIDUAL CONNECTION
                    if self.parent.params["AE_residual"]:
                        output = output + output_res_prev
                if (self.parent.AE_convolutional
                        and l % 2 == 0) or (not self.parent.AE_convolutional
                                            and l % 2 == 1):
                    output_res_prev = output
                    layer_prev = l
                if self.parent.AE_convolutional:
                    output = pad_circular(output, self.padding)
                # print("layer = {:}".format(l))
                # print("layer input size {:}".format(output.size()))

                # print(output)
                # print(output.size())
                output = self.DECODER[l](output)

                if l < len(self.DECODER) - 1:
                    # NO ACTIVATION AND DROPOUT IN THE LAST LAYER
                    output = self.activation_general(output)
                    output = self.DROPOUT[0](output)
                    # print("layer output size {:}".format(output.size()))
                elif l == len(self.DECODER) - 1:  #
                    # LAST LAYER OF DECODER
                    if self.parent.MDN_bool:
                        pi, MDN_var1, MDN_var2, MDN_var3, MDN_var4 = output
                        # print("MDN layer output size {:}".format(pi.size()))
                    else:
                        # SIGMOID ACTIVATION IN LAST LAYER IF SCALER IS MIN-MAX in [0,1]
                        if self.parent.scaler == "MinMaxZeroOne":
                            output = torch.sigmoid(output)
                        elif self.parent.scaler == "MinMaxMinusOneOne":
                            output = torch.tanh(output)
                else:
                    raise ValueError("This should not happen.")

            if self.parent.MDN_bool:
                pi_.append(pi)
                MDN_var1_.append(MDN_var1)
                MDN_var2_.append(MDN_var2)
                MDN_var3_.append(MDN_var3)
                MDN_var4_.append(MDN_var4)
            else:
                outputs.append(output)

        if self.parent.MDN_bool:
            pi_ = torch.stack(pi_)
            MDN_var1_ = torch.stack(MDN_var1_)
            MDN_var2_ = torch.stack(MDN_var2_)
            if self.parent.MDN_distribution in ["alanine", "trp"]:
                MDN_var3_ = torch.stack(MDN_var3_)
            if self.parent.MDN_distribution in ["alanine", "trp"]:
                MDN_var4_ = torch.stack(MDN_var4_)
            return [pi_, MDN_var1_, MDN_var2_, MDN_var3_, MDN_var4_]
        else:
            outputs = torch.stack(outputs)
            return outputs

    def forwardRNNOutput(self, inputs):
        # print("forwardRNNOutput()")
        # print(inputs.size())
        # Dimension of inputs: [T, K, latend_dim + noise_dim]
        if self.parent.RNN_MDN_bool:
            pi_ = []
            MDN_var1_ = []
            MDN_var2_ = []
            for input_t in inputs:
                output = self.RNN_OUTPUT[0](input_t)
                pi, MDN_var1, MDN_var2, _, _ = output
                pi_.append(pi)
                MDN_var1_.append(MDN_var1)
                MDN_var2_.append(MDN_var2)
            pi_ = torch.stack(pi_)
            MDN_var1_ = torch.stack(MDN_var1_)
            MDN_var2_ = torch.stack(MDN_var2_)
            return pi_, MDN_var1_, MDN_var2_, [None], [None]
        else:
            outputs = self.RNN_OUTPUT[0](inputs)
            return outputs
