#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class MixtureDensityNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_kernels,
        n_hidden=None,
        dist="normal",
        weight_sharing=0,
        multivariate=False,
        bounded=False,
        parametrization="scale_tril",
        activation="selu",
        sigma_max=0.4,
        fixed_kernels=False,
        train_kernels=False,
        multivariate_covariance_layer=False,
        MDN_multivariate_pretrain_diagonal=False,
    ):
        super(MixtureDensityNetwork, self).__init__()

        ########################################################
        ##### WEIGHT SHARING (option: weight_sharing)
        ########################################################

        # EXPLANATION :
        # The outputs have the dimension [K, output_dim]
        # If weight_sharing=1 - the output_dim outputs follow the same distribution p.
        # This distribution depends on the inputs. However, it is the SAME for all outputs if weight sharing=1.
        # So every output (from the total output_dim) is a sample from the same distribution.
        # If the inputs are permutation invariant and output_dim is the number of particles, the likelihood of every output should be the same (weight sharing)
        # All outputs belong to the same class, have the same likelihood, e.g. perm. invariance, particles, etc.

        # APPLICATION :
        # When modeling a system of particles with stochastic behavior, the output_dim might be the positions of some particles.
        # In this case, the outputs are of the same "kind" and should be additionally permutation invariant.
        # This is captured by the loss function, using a distributional loss on the output distribution,
        # where the output_dim positions are sampled from.
        # This is not the case if the output_dim is the position (x,y,z) of one particle (multiple channels in the output). In this case, weight sharing would mean that the distribution in all directions is the same.

        ########################################################
        ##### MULTIVARIATE MODELING (option: multivariate)
        ########################################################

        # EXPLANATION :
        # The outputs have the dimension [K, output_dim]
        # If multivariate=1 - we take into account correlations in the output by modeling a single vector random variable.
        # If dist="normal" we achieve this by modeling the covariance matrix

        assert (activation in ["selu", "tanh"])
        self.activation = nn.SELU() if (activation == "selu") else (
            nn.Tanh() if (activation == "tanh") else None)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=-1)
        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.n_kernels = n_kernels
        self.output_dim = output_dim
        self.weight_sharing = weight_sharing
        self.multivariate = multivariate
        self.bounded = bounded
        self.parametrization = parametrization
        self.fixed_kernels = fixed_kernels
        self.train_kernels = train_kernels
        self.multivariate_covariance_layer = multivariate_covariance_layer
        self.MDN_multivariate_pretrain_diagonal = MDN_multivariate_pretrain_diagonal
        assert (0 < sigma_max < 1.0)
        self.sigma_max = sigma_max

        if (weight_sharing and multivariate):
            raise ValueError(
                "The options multivariate and weight_sharing are not compatible."
            )

        self.dist = dist
        if self.dist == "normal":
            if self.weight_sharing:
                print("Mixture density network with weight sharing.")
                self.all_kernels_output_dim = n_kernels
            elif self.multivariate:
                print(
                    "Mixture density network with MULTIVARIATE distribution.")
                self.all_kernels_output_dim = n_kernels
            elif not self.weight_sharing and not self.multivariate:
                print("Mixture density network without weight sharing.")
                self.all_kernels_output_dim = output_dim * n_kernels
            else:
                raise ValueError("Invalid weight sharing variable {:}.".format(
                    self.weight_sharing))
        elif self.dist == "alanine":
            # self.output_dim_normal = 9
            # self.output_dim_von_mishes = 15
            self.defineAlanineVars()
            self.output_dim_normal = 17
            self.output_dim_von_mishes = 7
            self.all_kernels_output_dim_normal = self.output_dim_normal * n_kernels
            self.all_kernels_output_dim_von_mises = self.output_dim_von_mishes * n_kernels
        elif self.dist == "trp":
            self.defineTRPVars()
            self.output_dim_normal = 153 + 152
            self.output_dim_von_mishes = 151
            self.all_kernels_output_dim_normal = self.output_dim_normal * n_kernels
            self.all_kernels_output_dim_von_mises = self.output_dim_von_mishes * n_kernels
        else:
            raise ValueError("Unknown distribution {:}".format(self.dist))

        if self.dist == "normal" and not self.multivariate:
            self.buildGaussianMDN()
        elif self.dist == "normal" and self.multivariate:
            self.buildMultivariateGaussianMDN()
        elif self.dist in ["alanine", "trp"]:
            self.buildBADMDN()
        else:
            raise ValueError("Unknown distribution in MDN.")

        if self.parametrization not in [
                "scale_tril", "covariance_matrix", "precision_matrix"
        ]:
            raise ValueError("Invalid parametrization {:}".format(
                self.parametrization))

        if self.bounded:
            print("-- MDN with bounded output --")
        else:
            print("-- MDN with unbounded output --")

        self.initializeWeights()
        self.countParams()

        if self.fixed_kernels and not self.train_kernels:
            # FIXING KERNELS
            for name, param in self.named_parameters():
                if ("z_pi" in name) or ("z_h_pi" in name):
                    pass
                else:
                    param.requires_grad = False

    def isNoneOrZero(self, x):
        if x is None:
            return True
        if x == 0:
            return True
        return False

    def buildGaussianMDN(self):
        if self.isNoneOrZero(self.n_hidden):
            self.z_h = torch.nn.Identity()
            if self.fixed_kernels: self.z_h_pi = torch.nn.Identity()
            self.z_pi = nn.Linear(self.input_dim, self.all_kernels_output_dim)
            self.z_mu = nn.Linear(self.input_dim, self.all_kernels_output_dim)
            self.z_sigma = nn.Linear(self.input_dim,
                                     self.all_kernels_output_dim)
        else:
            # Common latent space
            self.z_h = nn.Sequential(
                nn.Linear(self.input_dim, self.n_hidden),
                self.activation,
            )
            if self.fixed_kernels:
                # Adding an additional latent space - different from the fixed kernels
                self.z_h_pi = nn.Sequential(
                    nn.Linear(self.input_dim, self.n_hidden),
                    self.activation,
                )
            self.z_pi = nn.Linear(self.n_hidden, self.all_kernels_output_dim)
            self.z_mu = nn.Linear(self.n_hidden, self.all_kernels_output_dim)
            self.z_sigma = nn.Linear(self.n_hidden,
                                     self.all_kernels_output_dim)
        return 0

    def buildMultivariateGaussianMDN(self):
        if self.isNoneOrZero(self.n_hidden):
            self.z_h = torch.nn.Identity()
            if self.fixed_kernels: self.z_h_pi = torch.nn.Identity()
            self.z_pi = nn.Linear(self.input_dim, self.all_kernels_output_dim)
            self.z_mu = nn.Linear(
                self.input_dim, self.all_kernels_output_dim * self.output_dim)
            # Covariance matrix - positive definite
            # Parametrization - C = L L^T, where L is lower triangular
            # Positive diagonal elements
            self.z_L_diag = nn.Linear(
                self.input_dim, self.all_kernels_output_dim * self.output_dim)
            # Elements on the lower part of L (lower triangular matrix)
            self.z_L_lower = nn.Linear(
                self.input_dim,
                int(self.all_kernels_output_dim * self.output_dim *
                    (self.output_dim - 1) / 2))
        else:
            # Common latent space
            self.z_h = nn.Sequential(
                nn.Linear(self.input_dim, self.n_hidden),
                self.activation,
            )
            if self.fixed_kernels:
                # Adding an additional latent space - different from the fixed kernels
                self.z_h_pi = nn.Sequential(
                    nn.Linear(self.input_dim, self.n_hidden),
                    self.activation,
                )

            self.z_pi = nn.Linear(self.n_hidden, self.all_kernels_output_dim)
            self.z_mu = nn.Linear(
                self.n_hidden, self.all_kernels_output_dim * self.output_dim)
            self.z_L_diag = nn.Linear(
                self.n_hidden, self.all_kernels_output_dim * self.output_dim)

            if self.multivariate_covariance_layer:
                self.z_h_L_lower = nn.Sequential(
                    nn.Linear(self.input_dim, self.n_hidden * self.n_hidden),
                    self.activation,
                )

            if self.multivariate_covariance_layer:
                self.z_L_lower = nn.Linear(
                    self.n_hidden * self.n_hidden,
                    int(self.all_kernels_output_dim * self.output_dim *
                        (self.output_dim - 1) / 2))
            else:
                self.z_L_lower = nn.Linear(
                    self.n_hidden,
                    int(self.all_kernels_output_dim * self.output_dim *
                        (self.output_dim - 1) / 2))
        return 0

    def buildBADMDN(self):
        # Bonds/Angles/Dihedrals Mixture Density Network
        if self.isNoneOrZero(self.n_hidden):
            raise ValueError("ERROR: Not implemented.")
        else:
            # Common latent space
            self.z_h = nn.Sequential(
                nn.Linear(self.input_dim, self.n_hidden),
                self.activation,
            )
            if self.fixed_kernels:
                # Adding an additional latent space - different from the fixed kernels
                self.z_h_pi = nn.Sequential(
                    nn.Linear(self.input_dim, self.n_hidden),
                    self.activation,
                )
            self.z_pi = nn.Linear(
                self.n_hidden, self.all_kernels_output_dim_normal +
                self.all_kernels_output_dim_von_mises)
            self.z_mu = nn.Linear(self.n_hidden,
                                  self.all_kernels_output_dim_normal)
            self.z_sigma = nn.Linear(self.n_hidden,
                                     self.all_kernels_output_dim_normal)
            self.z_loc = nn.Linear(self.n_hidden,
                                   self.all_kernels_output_dim_von_mises)
            self.z_concentration = nn.Linear(
                self.n_hidden, self.all_kernels_output_dim_von_mises)
        return 0

    def initializeWeights(self):
        print("Initializing parameters...\n")
        for name, param in self.named_parameters():
            # print(name)
            if ('z_L_lower' in name) or ('z_h_L_lower' in name):
                if self.multivariate_covariance_layer:
                    param.data.fill_(1e-6)
                else:
                    param.data.fill_(1e-3)
            elif 'weight' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
            else:
                raise ValueError("NAME {:} NOT FOUND!".format(name))
        print("Parameteres initialized!")
        return 0

    def countParams(self):
        from functools import reduce
        total_params = sum(
            reduce(lambda a, b: a * b, x.size()) for x in self.parameters())
        print("MDN - Total number of trainable parameters {:}\n".format(
            total_params))
        return total_params

    def reshapeOutput(self, var, var_dim=None):
        if var_dim is None: var_dim = self.output_dim
        if not self.weight_sharing:
            var = var.view(-1, var_dim, self.n_kernels)
            return var
        elif self.weight_sharing:
            var = var[:, None, :]
            var = self.repeatAlongDim(var, axis=1, repeat_times=var_dim)
            # var = var.repeat(1, self.output_dim, 1)
            return var

    def forward(self, data):
        if self.dist == "normal" and not self.multivariate:
            return self.forwardGaussianMDN(data)
        elif self.dist == "normal" and self.multivariate:
            return self.forwardMultivariateGaussianMDN(data)
        elif self.dist in ["alanine", "trp"]:
            return self.forwardBADMDN(data)
        else:
            raise ValueError("Unknown distribution in MDN.")

    def getLatentState(self, data):
        if not self.fixed_kernels:
            z_h = self.z_h(data)
            z_h_pi = z_h
        elif self.fixed_kernels and self.train_kernels:
            z_h = self.z_h(torch.zeros_like(data))
            z_h_pi = z_h
        elif self.fixed_kernels and not self.train_kernels:
            z_h = self.z_h(torch.zeros_like(data))
            z_h_pi = self.z_h_pi(data)
        else:
            raise ValueError("Error.")

        if self.multivariate:
            if self.multivariate_covariance_layer:
                z_h_L_lower = self.z_h_L_lower(data)
            else:
                z_h_L_lower = z_h
        else:
            z_h_L_lower = None

        return z_h, z_h_pi, z_h_L_lower

    def forwardBADMDN(self, data):
        z_h, z_h_pi, _ = self.getLatentState(data)
        pi = self.softmax(self.reshapeOutput(self.z_pi(z_h_pi)))

        mu = self.reshapeOutput(self.z_mu(z_h), var_dim=self.output_dim_normal)
        sigma = self.reshapeOutput(self.sigma_max *
                                   torch.sigmoid(self.z_sigma(z_h)),
                                   var_dim=self.output_dim_normal)

        loc = self.reshapeOutput(self.z_loc(z_h),
                                 var_dim=self.output_dim_von_mishes)
        concentration = self.reshapeOutput(self.softplus(
            self.z_concentration(z_h)),
                                           var_dim=self.output_dim_von_mishes)

        # print(mu.max())
        # print(mu.min())

        # print(sigma.max())
        # print(sigma.min())

        # print(loc.max())
        # print(loc.min())

        # print(concentration.max())
        # print(concentration.min())

        # print(ark)
        return pi, mu, sigma, loc, concentration

    def forwardGaussianMDN(self, data):
        # z_h = self.z_h(data)
        # Defining the mixing coefficients of the Gaussian Mixture
        z_h, z_h_pi, _ = self.getLatentState(data)
        # pi = self.reshapeOutput(self.softmax(self.z_pi(z_h)))
        # print(self.z_pi(z_h_pi).size())
        pi = self.softmax(self.reshapeOutput(self.z_pi(z_h_pi)))
        mu = self.reshapeOutput(self.z_mu(z_h))
        # sigma = self.reshapeOutput(torch.exp(self.z_sigma(z_h)))
        # sigma = self.reshapeOutput(self.softplus(self.z_sigma(z_h)))
        # sigma = 0.001 * sigma
        # sigma = 0.1 * torch.ones_like(sigma)
        # sigma = 0.2 * torch.ones_like(sigma)
        sigma = self.reshapeOutput(self.sigma_max *
                                   torch.sigmoid(self.z_sigma(z_h)))
        # print(mu.max())
        # print(mu.min())

        # print(sigma.max())
        # print(sigma.min())
        # print(ark)
        return pi, mu, sigma, [None], [None]

    def forwardMultivariateGaussianMDN(self, data):
        # z_h = self.z_h(data)
        z_h, z_h_pi, z_h_L_lower = self.getLatentState(data)
        # Defining the mixing coefficients of the Gaussian Mixture
        # pi = self.softmax(self.z_pi(z_h))
        pi = self.softmax(self.z_pi(z_h_pi))
        pi = pi.view(-1, self.n_kernels)

        mu = self.z_mu(z_h)
        mu = mu.view(-1, self.n_kernels, self.output_dim)
        # Covariance matrix - parametrized as sigma = L L^T
        # L - lower triangular matrix with positive elements in the diagonal
        z_L_diag = self.z_L_diag(z_h)
        # L_diag = self.softplus(z_L_diag)
        L_diag = np.sqrt(self.sigma_max) * self.softplus(z_L_diag)
        L_diag = L_diag.view(-1, self.n_kernels, self.output_dim)
        # assert(torch.all(L_diag>0))
        # L_diag = torch.clamp(L_diag, min=1e-4)
        self.L_diag = L_diag

        L_lower = self.z_L_lower(z_h_L_lower)
        if self.MDN_multivariate_pretrain_diagonal:
            # Pretraining the diagonal of the Covariance matrix
            # All non-diagonal elements are set to zero
            L_lower = 0.0 * L_lower

        L_lower = L_lower.view(
            -1, self.n_kernels,
            int(self.output_dim * (self.output_dim - 1) / 2))
        batch_size, num_kernels, output_dim = L_diag.size()
        L = torch.diag_embed(L_diag, offset=0, dim1=-2, dim2=-1)
        element_iter = 0
        for i in range(output_dim - 1):
            offset = i + 1
            elements_in_diag = output_dim - offset
            elements_up_to = element_iter + elements_in_diag
            temp = torch.diag_embed(L_lower[:, :, element_iter:elements_up_to],
                                    offset=(-offset),
                                    dim1=-2,
                                    dim2=-1)
            L += temp
            element_iter += elements_in_diag

        # L_T = L.transpose(2,3)
        # L_ = L.view(batch_size*num_kernels, output_dim, output_dim)
        # L_T = L_T.view(batch_size*num_kernels, output_dim, output_dim)
        # COV = torch.matmul(L_, L_T)
        # var2 = COV.view(batch_size, num_kernels, output_dim, output_dim)
        # print(COV)
        # print(ark)

        # # print(L_lower)
        # # print(L_diag)

        if self.parametrization in ["covariance_matrix", "precision_matrix"]:
            L_T = L.transpose(2, 3)
            L = L.view(batch_size * num_kernels, output_dim, output_dim)
            L_T = L_T.view(batch_size * num_kernels, output_dim, output_dim)
            COV = torch.matmul(L, L_T)
            var2 = COV.view(batch_size, num_kernels, output_dim, output_dim)
        elif self.parametrization in ["scale_tril"]:
            var2 = L
        else:
            raise ValueError("Invalid parametrization {:}".format(
                self.parametrization))
        # print(var2.size())
        # print(var2[0,0])
        # print(var2[1,0])
        # print(var2[2,0])
        # print(var2[3,0])
        # print(var2[4,0])
        # print(ark)
        self.L = L

        # print(var2)
        # print(ark)

        # print(pi) # 0.1556, 0.8444
        # print(mu) #tensor([[[ 0.9368, -1.3900], [-0.0618,  0.2420]]])
        return pi, mu, var2, [None], [None]

    # def getPropagationMean(self, pi, mu):
    #     values, indices = torch.max(pi, 1)
    #     output = mu[:,:,indices[0]]
    #     return output

    def sampleFromOutput(self, pi, var1, var2, var3=None, var4=None):
        # print(pi.size())
        # print(var1.size())
        # print(var2.size())
        # print(ark)

        # Sample every mixture
        # output = torch.normal(mu, sigma)
        dists = self.getDistributions(var1, var2, var3, var4)
        if self.dist == "normal":
            output = dists.sample()
        elif self.dist in ["alanine", "trp"]:
            output_normal = dists[0].sample()
            output_angles = dists[1].sample()
            output = torch.cat((output_normal, output_angles), dim=1)
            # print(output_normal.size())
            # print(output_angles.size())
            # print(output.size())
            # print(ark)
        # output = dist.sample()
        output = self.processOutputs(output)
        # print(np.shape(output))

        # print(ark)
        # Sample the mixture coefficient (the kernel number)
        pi_sampled = self.sampleMixtureCoefficient(pi)
        # Select only the output of the sampled gaussian kernel from the total n_kernels kernels
        # output = torch.stack([torch.stack([output[i,j,pi_sampled[i,j]] for j in range(self.output_dim)]) for i in range(K)])
        # print(data.size())
        # print(output.size())
        # print(ark)
        # print(pi_sampled.size())
        # print(output.size())
        output = self.selectMixture(output, pi_sampled)
        # print(output.size())
        # print(ark)
        output = output.cpu().detach().numpy()
        return output

    def getMultivariateNormalDistribution(self, var1, var2):
        if self.parametrization == "covariance_matrix":
            return torch.distributions.MultivariateNormal(
                loc=var1, covariance_matrix=var2)
        elif self.parametrization == "precision_matrix":
            return torch.distributions.MultivariateNormal(
                loc=var1, precision_matrix=var2)
        elif self.parametrization == "scale_tril":
            return torch.distributions.MultivariateNormal(loc=var1,
                                                          scale_tril=var2)
        else:
            raise ValueError("Invalid parametrization {:}".format(
                self.parametrization))

    def getDistributions(self, var1, var2, var3=None, var4=None):
        if self.dist == "normal" and not self.multivariate:
            dist = torch.distributions.Normal(loc=var1, scale=var2)
        elif self.dist == "normal" and self.multivariate:
            dist = self.getMultivariateNormalDistribution(var1, var2)
        elif self.dist in ["alanine", "trp"]:
            assert (var3 is not None)
            assert (var4 is not None)
            dist_normal = torch.distributions.Normal(loc=var1, scale=var2)
            dist_angles = torch.distributions.VonMises(loc=var3,
                                                       concentration=var4)
            dist = [dist_normal, dist_angles]
        else:
            raise ValueError("Unknown distribution in MDN.")
        return dist

    def sample(self, data):
        K, D = data.size()
        # Get the mixture parameters
        pi, var1, var2, var3, var4 = self.forward(data)
        output = self.sampleFromOutput(pi, var1, var2, var3, var4)
        return output

    def sampleMixtureCoefficient(self, pi):
        if not self.multivariate:
            K, O, n_kernels = pi.size()
            assert (n_kernels == self.n_kernels)
            assert (O == self.output_dim)
            pi = pi.view(K * O, self.n_kernels)
            coef = torch.multinomial(pi, 1).view(K * O)
            coef = coef.view(K, O)
        elif self.multivariate:
            K, n_kernels = pi.size()
            coef = torch.multinomial(pi, 1).view(K)
        return coef

    def selectMixture(self, var, pi_sampled):
        if not self.multivariate:
            K, O = pi_sampled.size()
            assert (O == self.output_dim)
            var = torch.stack([
                torch.stack([
                    var[k, j, pi_sampled[k, j]] for j in range(self.output_dim)
                ]) for k in range(K)
            ])
        elif self.multivariate:
            K = pi_sampled.size()[0]
            var = torch.stack([
                torch.stack(
                    [var[k, pi_sampled[k], j] for j in range(self.output_dim)])
                for k in range(K)
            ])
        return var

    def repeatAlongDim(self, var, axis, repeat_times, interleave=False):
        if not interleave:
            repeat_idx = len(var.size()) * [1]
            repeat_idx[axis] = repeat_times
            var = var.repeat(*repeat_idx)
        else:
            var = var.repeat_interleave(repeat_times, dim=axis)
        return var

    def processTargets(self, targets):
        if self.dist == "normal" and self.bounded:
            # Implementing the logit function z = log(x / (1-x))
            # Targets lie in (0,1)
            assert (torch.all(targets < 1))
            assert (torch.all(targets > 0))
            targets = torch.log(targets / (1 - targets))
        elif self.dist in ["alanine", "trp"] and self.bounded:
            assert (targets.size()[1] == self.dims_total)
            assert (torch.all(targets[:, self.scaling_dims, :] < 1))
            assert (torch.all(targets[:, self.scaling_dims, :] > 0))
            targets[:, self.scaling_dims, :] = torch.log(
                targets[:, self.scaling_dims, :] /
                (1.0 - targets[:, self.scaling_dims, :]))
        return targets

    def processOutputs(self, outputs):
        if self.dist == "normal" and self.bounded:
            # Implementing the logistic function x = 1/(1 + exp(-z))
            outputs = 1.0 / (1.0 + torch.exp(-outputs))
        elif self.dist in ["alanine", "trp"] and self.bounded:
            assert (outputs.size()[1] == self.dims_total)
            # print(outputs.size())
            # torch.Size([499, 24, 5])
            # print(self.scaling_dims)
            # print(outputs[:,0,:].max())
            # print(outputs[:,0,:].min())
            outputs[:, self.scaling_dims, :] = 1.0 / (
                1.0 + torch.exp(-outputs[:, self.scaling_dims, :]))
            # After the rescaling, the values should be bounded to [0,1]
            assert (torch.all(outputs[:, self.scaling_dims, :] < 1))
            assert (torch.all(outputs[:, self.scaling_dims, :] > 0))
            # print(outputs[:,0,:].max())
            # print(outputs[:,0,:].min())
            # print(ark)
        else:
            pass
        return outputs

    def defineTRPVars(self):
        self.dims_total = 456
        self.dims_bonds = 153
        self.dims_angles = 152
        self.dims_dehedrals = 151
        self.dims_bonds_ = list(np.arange(0, self.dims_bonds, 1))
        self.dims_angles_ = list(
            np.arange(self.dims_bonds, self.dims_bonds + self.dims_angles, 1))
        self.dims_dehedrals_ = list(
            np.arange(self.dims_bonds + self.dims_angles,
                      self.dims_bonds + self.dims_angles + self.dims_dehedrals,
                      1))
        self.scaling_dims = self.dims_bonds_ + self.dims_angles_
        return 0

    def defineAlanineVars(self):
        self.dims_total = 24
        self.dims_bonds = 9
        self.dims_angles = 8
        self.dims_dehedrals = 7
        self.dims_bonds_ = list(np.arange(0, self.dims_bonds, 1))
        self.dims_angles_ = list(
            np.arange(self.dims_bonds, self.dims_bonds + self.dims_angles, 1))
        self.dims_dehedrals_ = list(
            np.arange(self.dims_bonds + self.dims_angles,
                      self.dims_bonds + self.dims_angles + self.dims_dehedrals,
                      1))
        self.scaling_dims = self.dims_bonds_ + self.dims_angles_
        return 0

    def MDN_loss_fn(self, targets, pi, var1, var2, var3=None, var4=None):
        # print(targets.size())
        # print(var1.size())
        # print(var2.size())
        # # print(var3.size())
        # # print(var4.size())
        # print(pi.size())
        # print(self.dist)
        # print(ark)
        if self.dist == "normal" and not self.multivariate:
            m = torch.distributions.Normal(loc=var1, scale=var2)
        elif self.dist == "normal" and self.multivariate:
            # assert(torch.all(var1<=100))
            # assert(torch.all(var2<=100))
            try:
                m = self.getMultivariateNormalDistribution(var1, var2)
            except Exception as inst:
                # print(var2)
                # print(var1.size())
                # print(var2.size())
                # print(targets.size())
                print(self.L)
                print(inst)
                print(ark)
        elif self.dist in ["alanine", "trp"]:
            m_normal = torch.distributions.Normal(loc=var1, scale=var2)
            m_angles = torch.distributions.VonMises(loc=var3,
                                                    concentration=var4)

        else:
            raise ValueError("Unknown distribution in MDN.")

        # if torch.any(torch.isnan(m)): raise ValueError("m is nan")
        # if torch.any(torch.isinf(m)): raise ValueError("m is inf")

        # Adding a dimension for the kernels
        targets = targets[:, :, None]
        # Copying in order to compute the probability for each kernel
        # targets = targets.repeat(1, 1, self.n_kernels)
        targets = self.repeatAlongDim(targets,
                                      axis=2,
                                      repeat_times=self.n_kernels)

        if self.multivariate: targets = targets.transpose(1, 2)
        # print("targets.size()")
        # print(targets.size())
        # print(m.sample().size())

        targets = self.processTargets(targets)

        # print("targets.size()")
        # print(targets.size())
        # print(m.sample().size())

        # print(ark)
        # print(targets.size())
        # print(ark)

        # torch.Size([272, 3, 2])
        # torch.Size([272, 3, 2, 2])
        # torch.Size([272, 3, 2])
        # torch.Size([272, 2, 3])

        if self.dist in ["alanine", "trp"]:

            log_prob_normal = m_normal.log_prob(
                targets[:, :self.output_dim_normal])
            # print(targets[:,self.output_dim_normal:].max())
            # print(targets[:,self.output_dim_normal:].min())
            # print(ark)
            log_prob_angles = m_angles.log_prob(
                targets[:, self.output_dim_normal:])
            # print(log_prob_normal.size())
            # print(log_prob_angles.size())
            log_prob = torch.cat((log_prob_normal, log_prob_angles), dim=1)
            # print(log_prob.size())
            # print(ark)
        else:
            log_prob = m.log_prob(targets)

        # print(pi.max())
        # print(pi.min())

        # print(ark)
        pi = torch.clamp(pi, min=1e-15)
        log_pi = torch.log(pi)

        # log_pi = torch.clamp(log_pi, min=-1, max=2)
        # log_prob = torch.clamp(log_prob, min=-1, max=2)

        # log_pi = log_pi[:,None,:]
        # log_pi = log_pi.repeat(1, self.output_dim, 1)

        # print(var1.size())
        # print(var2.size())

        # print(log_pi.size())
        # print(log_prob.size())
        # print(log_pi.max())
        # print(log_pi.min())
        # print(log_prob.max())
        # print(log_prob.min())

        # print(ark)
        #
        # print(torch.max(log_prob, 1))
        # # print(torch.max(log_prob, 1)[0])
        # print(ark)
        # # torch.Size([272, 3])
        # # torch.Size([272, 3])

        sum_logs = log_prob + log_pi
        # print(sum_logs)
        # print(ark)
        # if sum_logs.max() > 15:
        #     print(log_prob.max())
        #     print(log_pi.max())
        #     print(ark)

        if not self.multivariate:
            # LOG-SUM STABILIZATION TRICK
            # print(sum_logs.size())
            sum_logs_max = torch.max(sum_logs, 2)[0]
            sum_logs_max = sum_logs_max[:, :, None]
            loss = torch.exp(sum_logs - sum_logs_max)
            # print(loss.max())
            # print(loss.min())
            # print(ark)
            loss = torch.sum(loss, dim=2)
            if torch.any(torch.isnan(loss)):
                raise ValueError("torch.sum(loss, dim=2) is nan")
            if torch.any(torch.isinf(loss)):
                raise ValueError("torch.sum(loss, dim=2) is inf")
            loss = -torch.log(loss) - sum_logs_max[:, :, 0]
        elif self.multivariate:
            # LOG-SUM STABILIZATION TRICK
            sum_logs_max = torch.max(sum_logs, 1)[0]
            sum_logs_max = sum_logs_max[:, None]
            loss = torch.exp(sum_logs - sum_logs_max)
            loss = torch.sum(loss, dim=1)
            if torch.any(torch.isnan(loss)):
                raise ValueError("torch.sum(loss, dim=1) is nan")
            if torch.any(torch.isinf(loss)):
                raise ValueError("torch.sum(loss, dim=1) is inf")
            loss = -torch.log(loss) - sum_logs_max[:, 0]

        # print(loss.min())
        # print(loss.mean())
        # print(loss.max())
        # print("###")
        # print(loss.size())
        # print(sum_logs_max.size())
        # print(ark)
        if torch.any(torch.isnan(loss)):
            raise ValueError("-torch.log(loss) - sum_logs_max is nan")
        if torch.any(torch.isinf(loss)):
            raise ValueError("-torch.log(loss) - sum_logs_max is inf")
        return torch.mean(loss)


if __name__ == '__main__':
    # # m = torch.distributions.Normal(loc=var1, scale=var2)

    # # loc = torch.tensor([1.0])
    # loc = torch.tensor([0.0])
    # concentration = torch.tensor([1.0])
    # m = torch.distributions.VonMises(loc, concentration)

    # temp = m.sample_n(1000)
    # print(temp.min()*180/3.14)
    # print(temp.max()*180/3.14)

    # print(temp.min())
    # print(temp.max())

    # temp2 = m.log_prob(temp)
    # print(temp2)

    n_samples = 1000
    epsilon1 = torch.randn(n_samples).view(-1, 1)
    epsilon2 = torch.randn(n_samples).view(-1, 1)
    x_train = torch.linspace(-10, 10, n_samples)
    x_train = x_train.view(-1, 1)

    y_train1 = 3 * np.cos(0.4 * x_train) + 0.2 * x_train + 2 * epsilon1
    y_train2 = 10.0 + 3 * np.cos(0.4 * x_train) + 0.2 * x_train + 2 * epsilon2
    # y_train2 = 3*np.cos(0.4*x_train) + 0.2*x_train + 2*epsilon2
    y_train3 = 2 * np.sin(
        0.4 * x_train) + 0.2 * x_train + 1 * epsilon2 + 2 * epsilon1
    y_train4 = 1 * np.sin(
        0.4 * x_train) + 0.2 * x_train + 5 * epsilon2 + 2 * epsilon1
    y_train5 = 2 * np.sin(0.4 * x_train +
                          0.05) + 0.2 * x_train + 0.2 * epsilon2

    x_test = torch.linspace(-3, 3, n_samples).view(-1, 1)

    # y_train = torch.cat((y_train1, y_train2, y_train3, y_train4, y_train5), axis=1)
    y_train = torch.cat((y_train1, y_train2), axis=1)

    # print(np.shape(x_train))
    # print(np.shape(y_train))
    # # x_test = torch.linspace(-15, 15, n_samples).view(-1, 1)
    # print(np.shape(x_test))

    # # for output_dim in range(y_train.size()[1]):
    # #     plt.figure(figsize=(8, 8))
    # #     plt.scatter(x_train, y_train[:,output_dim], alpha=0.4)
    # #     plt.xlabel('x')
    # #     plt.ylabel('f(x)')
    # #     plt.show()

    # # n_input = 1
    # # n_hidden = 20
    # # n_output = 2

    # # model = nn.Sequential(
    # #     nn.Linear(n_input, n_hidden),
    # #     nn.Tanh(),
    # #     nn.Linear(n_hidden, n_output)
    # #     )

    # # loss_fn = nn.MSELoss()

    # # optimizer = torch.optim.RMSprop(model.parameters())

    # # for epoch in range(epochs_max):
    # #     y_pred = model(x_train)
    # #     loss = loss_fn(y_pred, y_train)
    # #     optimizer.zero_grad()
    # #     loss.backward()
    # #     optimizer.step()
    # #     if epoch % 1000 == 0:
    # #         print(loss.data.tolist())

    # torch.manual_seed(1)
    # np.random.seed(1)

    # # epochs_max = 10000
    # epochs_max = 3000
    # plot_every = 50
    # weight_sharing = 0
    # multivariate = 0
    # dist = "normal"
    # n_kernels = 1
    # bounded = True
    # learning_rate = 0.0001
    # n_hidden = 500
    # activation = "selu"
    # # activation = "tanh"
    # sigma_max = 0.1

    # epochs_max = 10000
    # # epochs_max = 3000
    # plot_every = 50
    # weight_sharing = 0
    # multivariate = 0
    # dist = "normal"
    # n_kernels = 1
    # bounded = False
    # learning_rate = 0.0001
    # n_hidden = 50
    # # activation = "selu"
    # activation = "tanh"
    # sigma_max = 0.6

    # # epochs_max = 10000
    # epochs_max = 1000
    # plot_every = 50
    # weight_sharing = 0
    # multivariate = 0
    # dist = "normal"
    # n_kernels = 1
    # bounded = True
    # learning_rate = 0.01
    # n_hidden = 50
    # # activation = "selu"
    # activation = "tanh"
    # sigma_max = 0.2

    # # epochs_max = 10000
    # # # epochs_max = 1000
    # # plot_every = 50
    # # weight_sharing = 0
    # # multivariate = 0
    # # dist = "normal"
    # # n_kernels = 1
    # # bounded = False
    # # learning_rate = 0.001
    # # n_hidden = 50
    # # # activation = "selu"
    # # activation = "tanh"
    # # sigma_max = 0.3

    # # epochs_max = 10000
    # # # epochs_max = 2000
    # # plot_every = 50
    # # weight_sharing = 0
    # # multivariate = 1
    # # dist = "normal"
    # # n_kernels = 1
    # # bounded = False
    # # learning_rate = 0.00001
    # # n_hidden = 400
    # # # activation = "selu"
    # # activation = "tanh"
    # # sigma_max = 0.3

    model = MixtureDensityNetwork(input_dim=1,
                                  output_dim=y_train.size(1),
                                  n_hidden=n_hidden,
                                  n_kernels=n_kernels,
                                  dist=dist,
                                  weight_sharing=weight_sharing,
                                  multivariate=multivariate,
                                  bounded=bounded,
                                  activation=activation,
                                  sigma_max=sigma_max)

    # x_train = 0.9*(x_train - x_train.min()) / (x_train.max() - x_train.min()) + 0.01
    # x_test = 0.9*(x_test - x_test.min()) / (x_test.max() - x_test.min()) + 0.01
    # y_train = 0.9*(y_train - y_train.min()) / (y_train.max() - y_train.min()) + 0.01

    # SLACK = 0.05
    SLACK = 1e-3
    x_train_min = x_train.min() - SLACK
    x_train_max = x_train.max() + SLACK

    x_test_min = x_test.min() - SLACK
    x_test_max = x_test.max() + SLACK

    y_train_min = y_train.min() - SLACK
    y_train_max = y_train.max() + SLACK

    # x_train = (x_train - x_train_min) / (x_train_max - x_train_min)
    # x_test = (x_test - x_test_min) / (x_test_max - x_test_min)
    # y_train = (y_train - y_train_min) / (y_train_max - y_train_min)

    samples_ = model.sample(x_test)

    # temp = torch.rand(10)
    # print(temp)
    # temp = model.processOutputs(temp)
    # print(temp)
    # temp = model.processTargets(temp)
    # print(temp)
    # print(ark)

    # print(np.shape(x_train))
    # print(np.shape(y_train))
    # print(np.shape(x_test))
    # print(np.shape(samples_))
    # # print(ark)
    # for output_dim in range(y_train.size()[1]):
    #     plt.figure(figsize=(8, 8))
    #     plt.scatter(x_train, y_train[:,output_dim], alpha=0.4, label="data")
    #     plt.scatter(x_test, samples_[:,output_dim], alpha=0.4, label="model samples")
    #     plt.xlabel('x')
    #     plt.ylabel('f(x)')
    #     plt.legend()
    #     plt.show()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs_max):
        pi, var1, var2, var3, var4 = model.forward(x_train)
        loss = model.MDN_loss_fn(y_train, pi, var1, var2, var3, var4)
        if epoch % plot_every == 0:
            print("Epoch {:}/{:}, {:.2f} percent".format(
                epoch, epochs_max, epoch / epochs_max * 100.0))
            print(loss.data.tolist())
            samples = model.sample(x_test)
            # for output_dim in range(y_train.size()[1]):
            #     plt.figure(figsize=(8, 8))
            #     plt.scatter(x_train, y_train[:,output_dim], alpha=0.4)
            #     plt.scatter(x_test, samples[:,output_dim], alpha=0.4)
            #     plt.xlabel('x')
            #     plt.ylabel('f(x)')
            #     plt.show()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for output_dim in range(y_train.size()[1]):
        plt.figure(figsize=(8, 8))
        plt.scatter(x_train,
                    y_train[:, output_dim],
                    alpha=0.4,
                    label="data",
                    rasterized=True)
        plt.scatter(x_test,
                    samples[:, output_dim],
                    alpha=0.4,
                    label="model samples",
                    rasterized=True)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.show()

    # # torch.save(model.state_dict(), "./model")
