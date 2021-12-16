#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
#!/usr/bin/env python
import numpy as np


class scaler(object):
    def __init__(
        self,
        scaler_type,
        data_min,
        data_max,
        common_scaling_per_input_dim=False,
        common_scaling_per_channels=False,
        channels=0,
        slack=0.0,
    ):
        # Data are of three possible types:
        # Type 1:
        #         (T, input_dim)
        # Type 2:
        #         (T, input_dim, Cx) (input_dim is the N_particles, perm_inv, etc.)
        # Type 3:
        #         (T, input_dim, Cx, Cy, Cz)
        self.scaler_type = scaler_type

        if self.scaler_type not in ["MinMaxZeroOne"]:
            raise ValueError("Scaler {:} not implemented.".format(
                self.scaler_type))

        data_max = np.array(data_max)
        data_min = np.array(data_min)
        range_ = data_max - data_min
        self.data_min = data_min - slack * range_
        self.data_max = data_max + slack * range_

        self.common_scaling_per_input_dim = common_scaling_per_input_dim
        self.common_scaling_per_channels = common_scaling_per_channels
        self.channels = channels

    def scaleData(self, batch_of_sequences, single_sequence=False):
        if single_sequence: batch_of_sequences = batch_of_sequences[np.newaxis]
        # Size of the batch_of_sequences is [K, T, ...]
        # Size of the batch_of_sequences is [K, T, D]
        # Size of the batch_of_sequences is [K, T, D, C]
        # Size of the batch_of_sequences is [K, T, D, C, C]
        # Size of the batch_of_sequences is [K, T, D, C, C, C]

        self.data_shape = np.shape(batch_of_sequences)
        self.data_shape_length = len(self.data_shape)

        if self.scaler_type == "MinMaxZeroOne":
            data_min = self.repeatScalerParam(
                self.data_min, self.data_shape,
                self.common_scaling_per_input_dim,
                self.common_scaling_per_channels)
            data_max = self.repeatScalerParam(
                self.data_max, self.data_shape,
                self.common_scaling_per_input_dim,
                self.common_scaling_per_channels)
            assert (np.all(np.shape(batch_of_sequences) == np.shape(data_min)))
            assert (np.all(np.shape(batch_of_sequences) == np.shape(data_max)))
            batch_of_sequences_scaled = np.array(
                (batch_of_sequences - data_min) / (data_max - data_min))
        # elif self.scaler_type == "MinMaxMinusOneOne":
        #     data_min = self.repeatScalerParam(self.data_min, self.data_shape, self.common_scaling_per_input_dim, self.common_scaling_per_channels)
        #     data_max = self.repeatScalerParam(self.data_max, self.data_shape, self.common_scaling_per_input_dim, self.common_scaling_per_channels)
        #     assert(np.all(np.shape(batch_of_sequences)==np.shape(data_min)))
        #     assert(np.all(np.shape(batch_of_sequences)==np.shape(data_max)))
        #     batch_of_sequences_scaled = np.array((2*batch_of_sequences-data_min-data_max)/(data_max-data_min))
        # elif self.scaler_type == "Standard":
        #     data_mean = self.repeatScalerParam(self.data_mean, self.data_shape, self.common_scaling_per_input_dim, self.common_scaling_per_channels)
        #     data_std = self.repeatScalerParam(self.data_std, self.data_shape, self.common_scaling_per_input_dim, self.common_scaling_per_channels)
        #     assert(np.all(np.shape(batch_of_sequences)==np.shape(data_mean)))
        #     assert(np.all(np.shape(batch_of_sequences)==np.shape(data_std)))
        #     batch_of_sequences_scaled = np.array((batch_of_sequences-data_mean)/data_std)
        else:
            raise ValueError("Scaler not implemented.")

        if single_sequence:
            batch_of_sequences_scaled = batch_of_sequences_scaled[0]
        return batch_of_sequences_scaled

    def repeatScalerParam(self, data, shape, common_scaling_per_input_dim,
                          common_scaling_per_channels):
        # Size of the batch_of_sequences is [K, T, ...]
        # Size of the batch_of_sequences is [K, T, D]
        # Size of the batch_of_sequences is [K, T, D, C]
        # Size of the batch_of_sequences is [K, T, D, C, C]
        # Size of the batch_of_sequences is [K, T, D, C, C, C]
        # Running through the shape in reverse order
        if common_scaling_per_input_dim:
            D = shape[2]
            # Commong scaling for all inputs !
            data = np.repeat(data[np.newaxis], D, 0)

        # Running through the shape in reverse order
        if common_scaling_per_channels:
            # Repeating the scaling for each channel
            assert (len(shape[::-1][:-3]) == self.channels)
            for channel_dim in shape[::-1][:-3]:
                data = np.repeat(data[np.newaxis], channel_dim, 0)
                data = np.swapaxes(data, 0, 1)
        T = shape[1]
        data = np.repeat(data[np.newaxis], T, 0)
        K = shape[0]
        data = np.repeat(data[np.newaxis], K, 0)
        return data

    def descaleData(self, batch_of_sequences_scaled, single_sequence=True):
        if single_sequence:
            batch_of_sequences_scaled = batch_of_sequences_scaled[np.newaxis]
        # Size of the batch_of_sequences_scaled is [K, T, ...]
        # Size of the batch_of_sequences_scaled is [K, T, D]
        # Size of the batch_of_sequences_scaled is [K, T, D, C]
        # Size of the batch_of_sequences_scaled is [K, T, D, C, C]
        # Size of the batch_of_sequences_scaled is [K, T, D, C, C, C]
        self.data_shape = np.shape(batch_of_sequences_scaled)
        self.data_shape_length = len(self.data_shape)
        if self.scaler_type == "MinMaxZeroOne":
            data_min = self.repeatScalerParam(
                self.data_min, self.data_shape,
                self.common_scaling_per_input_dim,
                self.common_scaling_per_channels)
            data_max = self.repeatScalerParam(
                self.data_max, self.data_shape,
                self.common_scaling_per_input_dim,
                self.common_scaling_per_channels)
            assert (np.all(
                np.shape(batch_of_sequences_scaled) == np.shape(data_min)))
            assert (np.all(
                np.shape(batch_of_sequences_scaled) == np.shape(data_max)))
            batch_of_sequences = np.array(batch_of_sequences_scaled *
                                          (data_max - data_min) + data_min)
        # elif self.scaler_type == "MinMaxMinusOneOne":
        #     data_min = self.repeatScalerParam(self.data_min, self.data_shape, self.common_scaling_per_input_dim, self.common_scaling_per_channels)
        #     data_max = self.repeatScalerParam(self.data_max, self.data_shape, self.common_scaling_per_input_dim, self.common_scaling_per_channels)
        #     assert(np.all(np.shape(batch_of_sequences_scaled)==np.shape(data_min)))
        #     assert(np.all(np.shape(batch_of_sequences_scaled)==np.shape(data_max)))
        #     batch_of_sequences = np.array(batch_of_sequences_scaled*(data_max - data_min) + data_min + data_max)/2.0
        # elif self.scaler_type == "Standard":
        #     data_mean = self.repeatScalerParam(self.data_mean, self.data_shape, self.common_scaling_per_input_dim, self.common_scaling_per_channels)
        #     data_std = self.repeatScalerParam(self.data_std, self.data_shape, self.common_scaling_per_input_dim, self.common_scaling_per_channels)
        #     assert(np.all(np.shape(batch_of_sequences_scaled)==np.shape(data_mean)))
        #     assert(np.all(np.shape(batch_of_sequences_scaled)==np.shape(data_std)))
        #     batch_of_sequences = np.array(batch_of_sequences_scaled*data_std + data_mean)
        else:
            raise ValueError("Scaler not implemented.")

        if single_sequence: batch_of_sequences = batch_of_sequences[0]
        return np.array(batch_of_sequences)


class scalerBAD(object):
    # Bonds/Angles/Dihedrals scaler
    def __init__(
        self,
        scaler_type,
        dims_total,
        dims_bonds,
        dims_angles,
        dims_dehedrals,
        data_min_bonds,
        data_max_bonds,
        data_min_angles,
        data_max_angles,
        slack,
    ):
        self.scaler_type = scaler_type
        if self.scaler_type not in ["MinMaxZeroOne"]:
            raise ValueError("Scaler {:} not implemented.".format(
                self.scaler_type))
        # print(np.shape(data_min_bonds))
        # print(np.shape(data_max_bonds))
        # print(np.shape(data_min_angles))
        # print(np.shape(data_max_angles))

        range_bonds = data_max_bonds - data_min_bonds
        self.data_min_bonds = data_min_bonds - slack * range_bonds
        self.data_max_bonds = data_max_bonds + slack * range_bonds

        range_angles = data_max_angles - data_min_angles
        self.data_min_angles = data_min_angles - slack * np.abs(range_angles)
        self.data_max_angles = data_max_angles + slack * np.abs(range_angles)

        self.data_min = np.concatenate(
            (self.data_min_bonds, self.data_min_angles), axis=0)
        self.data_max = np.concatenate(
            (self.data_max_bonds, self.data_max_angles), axis=0)

        self.dims_total = dims_total
        self.dims_bonds = dims_bonds
        self.dims_angles = dims_angles
        self.dims_dehedrals = dims_dehedrals

        self.dims_bonds_ = list(np.arange(0, self.dims_bonds, 1))
        self.dims_angles_ = list(
            np.arange(self.dims_bonds, self.dims_bonds + self.dims_angles, 1))
        self.dims_dehedrals_ = list(
            np.arange(self.dims_bonds + self.dims_angles,
                      self.dims_bonds + self.dims_angles + self.dims_dehedrals,
                      1))

        self.scaling_dims = self.dims_bonds_ + self.dims_angles_
        # print(self.dims_bonds_)
        # print(self.dims_angles_)
        # print(self.dims_dehedrals_)

    def scaleData(self, batch_of_sequences, single_sequence=False):
        if single_sequence: batch_of_sequences = batch_of_sequences[np.newaxis]
        # Size of the batch_of_sequences is [K, T, ...]
        # Size of the batch_of_sequences is [K, T, D]
        # Size of the batch_of_sequences is [K, T, D, C]
        # Size of the batch_of_sequences is [K, T, D, C, C]
        # Size of the batch_of_sequences is [K, T, D, C, C, C]

        self.data_shape = np.shape(batch_of_sequences)
        self.data_shape_length = len(self.data_shape)

        batch_of_sequences_scaled = batch_of_sequences.copy()
        if self.scaler_type == "MinMaxZeroOne":
            data_min = self.repeatScalerParam(self.data_min, self.data_shape)
            data_max = self.repeatScalerParam(self.data_max, self.data_shape)
            assert (np.all(
                np.shape(batch_of_sequences_scaled[:, :, self.scaling_dims]) ==
                np.shape(data_min)))
            assert (np.all(
                np.shape(batch_of_sequences_scaled[:, :, self.scaling_dims]) ==
                np.shape(data_max)))
            batch_of_sequences_scaled[:, :, self.scaling_dims] = np.array(
                (batch_of_sequences_scaled[:, :, self.scaling_dims] - data_min)
                / (data_max - data_min))
        else:
            raise ValueError("Scaler not implemented.")

        if single_sequence:
            batch_of_sequences_scaled = batch_of_sequences_scaled[0]
        return batch_of_sequences_scaled

    def repeatScalerParam(self, data, shape):
        T = shape[1]
        data = np.repeat(data[np.newaxis], T, 0)
        K = shape[0]
        data = np.repeat(data[np.newaxis], K, 0)
        return data

    def descaleData(self, batch_of_sequences_scaled, single_sequence=True):
        if single_sequence:
            batch_of_sequences_scaled = batch_of_sequences_scaled[np.newaxis]
        batch_of_sequences = batch_of_sequences_scaled.copy()
        self.data_shape = np.shape(batch_of_sequences)
        self.data_shape_length = len(self.data_shape)
        if self.scaler_type == "MinMaxZeroOne":
            data_min = self.repeatScalerParam(self.data_min, self.data_shape)
            data_max = self.repeatScalerParam(self.data_max, self.data_shape)
            assert (np.all(
                np.shape(batch_of_sequences[:, :, self.scaling_dims]) ==
                np.shape(data_min)))
            assert (np.all(
                np.shape(batch_of_sequences[:, :, self.scaling_dims]) ==
                np.shape(data_max)))
            batch_of_sequences[:, :, self.scaling_dims] = np.array(
                batch_of_sequences[:, :, self.scaling_dims] *
                (data_max - data_min) + data_min)
        else:
            raise ValueError("Scaler not implemented.")

        if single_sequence: batch_of_sequences = batch_of_sequences[0]
        return np.array(batch_of_sequences)
