#!/bin/bash


cd ../../Methods

##############################################
# LOCAL TESTING OF MODEL TRAINED ON BARRY
##############################################

#######################
# SYSTEM PARAMETERS
#######################
CUDA_DEVICES=3
system_name=Alanine
input_dim=24

##############################################
# AUTOENCODER
##############################################
AE_layers_num=4
AE_layers_size=50
AE_residual=0
activation_str_general=tanh
latent_state_dim=1


##############################################
# TRAINING
##############################################
weight_decay=0.0
scaler=MinMaxZeroOne
batch_size=32
learning_rate=0.001
max_rounds=4
overfitting_patience=10
make_videos=0

retrain=0
# max_epochs=1000
max_epochs=1

##############################################
# MDN AT AUTOENCODER OUTPUT
##############################################
MDN_kernels=5
MDN_hidden_units=50
MDN_weight_sharing=0
MDN_multivariate=0
MDN_sigma_max=0.8
MDN_multivariate_covariance_layer=0
MDN_distribution=alanine

##############################################
# LOSSES
##############################################


sequence_length=1


##############################################################################
##############################################################################
### TRAIN DECODER MDN KERNELS
### (NETWORK IS PLUGGED BUT NOT TRAINED
### THE KERNELS ARE INDEPENDENT OF THE INPUT)
##############################################################################
##############################################################################

mode=all
# mode=test
# mode=plot
MDN_fixed_kernels=1
MDN_train_kernels=1
retrain=0
reconstruction_loss=1
output_forecasting_loss=0
latent_forecasting_loss=0
prediction_horizon=8000
# num_test_ICS=224
num_test_ICS=2

write_to_log=0

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 RUN.py md_arnn \
--mode $mode \
--system_name $system_name \
--cudnn_benchmark 1 \
--write_to_log $write_to_log \
--input_dim $input_dim \
--output_forecasting_loss $output_forecasting_loss \
--latent_forecasting_loss $latent_forecasting_loss \
--reconstruction_loss $reconstruction_loss \
--scaler $scaler \
--sequence_length $sequence_length \
--learning_rate $learning_rate \
--weight_decay $weight_decay \
--batch_size $batch_size \
--overfitting_patience $overfitting_patience \
--max_epochs $max_epochs \
--max_rounds $max_rounds \
--random_seed 7 \
--display_output 1 \
--retrain $retrain \
--make_videos $make_videos \
--activation_str_general $activation_str_general \
--AE_layers_num $AE_layers_num \
--AE_layers_size $AE_layers_size \
--AE_residual $AE_residual \
--latent_state_dim $latent_state_dim  \
--MDN_sigma_max $MDN_sigma_max \
--MDN_weight_sharing $MDN_weight_sharing \
--MDN_multivariate $MDN_multivariate \
--MDN_kernels $MDN_kernels \
--MDN_hidden_units $MDN_hidden_units \
--MDN_fixed_kernels $MDN_fixed_kernels \
--MDN_train_kernels $MDN_train_kernels \
--MDN_distribution $MDN_distribution \
--MDN_multivariate_covariance_layer $MDN_multivariate_covariance_layer \
--compute_spectrum 0 \
--prediction_horizon $prediction_horizon \
--num_test_ICS $num_test_ICS \
--test_on_train 0 \
--test_on_val 0 \
--test_on_test 1 \
--plot_state_distributions 0 \
--plot_state_distributions_system 0 \
--plot_system 1 \
--plot_testing_ics_examples 0


# mode=all
# mode=test

MDN_fixed_kernels=1
MDN_train_kernels=0
retrain=1
reconstruction_loss=1
output_forecasting_loss=0
latent_forecasting_loss=0

prediction_horizon=3800
num_test_ICS=10

write_to_log=1

# CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 RUN.py md_arnn \
# --mode $mode \
# --system_name $system_name \
# --cudnn_benchmark 1 \
# --write_to_log $write_to_log \
# --input_dim $input_dim \
# --output_forecasting_loss $output_forecasting_loss \
# --latent_forecasting_loss $latent_forecasting_loss \
# --reconstruction_loss $reconstruction_loss \
# --scaler $scaler \
# --sequence_length $sequence_length \
# --learning_rate $learning_rate \
# --weight_decay $weight_decay \
# --batch_size $batch_size \
# --overfitting_patience $overfitting_patience \
# --max_epochs $max_epochs \
# --max_rounds $max_rounds \
# --random_seed 7 \
# --display_output 1 \
# --retrain $retrain \
# --make_videos $make_videos \
# --activation_str_general $activation_str_general \
# --AE_layers_num $AE_layers_num \
# --AE_layers_size $AE_layers_size \
# --AE_residual $AE_residual \
# --latent_state_dim $latent_state_dim  \
# --MDN_sigma_max $MDN_sigma_max \
# --MDN_weight_sharing $MDN_weight_sharing \
# --MDN_multivariate $MDN_multivariate \
# --MDN_kernels $MDN_kernels \
# --MDN_hidden_units $MDN_hidden_units \
# --MDN_fixed_kernels $MDN_fixed_kernels \
# --MDN_train_kernels $MDN_train_kernels \
# --MDN_distribution $MDN_distribution \
# --MDN_multivariate_covariance_layer $MDN_multivariate_covariance_layer \
# --compute_spectrum 0 \
# --prediction_horizon $prediction_horizon \
# --num_test_ICS $num_test_ICS \
# --test_on_train 0 \
# --test_on_val 0 \
# --test_on_test 1 \
# --plot_state_distributions 1 \
# --plot_state_distributions_system 1 \
# --plot_system 1 \
# --plot_testing_ics_examples 1



# mode=all
mode=test
# mode=plot
retrain=0
reconstruction_loss=0
output_forecasting_loss=0
latent_forecasting_loss=1
train_rnn_only=1

sequence_length=200



rnn_layers_num=1
rnn_layers_size=10

# prediction_horizon=3800
# num_test_ICS=224

prediction_horizon=3800
num_test_ICS=1

RNN_MDN_kernels=4
RNN_MDN_multivariate=0
RNN_MDN_hidden_units=10
RNN_MDN_sigma_max=0.1
RNN_MDN_fixed_kernels=0
RNN_MDN_train_kernels=0
RNN_MDN_multivariate_covariance_layer=0

# CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 RUN.py md_arnn \
# --mode $mode \
# --system_name $system_name \
# --cudnn_benchmark 1 \
# --write_to_log $write_to_log \
# --input_dim $input_dim \
# --output_forecasting_loss $output_forecasting_loss \
# --latent_forecasting_loss $latent_forecasting_loss \
# --reconstruction_loss $reconstruction_loss \
# --scaler $scaler \
# --sequence_length $sequence_length \
# --learning_rate $learning_rate \
# --weight_decay $weight_decay \
# --batch_size $batch_size \
# --overfitting_patience $overfitting_patience \
# --max_epochs $max_epochs \
# --max_rounds $max_rounds \
# --random_seed 7 \
# --display_output 1 \
# --retrain $retrain \
# --make_videos $make_videos \
# --activation_str_general $activation_str_general \
# --AE_layers_num $AE_layers_num \
# --AE_layers_size $AE_layers_size \
# --AE_residual $AE_residual \
# --latent_state_dim $latent_state_dim  \
# --MDN_sigma_max $MDN_sigma_max \
# --MDN_weight_sharing $MDN_weight_sharing \
# --MDN_multivariate $MDN_multivariate \
# --MDN_kernels $MDN_kernels \
# --MDN_hidden_units $MDN_hidden_units \
# --MDN_fixed_kernels $MDN_fixed_kernels \
# --MDN_train_kernels $MDN_train_kernels \
# --MDN_distribution $MDN_distribution \
# --MDN_multivariate_covariance_layer $MDN_multivariate_covariance_layer \
# --RNN_cell_type lstm \
# --RNN_layers_num $rnn_layers_num  \
# --RNN_layers_size $rnn_layers_size  \
# --RNN_activation_str tanh \
# --teacher_forcing_forecasting 1 \
# --iterative_latent_forecasting 1 \
# --multiscale_forecasting 0 \
# --iterative_propagation_is_latent 1 \
# --train_rnn_only $train_rnn_only \
# --compute_spectrum 0 \
# --RNN_MDN_kernels $RNN_MDN_kernels \
# --RNN_MDN_multivariate $RNN_MDN_multivariate \
# --RNN_MDN_hidden_units $RNN_MDN_hidden_units \
# --RNN_MDN_sigma_max $RNN_MDN_sigma_max \
# --RNN_MDN_fixed_kernels $RNN_MDN_fixed_kernels \
# --RNN_MDN_train_kernels $RNN_MDN_train_kernels \
# --RNN_MDN_multivariate_covariance_layer $RNN_MDN_multivariate_covariance_layer \
# --prediction_horizon $prediction_horizon \
# --num_test_ICS $num_test_ICS \
# --test_on_train 1 \
# --test_on_val 1 \
# --test_on_test 1 \
# --plot_state_distributions 0 \
# --plot_state_distributions_system 0 \
# --plot_system 1 \
# --plot_testing_ics_examples 1




