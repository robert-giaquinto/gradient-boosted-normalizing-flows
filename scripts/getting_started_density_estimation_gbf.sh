# Load defaults for all experiments
source ./scripts/experiment_config_density.sh

# activate virtual environment
source ./venv/bin/activate

# variables specific to this experiment
exp_name=density_estimation
iters_per_component=20000
num_steps=160000
learning_rate=0.0005
plot_resolution=250
dataset=8gaussians
num_components=8
num_flows=1
regularization_rate=0.8

python -m toy_experiment --dataset ${dataset} \
       --experiment_name ${exp_name} \
       --no_cuda \
       --no_annealing \
       --no_lr_schedule \
       --num_workers 1 \
       --num_steps ${num_steps} \
       --learning_rate ${learning_rate} \
       --iters_per_component ${iters_per_component} \
       --flow boosted \
       --num_components ${num_components} \
       --num_flows ${num_flows} \
       --component_type realnvp \
       --num_base_layers 1 \
       --base_network tanh \
       --h_size 256 \
       --regularization_rate ${regularization_rate} \
       --batch_size 64 \
       --manual_seed 1 \
       --log_interval ${logging} \
       --plot_resolution ${plot_resolution} \
       --plot_interval 1000 ;



