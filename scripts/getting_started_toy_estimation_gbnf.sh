# Load defaults for all experiments
source ./scripts/experiment_config_density.sh

# activate virtual environment
source ./venv/bin/activate

# variables specific to this experiment
exp_name=toy_estimation
#iters_per_component=10000
iters_per_component=2500
learning_rate=0.001
plot_resolution=100
dataset=8gaussians
num_components=8
num_flows=4

num_steps=$((num_components * iters_per_component * 2))

python -m toy_experiment --dataset ${dataset} \
       --experiment_name ${exp_name} \
       --print_log \
       --no_cuda \
       --no_annealing \
       --lr_schedule cosine \
       --max_grad_norm 20.0 \
       --no_batch_norm \
       --warmup_iters 50 \
       --num_workers 3 \
       --num_steps ${num_steps} \
       --learning_rate ${learning_rate} \
       --iters_per_component ${iters_per_component} \
       --flow boosted \
       --num_components ${num_components} \
       --num_flows ${num_flows} \
       --component_type realnvp \
       --coupling_network_depth 1 \
       --coupling_network tanh \
       --h_size 256 \
       --batch_size 100 \
       --manual_seed 1 \
       --rho_iters 0 \
       --rho_init uniform \
       --log_interval 500 \
       --plot_resolution ${plot_resolution} \
       --plot_interval 500 ;



