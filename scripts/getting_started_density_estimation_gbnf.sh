# Load defaults for all experiments
source ./scripts/experiment_config_uci.sh

# activate virtual environment
source ./venv/bin/activate

# variables specific to this experiment
experiment_name=getting_started
dataset=miniboone
seed=1
num_flows=5
component_type=glow
h_size_factor=5
coupling_network=tanh

epochs_per_component=100  # override default for this small example
num_components=2
lr_schedule=cosine
lr_restarts=1
optimizer=adam
batch_size=512

early_stop=$((epochs_per_component / lr_restarts / 2))
epochs=$((num_components * epochs_per_component))

warmup_epochs=1

python -m density_experiment --dataset ${dataset} \
       --experiment_name ${experiment_name} \
       --print_log \
       --no_tensorboard \
       --testing \
       --no_cuda \
       --num_workers 1 \
       --optimizer ${optimizer} \
       --lr_schedule ${lr_schedule} \
       --lr_restarts ${lr_restarts} \
       --warmup_epochs ${warmup_epochs} \
       --epochs ${epochs} \
       --epochs_per_component ${epochs_per_component} \
       --early_stopping_epochs ${early_stop} \
       --batch_size ${batch_size} \
       --flow boosted \
       --rho_init decreasing \
       --component_type ${component_type} \
       --coupling_network ${coupling_network} \
       --h_size_factor ${h_size_factor} \
       --num_components ${num_components} \
       --num_flows ${num_flows} \
       --manual_seed ${seed} 

echo "Job complete!"
