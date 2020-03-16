# Load defaults for all experiments
source ./scripts/experiment_config.sh

# activate virtual environment, comment out if not using
source ./venv/bin/activate

# variables specific to this experiment
experiment_name=example
dataset=caltech # freyfaces
seed=123
num_flows=2
vae_layers=linear

# define boosting components:
component_type=realnvp
h_size=64
base_network=tanh  # use a TanH network for REALNVP
layers=1  # number of layers in the TanH networks used by REALNVP

regularization_rate=1.0
annealing_schedule=50     # override default for this small example
epochs_per_component=200  # override default for this small example
num_components=2
epochs=$((num_components * epochs_per_component))

python main_experiment.py --dataset ${dataset} \
       --experiment_name ${experiment_name} \
       --testing \
       --nll_samples 100 \
       --nll_mb 50 \
       --no_cuda \
       --num_workers 1 \
       --rho_init decreasing \
       --learning_rate ${learning_rate} \
       --epochs ${epochs} \
       --annealing_schedule ${annealing_schedule} \
       --epochs_per_component ${epochs_per_component} \
       --early_stopping_epochs ${early_stop} \
       --vae_layers ${vae_layers} \
       --flow boosted \
       --component_type ${component_type} \
       --num_base_layers ${layers} \
       --base_network ${base_network} \
       --h_size ${h_size} \
       --num_components ${num_components} \
       --regularization_rate ${regularization_rate} \
       --num_flows ${num_flows} \
       --z_size ${z_size} \
       --batch_size ${batch_size} \
       --manual_seed ${seed} \
       --plot_interval ${plotting};
