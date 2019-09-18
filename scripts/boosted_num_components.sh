# Hypothesis: More components should lower the loss
# Experiment: Run planar flows for varying numbers of components

# Load defaults for all experiments
source ./scripts/experiment_config.sh

# activate virtual environment
source ./venv/bin/activate

# define variable specific to this experiment
exp_name=num_components
annealing_schedule=25
burnin=50
regularization=0.1
flow_length=2


# 1 component
python main_experiment.py --dataset ${dataset} \
       --validation \
       --experiment_name ${exp_name} \
       --num_workers ${num_workers} \
       --epochs ${epochs} \
       --early_stopping_epochs ${early_stop} \
       --annealing_schedule ${annealing_schedule} \
       --burnin ${burnin} \
       --z_size ${z_size} \
       --flow boosted \
       --regularization_rate ${regularization} \
       --num_flows ${flow_length} \
       --component_type ${component_type} \
       --num_components 1 \
       --batch_size ${batch_size} \
       --manual_seed ${seed} \
       --log_interval ${log} \
       --plot_interval ${plot} &


# 2 component
python main_experiment.py --dataset ${dataset} \
       --validation \
       --experiment_name ${exp_name} \
       --num_workers ${num_workers} \
       --epochs ${epochs} \
       --early_stopping_epochs ${early_stop} \
       --annealing_schedule ${annealing_schedule} \
       --burnin ${burnin} \
       --z_size ${z_size} \
       --flow boosted \
       --regularization_rate ${regularization} \
       --num_flows ${flow_length} \
       --component_type ${component_type} \
       --num_components 2 \
       --batch_size ${batch_size} \
       --manual_seed ${seed} \
       --log_interval ${log} \
       --plot_interval ${plot} &


# 3 component
python main_experiment.py --dataset ${dataset} \
       --validation \
       --experiment_name ${exp_name} \
       --num_workers ${num_workers} \
       --epochs ${epochs} \
       --early_stopping_epochs ${early_stop} \
       --annealing_schedule ${annealing_schedule} \
       --burnin ${burnin} \
       --z_size ${z_size} \
       --flow boosted \
       --regularization_rate ${regularization} \
       --num_flows ${flow_length} \
       --component_type ${component_type} \
       --num_components 3 \
       --batch_size ${batch_size} \
       --manual_seed ${seed} \
       --log_interval ${log} \
       --plot_interval ${plot} &


# 4 component
python main_experiment.py --dataset ${dataset} \
       --validation \
       --experiment_name ${exp_name} \
       --num_workers ${num_workers} \
       --epochs ${epochs} \
       --early_stopping_epochs ${early_stop} \
       --annealing_schedule ${annealing_schedule} \
       --burnin ${burnin} \
       --z_size ${z_size} \
       --flow boosted \
       --regularization_rate ${regularization} \
       --num_flows ${flow_length} \
       --component_type ${component_type} \
       --num_components 4 \
       --batch_size ${batch_size} \
       --manual_seed ${seed} \
       --log_interval ${log} \
       --plot_interval ${plot} &

# 5 component
python main_experiment.py --dataset ${dataset} \
       --validation \
       --experiment_name ${exp_name} \
       --num_workers ${num_workers} \
       --epochs ${epochs} \
       --early_stopping_epochs ${early_stop} \
       --annealing_schedule ${annealing_schedule} \
       --burnin ${burnin} \
       --z_size ${z_size} \
       --flow boosted \
       --regularization_rate ${regularization} \
       --num_flows ${flow_length} \
       --component_type ${component_type} \
       --num_components 5 \
       --batch_size ${batch_size} \
       --manual_seed ${seed} \
       --log_interval ${log} \
       --plot_interval ${plot} &


# 6 component
python main_experiment.py --dataset ${dataset} \
       --validation \
       --experiment_name ${exp_name} \
       --num_workers ${num_workers} \
       --epochs ${epochs} \
       --early_stopping_epochs ${early_stop} \
       --annealing_schedule ${annealing_schedule} \
       --burnin ${burnin} \
       --z_size ${z_size} \
       --flow boosted \
       --regularization_rate ${regularization} \
       --num_flows ${flow_length} \
       --component_type ${component_type} \
       --num_components 6 \
       --batch_size ${batch_size} \
       --manual_seed ${seed} \
       --log_interval ${log} \
       --plot_interval ${plot} ;
