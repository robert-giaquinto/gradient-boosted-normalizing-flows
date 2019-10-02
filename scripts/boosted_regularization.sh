# Hypothesis: More components should lower the loss
# Experiment: Run planar flows for varying numbers of components

# Load defaults for all experiments
source ./scripts/experiment_config.sh

# activate virtual environment
source ./venv/bin/activate

# define variable specific to this experiment
exp_name=regularization
num_components=6
flow_length=2


# regularization = 0.01
python main_experiment.py --dataset ${dataset} \
       --validation \
       --experiment_name ${exp_name}1 \
       --num_workers ${num_workers} \
       --epochs ${epochs} \
       --early_stopping_epochs ${early_stop} \
       --annealing_schedule ${annealing_schedule} \
       --burnin ${burnin} \
       --z_size ${z_size} \
       --flow boosted \
       --regularization_rate 0.01 \
       --num_flows ${flow_length} \
       --component_type ${component_type} \
       --num_components ${num_components} \
       --batch_size ${batch_size} \
       --manual_seed ${seed} \
       --log_interval ${log} \
       --plot_interval ${plot} &

# regularization = 0.1
python main_experiment.py --dataset ${dataset} \
       --validation \
       --experiment_name ${exp_name}10 \
       --num_workers ${num_workers} \
       --epochs ${epochs} \
       --early_stopping_epochs ${early_stop} \
       --annealing_schedule ${annealing_schedule} \
       --burnin ${burnin} \
       --z_size ${z_size} \
       --flow boosted \
       --regularization_rate 0.1 \
       --num_flows ${flow_length} \
       --component_type ${component_type} \
       --num_components ${num_components} \
       --batch_size ${batch_size} \
       --manual_seed ${seed} \
       --log_interval ${log} \
       --plot_interval ${plot} &

# regularization = 0.25
python main_experiment.py --dataset ${dataset} \
       --validation \
       --experiment_name ${exp_name}25 \
       --num_workers ${num_workers} \
       --epochs ${epochs} \
       --early_stopping_epochs ${early_stop} \
       --annealing_schedule ${annealing_schedule} \
       --burnin ${burnin} \
       --z_size ${z_size} \
       --flow boosted \
       --regularization_rate 0.25 \
       --num_flows ${flow_length} \
       --component_type ${component_type} \
       --num_components ${num_components} \
       --batch_size ${batch_size} \
       --manual_seed ${seed} \
       --log_interval ${log} \
       --plot_interval ${plot} &

# regularization = 0.5
python main_experiment.py --dataset ${dataset} \
       --validation \
       --experiment_name ${exp_name}50 \
       --num_workers ${num_workers} \
       --epochs ${epochs} \
       --early_stopping_epochs ${early_stop} \
       --annealing_schedule ${annealing_schedule} \
       --burnin ${burnin} \
       --z_size ${z_size} \
       --flow boosted \
       --regularization_rate 0.5 \
       --num_flows ${flow_length} \
       --component_type ${component_type} \
       --num_components ${num_components} \
       --batch_size ${batch_size} \
       --manual_seed ${seed} \
       --log_interval ${log} \
       --plot_interval ${plot} &

# regularization = 1.0
python main_experiment.py --dataset ${dataset} \
       --validation \
       --experiment_name ${exp_name}100 \
       --num_workers ${num_workers} \
       --epochs ${epochs} \
       --early_stopping_epochs ${early_stop} \
       --annealing_schedule ${annealing_schedule} \
       --burnin ${burnin} \
       --z_size ${z_size} \
       --flow boosted \
       --regularization_rate 1.0 \
       --num_flows ${flow_length} \
       --component_type ${component_type} \
       --num_components ${num_components} \
       --batch_size ${batch_size} \
       --manual_seed ${seed} \
       --log_interval ${log} \
       --plot_interval ${plot} &

# regularization = 2.0
python main_experiment.py --dataset ${dataset} \
       --validation \
       --experiment_name ${exp_name}200 \
       --num_workers ${num_workers} \
       --epochs ${epochs} \
       --early_stopping_epochs ${early_stop} \
       --annealing_schedule ${annealing_schedule} \
       --burnin ${burnin} \
       --z_size ${z_size} \
       --flow boosted \
       --regularization_rate 2.0 \
       --num_flows ${flow_length} \
       --component_type ${component_type} \
       --num_components ${num_components} \
       --batch_size ${batch_size} \
       --manual_seed ${seed} \
       --log_interval ${log} \
       --plot_interval ${plot} ;

