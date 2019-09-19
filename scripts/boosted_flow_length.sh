# Hypothesis: A medium size of flow should be best
# Experiment: Run planar flows for varying flow lengths

# Load defaults for all experiments
source ./scripts/experiment_config.sh

# activate virtual environment
source ./venv/bin/activate

# define variable specific to this experiment
exp_name=flow_length
annealing_schedule=50
burnin=50
regularization=0.1
num_components=6
flow_length=2

# flow length: 1
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
       --num_flows 1 \
       --component_type ${component_type} \
       --num_components ${num_components} \
       --batch_size ${batch_size} \
       --manual_seed ${seed} \
       --log_interval ${log} \
       --plot_interval ${plot} ;

# flow length: 2
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
       --num_flows 2 \
       --component_type ${component_type} \
       --num_components ${num_components} \
       --batch_size ${batch_size} \
       --manual_seed ${seed} \
       --log_interval ${log} \
       --plot_interval ${plot} ;

# flow length: 4
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
       --num_flows 4 \
       --component_type ${component_type} \
       --num_components ${num_components} \
       --batch_size ${batch_size} \
       --manual_seed ${seed} \
       --log_interval ${log} \
       --plot_interval ${plot} ;

# flow length: 8
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
       --num_flows 8 \
       --component_type ${component_type} \
       --num_components ${num_components} \
       --batch_size ${batch_size} \
       --manual_seed ${seed} \
       --log_interval ${log} \
       --plot_interval ${plot} ;

# flow length: 16
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
       --num_flows 16 \
       --component_type ${component_type} \
       --num_components ${num_components} \
       --batch_size ${batch_size} \
       --manual_seed ${seed} \
       --log_interval ${log} \
       --plot_interval ${plot} ;
