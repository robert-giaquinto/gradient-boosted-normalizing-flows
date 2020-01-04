cd /export/scratch/robert/ensemble-normalizing-flows

# activate virtual environment
module unload soft/python
module load soft/python/anaconda
source /soft/python/anaconda/Linux_x86_64/etc/profile.d/conda.sh
conda activate env


# Load defaults for all experiments
source ./scripts/experiment_config.sh

# define variable specific to this experiment
exp_name=flow_length
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
       --plot_interval ${plot} &

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
       --plot_interval ${plot} &

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
       --plot_interval ${plot} &

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
       --plot_interval ${plot} &

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
