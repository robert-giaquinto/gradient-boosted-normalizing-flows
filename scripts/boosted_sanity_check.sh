# activate virtual environment
source ./venv/bin/activate

# variables specific to this experiment
experiment_name=sanity
num_components=1
num_flows=2
dataset=mnist
num_workers=1
epochs=150
early_stopping_epochs=0
annealing_schedule=100
z_size=64
component_type=planar
batch_size=16
seed=123
log=0
plot=5
min_beta=1.0
max_beta=1.0

python main_experiment.py --dataset ${dataset} \
       --validation \
       --experiment_name ${experiment_name} \
       --num_workers ${num_workers} \
       --epochs ${epochs} \
       --early_stopping_epochs ${early_stopping_epochs} \
       --annealing_schedule ${annealing_schedule} \
       --burnin 0 \
       --min_beta ${min_beta} \
       --max_beta ${max_beta} \
       --z_size ${z_size} \
       --flow boosted \
       --regularization_rate 0.0 \
       --num_flows ${num_flows} \
       --component_type ${component_type} \
       --num_components ${num_components} \
       --batch_size ${batch_size} \
       --manual_seed ${seed} \
       --log_interval ${log} \
       --plot_interval ${plot} ;

#python main_experiment.py --dataset ${dataset} \
#       --validation \
#       --experiment_name ${experiment_name} \
#       --num_workers ${num_workers} \
#       --epochs ${epochs} \
#       --early_stopping_epochs ${early_stopping_epochs} \
#       --annealing_schedule ${annealing_schedule} \
#       --min_beta ${min_beta} \
#       --max_beta ${max_beta} \
#       --z_size ${z_size} \
#       --flow ${component_type} \
#       --num_flows ${num_flows} \
#       --num_components ${num_components} \
#       --batch_size ${batch_size} \
#       --manual_seed ${seed} \
#       --log_interval ${log} \
#       --plot_interval ${plot} ;

deactivate
