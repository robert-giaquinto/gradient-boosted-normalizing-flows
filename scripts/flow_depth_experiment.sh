# Hypothesis: ensemble flows are relatively stronger for deeper flows
# Experiment: Run planar flows for varying flow lengths as baseline
# 			  Run ensemble flow as comparison


# Load defaults for all experiments
source ./scripts/experiment_config.sh

# activate virtual environment
source ./venv/bin/activate

# define variable specific to this experiment
num_components=8
exp_name=flow_depth

for flow_depth in 1 4 8 16; do
    # planar flow baseline
    python main_experiment.py --dataset ${dataset} \
           --validation \
           --batch_size ${batch_size} \
           --annealing_schedule ${annealing_schedule} \
           --epochs ${epochs} \
           --early_stopping_epochs ${early_stop} \
           --z_size ${z_size} \
           --flow planar \
           --num_flows ${flow_depth} \
           --manual_seed ${seed} \
           --exp_name ${exp_name} \
           --log_interval ${log_int} \
           --plot_interval ${plot_int} ;

    # ensemble flow: boosting
    python main_experiment.py --dataset ${dataset} \
           --validation \
           --batch_size ${batch_size} \
           --annealing_schedule ${annealing_schedule} \
           --epochs ${epochs} \
           --early_stopping_epochs ${early_stop} \
           --z_size ${z_size} \
           --flow boosted \
           --num_components ${num_components} \
           --component_type ${component_type} \
           --num_flows ${flow_depth} \
           --manual_seed ${seed} \
           --exp_name ${exp_name} \
           --log_interval ${log} \
           --plot_interval ${plot} ;

    # ensemble flow: bagging
    python main_experiment.py --dataset ${dataset} \
           --validation \
           --batch_size ${batch_size} \
           --annealing_schedule ${annealing_schedule} \
           --epochs ${epochs} \
           --early_stopping_epochs ${early_stop} \
           --z_size ${z_size} \
           --flow bagged \
           --num_components ${num_components} \
           --component_type ${component_type} \
           --num_flows ${flow_depth} \
           --manual_seed ${seed} \
           --exp_name ${exp_name} \
           --log_interval ${log} \
           --plot_interval ${plot} ;

done

# run no flow VAE for additional baseline
python main_experiment.py --dataset ${dataset} \
       --validation \
       --batch_size ${batch_size} \
       --annealing_schedule ${annealing_schedule} \
       --epochs ${epochs} \
       --early_stopping_epochs ${early_stop} \
       --z_size ${z_size} \
       --flow no_flow \
       --manual_seed ${seed} \
       --exp_name ${exp_name} \
       --log_interval ${log_int} \
       --plot_interval ${plot_int} ;
