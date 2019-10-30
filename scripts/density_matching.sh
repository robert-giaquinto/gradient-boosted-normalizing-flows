# Load defaults for all experiments
source ./scripts/experiment_config_density.sh

# activate virtual environment
source ./venv/bin/activate

# variables specific to this experiment
num_steps=100000
exp_name=density_matching
logging=1000
iters_per_component=20000


for u in 0 1 2 3 4
do

    for flow_depth in 2 4 8
    do

        for regularization_rate in  0.5 1.0 1.5
        do
            python density.py --dataset u${u} \
                   --experiment_name ${exp_name} \
                   --no_cuda \
                   --num_workers ${num_workers} \
                   --num_steps ${num_steps} \
                   --min_beta 1.0 \
                   --iters_per_component ${iters_per_component} \
                   --flow boosted \
                   --component_type ${component_type} \
                   --num_components 2 \
                   --regularization_rate ${regularization_rate} \
                   --num_flows ${flow_depth} \
                   --z_size ${z_size} \
                   --batch_size ${batch_size} \
                   --manual_seed ${seed} \
                   --log_interval ${logging} \
                   --plot_interval ${logging} &
        done

        python density.py --dataset u${u} \
               --experiment_name ${exp_name} \
               --no_cuda \
               --num_steps ${num_steps} \
               --min_beta 1.0 \
               --iters_per_component ${iters_per_component} \
               --num_workers ${num_workers} \
               --flow ${component_type} \
               --num_flows ${flow_depth} \
               --z_size ${z_size} \
               --batch_size ${batch_size} \
               --manual_seed ${seed} \
               --log_interval ${logging} \
               --plot_interval ${logging} ;


    done

done


    
