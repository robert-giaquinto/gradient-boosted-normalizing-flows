# Load defaults for all experiments
source ./scripts/experiment_config_density.sh

# activate virtual environment
source ./venv/bin/activate

# variables specific to this experiment
num_steps=150001
exp_name=density_matching
logging=1000
iters_per_component=25000
regularization_rate=0.4
plot_resolution=250


for u in 1 2 3 4 0
do

    # boosted model
    python density.py --dataset u${u} \
           --experiment_name ${exp_name} \
           --no_cuda \
           --num_workers ${num_workers} \
           --plot_resolution ${plot_resolution} \
           --num_steps ${num_steps} \
           --no_annealing \
           --iters_per_component ${iters_per_component} \
           --flow boosted \
           --component_type ${component_type} \
           --num_components 2 \
           --regularization_rate ${regularization_rate} \
           --num_flows 1 \
           --z_size ${z_size} \
           --batch_size ${batch_size} \
           --manual_seed ${seed} \
           --log_interval ${logging} \
           --plot_interval ${iters_per_component} &

    # boosted model
    python density.py --dataset u${u} \
           --experiment_name ${exp_name} \
           --no_cuda \
           --num_workers ${num_workers} \
           --plot_resolution ${plot_resolution} \
           --num_steps ${num_steps} \
           --no_annealing \
           --iters_per_component ${iters_per_component} \
           --flow boosted \
           --component_type ${component_type} \
           --num_components 4 \
           --regularization_rate ${regularization_rate} \
           --num_flows 1 \
           --z_size ${z_size} \
           --batch_size ${batch_size} \
           --manual_seed ${seed} \
           --log_interval ${logging} \
           --plot_interval ${iters_per_component} ;

done


    
