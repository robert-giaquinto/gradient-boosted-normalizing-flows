# Load defaults for all experiments
source ./scripts/experiment_config_density.sh

# activate virtual environment
source ./venv/bin/activate

# variables specific to this experiment
exp_name=too_much_tuna
dataset=u1
iters_per_component=50000
num_steps=1200001
resolution=250
logging=1000


for reg in 0.2 0.3 0.4 0.5
do
    for num_components in 4 8
    do
        
        python density.py --dataset ${dataset} \
               --experiment_name ${exp_name} \
               --no_cuda \
               --no_annealing \
               --num_workers ${num_workers} \
               --plot_resolution ${resolution} \
               --num_steps ${num_steps} \
               --iters_per_component ${iters_per_component} \
               --learning_rate 0.005 \
               --flow boosted \
               --component_type affine \
               --num_components ${num_components} \
               --regularization_rate ${reg} \
               --num_flows 1 \
               --z_size ${z_size} \
               --batch_size ${batch_size} \
               --manual_seed ${seed} \
               --log_interval ${logging} \
               --plot_interval ${iters_per_component} &

    done

    # show for a two components too
    python density.py --dataset ${dataset} \
           --experiment_name ${exp_name} \
           --no_cuda \
           --no_annealing \
           --num_workers ${num_workers} \
           --plot_resolution ${resolution} \
           --num_steps ${num_steps} \
           --iters_per_component ${iters_per_component} \
           --learning_rate 0.005 \
           --flow boosted \
           --component_type affine \
           --num_components 2 \
           --regularization_rate ${reg} \
           --num_flows 1 \
           --z_size ${z_size} \
           --batch_size ${batch_size} \
           --manual_seed ${seed} \
           --log_interval ${logging} \
           --plot_interval ${iters_per_component} ;
    

done
