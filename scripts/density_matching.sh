# Load defaults for all experiments
source ./scripts/experiment_config_density.sh

# activate virtual environment
source ./venv/bin/activate

# variables specific to this experiment
num_steps=150001
exp_name=baseline_density_matching
logging=1000
iters_per_component=50000
regularization_rate=0.4
plot_resolution=250
learning_rate=0.005


for u in 1 2 3 4
do

    for flow_depth in 1 2 4 8 16 32
    do
        # planar flow
        python density.py --dataset u${u} \
               --experiment_name ${exp_name} \
               --no_cuda \
               --num_steps ${num_steps} \
               --plot_resolution ${plot_resolution} \
               --learning_rate ${learning_rate} \
               --no_annealing \
               --num_workers ${num_workers} \
               --flow planar \
               --num_flows ${flow_depth} \
               --batch_size ${batch_size} \
               --z_size ${z_size} \
               --manual_seed ${seed} \
               --log_interval ${logging} \
               --plot_interval ${iters_per_component} &

        # radial flow
        python density.py --dataset u${u} \
               --experiment_name ${exp_name} \
               --no_cuda \
               --num_steps ${num_steps} \
               --plot_resolution ${plot_resolution} \
               --learning_rate ${learning_rate} \
               --no_annealing \
               --num_workers ${num_workers} \
               --flow radial \
               --num_flows ${flow_depth} \
               --batch_size ${batch_size} \
               --z_size ${z_size} \
               --manual_seed ${seed} \
               --log_interval ${logging} \
               --plot_interval ${iters_per_component} &

        # non-linear squared flow
        python density.py --dataset u${u} \
               --experiment_name ${exp_name} \
               --no_cuda \
               --num_steps ${num_steps} \
               --plot_resolution ${plot_resolution} \
               --learning_rate ${learning_rate} \
               --no_annealing \
               --num_workers ${num_workers} \
               --flow nlsq \
               --num_flows ${flow_depth} \
               --batch_size ${batch_size} \
               --z_size ${z_size} \
               --manual_seed ${seed} \
               --log_interval ${logging} \
               --plot_interval ${iters_per_component} ;

    done

    # affine
    python density.py --dataset ${dataset} \
       --experiment_name ${exp_name} \
       --no_cuda \
       --num_steps ${num_steps} \
       --plot_resolution ${plot_resolution} \
       --learning_rate ${learning_rate} \
       --no_annealing \
       --num_workers ${num_workers} \
       --flow affine \
       --num_flows 1 \
       --z_size ${z_size} \
       --batch_size ${batch_size} \
       --manual_seed ${seed} \
       --log_interval ${logging} \
       --plot_interval ${iters_per_component} ;

    

done


    
