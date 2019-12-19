source ./scripts/experiment_config_density.sh

# activate virtual environment
source ./venv/bin/activate

# variables specific to this experiment
num_steps=400001
exp_name=bboosted_density_matching
iters_per_component=25000
logging=1000


for u in 1 2 3 4
do
    for num_components in 2 4 8
    do
        for regularization_rate in 0.3 0.4 0.50
        do
            # boosted RealNVPS
            for h_size in 8 32 128
            do
                for num_flows in 1 2
                do
                    python density.py --dataset u${u} \
                           --experiment_name ${exp_name} \
                           --no_cuda \
                           --num_workers ${num_workers} \
                           --num_steps ${num_steps} \
                           --learning_rate ${learning_rate} \
                           --no_lr_schedule \
                           --no_annealing \
                           --flow boosted \
                           --iters_per_component ${iters_per_component} \
                           --num_components ${num_components} \
                           --num_flows ${num_flows} \
                           --component_type realnvp \
                           --num_base_layers 1 \
                           --base_network relu \
                           --h_size ${h_size} \
                           --regularization_rate ${regularization_rate} \
                           --batch_size ${batch_size} \
                           --manual_seed ${manual_seed} \
                           --log_interval ${logging} \
                           --plot_resolution ${plot_resolution} \
                           --plot_interval ${iters_per_component} &
                    
                done
            done

            for flow_depth in 1 2 4 8 16 32
            do
                # non-linear squared flow
                python density.py --dataset u${u} \
                       --experiment_name ${exp_name} \
                       --no_cuda \
                       --num_workers ${num_workers} \
                       --num_steps ${num_steps} \
                       --learning_rate ${learning_rate} \
                       --no_lr_schedule \
                       --no_annealing \
                       --flow boosted \
                       --iters_per_component ${iters_per_component} \
                       --regularization_rate ${regularization_rate} \
                       --num_components ${num_components} \
                       --component_type nlsq \
                       --num_flows ${flow_depth} \
                       --batch_size ${batch_size} \
                       --manual_seed ${manual_seed} \
                       --log_interval ${logging} \
                       --plot_resolution ${plot_resolution} \
                       --plot_interval ${iters_per_component} &
            done
            # affine
            python density.py --dataset u${u} \
                   --experiment_name ${exp_name} \
                   --no_cuda \
                   --num_workers ${num_workers} \
                   --num_steps ${num_steps} \
                   --learning_rate ${learning_rate} \
                   --no_lr_schedule \
                   --no_annealing \
                   --flow boosted \
                   --iters_per_component ${iters_per_component} \
                   --regularization_rate ${regularization_rate} \
                   --num_components ${num_components} \
                   --component_type affine \
                   --num_flows 1 \
                   --batch_size ${batch_size} \
                   --manual_seed ${manual_seed} \
                   --log_interval ${logging} \
                   --plot_resolution ${plot_resolution} \
                   --plot_interval ${iters_per_component} ;
        done
    done
done

