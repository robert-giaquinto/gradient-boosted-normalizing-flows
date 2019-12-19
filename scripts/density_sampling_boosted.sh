# Load defaults for all experiments
source ./scripts/experiment_config_density.sh

# activate virtual environment
source ./venv/bin/activate

# variables specific to this experiment
num_steps=400001
exp_name=boosted_density_sampling
logging=1000
iters_per_component=25000
plot_resolution=500

for dataset in 8gaussians 2gaussians swissroll rings moons pinwheel 2spirals checkerboard line circles joint_gaussian
do
    for regularization_rate in 0.01 0.05 0.25
    do
        for num_components in 2 4 8
        do

            # boosted RealNVPS
            for h_size in 8 64
            do
                
                for num_flows in 1 2
                do
                    python density.py --dataset ${dataset} \
                           --experiment_name ${exp_name} \
                           --no_cuda \
                           --no_lr_schedule \
                           --num_workers ${num_workers} \
                           --num_steps ${num_steps} \
                           --no_annealing \
                           --learning_rate ${learning_rate} \
                           --iters_per_component ${iters_per_component} \
                           --flow boosted \
                           --num_components ${num_components} \
                           --num_flows ${num_flows} \
                           --component_type realnvp \
                           --num_base_layers 1 \
                           --base_network relu \
                           --h_size ${h_size} \
                           --regularization_rate ${regularization_rate} \
                           --batch_size ${batch_size} \
                           --manual_seed ${seed} \
                           --log_interval ${logging} \
                           --plot_resolution ${plot_resolution} \
                           --plot_interval ${iters_per_component} ;
                    
                done
            done
            
            # affine (1 flow)
            python density.py --dataset ${dataset} \
                   --experiment_name ${exp_name} \
                   --no_cuda \
                   --no_lr_schedule \
                   --num_workers ${num_workers} \
                   --num_steps ${num_steps} \
                   --no_annealing \
                   --learning_rate ${learning_rate} \
                   --iters_per_component ${iters_per_component} \
                   --flow boosted \
                   --num_components ${num_components} \
                   --num_flows 1 \
                   --component_type affine \
                   --regularization_rate ${regularization_rate} \
                   --batch_size ${batch_size} \
                   --manual_seed ${seed} \
                   --log_interval ${logging} \
                   --plot_resolution ${plot_resolution} \
                   --plot_interval ${iters_per_component} ;
        done
    done
done


    
