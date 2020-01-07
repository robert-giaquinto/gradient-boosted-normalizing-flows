cd /export/scratch/robert/ensemble-normalizing-flows

# activate virtual environment
module unload soft/python
module load soft/python/anaconda
source /soft/python/anaconda/Linux_x86_64/etc/profile.d/conda.sh
conda activate env

# Load defaults for all experiments
source ./scripts/experiment_config_density.sh

# variables specific to this experiment
num_steps=200001
exp_name=boosted_density_sampling
logging=1000
iters_per_component=25000
plot_resolution=500

for dataset in 8gaussians swissroll moons pinwheel 2spirals checkerboard  circles
do

    for num_components in 2 4 #8
    do
        for regularization_rate in 0.7 0.8 0.9 1.0 1.1
        do
            # realnvp and iaf with various h_sizes
            for h_size in 64 128 #256
            do
                # realnvp
                for num_flows in 1 #2
                do
                    for network in relu #tanh
                    do
                        python -m density_experiment --dataset ${dataset} \
                               --experiment_name ${exp_name} \
                               --no_cuda \
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
                               --base_network ${network} \
                               --h_size ${h_size} \
                               --regularization_rate ${regularization_rate} \
                               --batch_size ${batch_size} \
                               --manual_seed ${manual_seed} \
                               --log_interval ${logging} \
                               --plot_resolution ${plot_resolution} \
                               --plot_interval ${iters_per_component} &
                    done
                done
            done
        done
            
        # affine (1 flow)
        python -m density_experiment --dataset ${dataset} \
               --experiment_name ${exp_name} \
               --no_cuda \
               --num_workers ${num_workers} \
               --num_steps ${num_steps} \
               --no_annealing \
               --learning_rate ${learning_rate} \
               --iters_per_component ${iters_per_component} \
               --flow boosted \
               --num_components ${num_components} \
               --num_flows 1 \
               --component_type affine \
               --regularization_rate 0.75 \
               --batch_size ${batch_size} \
               --manual_seed ${manual_seed} \
               --log_interval ${logging} \
               --plot_resolution ${plot_resolution} \
               --plot_interval ${iters_per_component} &
    done
    wait
    
done
wait
echo "Job complete"


    
