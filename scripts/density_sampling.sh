cd /export/scratch/robert/ensemble-normalizing-flows

# activate virtual environment
module unload soft/python
module load soft/python/anaconda
source /soft/python/anaconda/Linux_x86_64/etc/profile.d/conda.sh
conda activate env

# Load defaults for all experiments
source ./scripts/experiment_config_density.sh

# variables specific to this experiment
num_steps=100001
exp_name=baseline_density_sampling
logging=1000
plotting=25000
plot_resolution=500


for dataset in 8gaussians 2gaussians 1gaussian swissroll rings moons pinwheel cos 2spirals checkerboard line circles joint_gaussian
do

    # realnvp and iaf with various h_sizes
    for h_size in 8 16 32 64 128
    do
        # realnvp
        for num_flows in 1 2
        do
            python -m density_experiment --dataset ${dataset} \
                   --experiment_name ${exp_name} \
                   --no_annealing \
                   --no_cuda \
                   --num_workers ${num_workers} \
                   --num_steps ${num_steps} \
                   --learning_rate ${learning_rate} \
                   --num_workers ${num_workers} \
                   --flow realnvp \
                   --num_flows ${num_flows} \
                   --num_base_layers 1 \
                   --base_network relu \
                   --h_size ${h_size} \
                   --z_size ${z_size} \
                   --batch_size ${batch_size} \
                   --manual_seed ${seed} \
                   --log_interval ${logging} \
                   --plot_resolution ${plot_resolution} \
                   --plot_interval ${plotting} &
        done
        # iaf
        python -m density_experiment --dataset ${dataset} \
               --experiment_name ${exp_name} \
               --no_annealing \
               --no_cuda \
               --num_workers ${num_workers} \
               --num_steps ${num_steps} \
               --learning_rate ${learning_rate} \
               --flow iaf \
               --num_flows 1 \
               --h_size ${h_size} \
               --z_size ${z_size} \
               --batch_size ${batch_size} \
               --manual_seed ${seed} \
               --log_interval ${logging} \
               --plot_resolution ${plot_resolution} \
               --plot_interval ${plotting} ;
    done

    # basic flows only need to tune num_flows
    for flow in planar radial affine nlsq
    do

        # run each basic flow in parallel, 1 job for each "num_flows"
        for num_flows in 1 2 4 8 16
        do

            python -m density_experiment --dataset ${dataset} \
                   --experiment_name ${exp_name} \
                   --no_annealing \
                   --no_cuda \
                   --num_workers ${num_workers} \
                   --num_steps ${num_steps} \
                   --learning_rate ${learning_rate} \
                   --flow ${flow} \
                   --num_flows ${num_flows} \
                   --z_size ${z_size} \
                   --batch_size ${batch_size} \
                   --manual_seed ${seed} \
                   --log_interval ${logging} \
                   --plot_resolution ${plot_resolution} \
                   --plot_interval ${plotting} &
        done
        python -m density_experiment --dataset ${dataset} \
               --experiment_name ${exp_name} \
               --no_annealing \
               --no_cuda \
               --num_workers ${num_workers} \
               --num_steps ${num_steps} \
               --learning_rate ${learning_rate} \
               --flow ${flow} \
               --num_flows 32 \
               --z_size ${z_size} \
               --batch_size ${batch_size} \
               --manual_seed ${seed} \
               --log_interval ${logging} \
               --plot_resolution ${plot_resolution} \
               --plot_interval ${plotting} ;

    done

done

