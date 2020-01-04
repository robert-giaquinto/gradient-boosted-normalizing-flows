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
experiment_name=baseline_density_sampling
plotting=25000


for dataset in 8gaussians swissroll rings moons pinwheel cos 2spirals checkerboard line circles joint_gaussian
do
    # realnvp and iaf
    for h_size in 64 128 256
    do
        for num_flows in 1 2
        do
            for layers in 1 2
            do
                for network in tanh relu
                do
                    python -m density_experiment --dataset ${dataset} \
                           --experiment_name ${experiment_name} \
                           --no_annealing \
                           --no_cuda \
                           --num_workers ${num_workers} \
                           --num_steps ${num_steps} \
                           --learning_rate ${learning_rate} \
                           --num_workers ${num_workers} \
                           --flow realnvp \
                           --num_flows ${num_flows} \
                           --num_base_layers ${layers} \
                           --base_network ${network} \
                           --h_size ${h_size} \
                           --batch_size ${batch_size} \
                           --manual_seed ${manual_seed} \
                           --log_interval ${logging} \
                           --plot_resolution ${plot_resolution} \
                           --plot_interval ${plotting} &
                done
            done
        done
    done

    # # run each basic flows
    # for num_flows in 1 2 4 8 16 32
    # do
    #     # basic flows: planar and radial
    #     for flow in planar radial
    #     do

    #         python -m density_experiment --dataset ${dataset} \
    #                --experiment_name ${experiment_name} \
    #                --no_annealing \
    #                --no_cuda \
    #                --num_workers ${num_workers} \
    #                --num_steps ${num_steps} \
    #                --learning_rate ${learning_rate} \
    #                --flow ${flow} \
    #                --num_flows ${num_flows} \
    #                --z_size ${z_size} \
    #                --batch_size ${batch_size} \
    #                --manual_seed ${seed} \
    #                --log_interval ${logging} \
    #                --plot_resolution ${plot_resolution} \
    #                --plot_interval ${plotting} &
    #     done
    #     # run non-linear squared flow with a lower learning rate for safety
    #     python -m density_experiment --dataset ${dataset} \
    #            --experiment_name ${experiment_name} \
    #            --no_annealing \
    #            --no_cuda \
    #            --num_workers ${num_workers} \
    #            --num_steps ${num_steps} \
    #            --learning_rate 0.0001 \
    #            --flow ${flow} \
    #            --num_flows ${num_flows} \
    #            --z_size ${z_size} \
    #            --batch_size ${batch_size} \
    #            --manual_seed ${seed} \
    #            --log_interval ${logging} \
    #            --plot_resolution ${plot_resolution} \
    #            --plot_interval ${plotting} &
    # done
done
wait
echo "Job complete"

