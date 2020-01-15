cd /export/scratch/robert/ensemble-normalizing-flows

# activate virtual environment
module unload soft/python
module load soft/python/anaconda
source /soft/python/anaconda/Linux_x86_64/etc/profile.d/conda.sh
conda activate env

# Load defaults for all experiments
source ./scripts/experiment_config.sh

# define variable specific to this experiment
experiment_name=num_components
vae_layers=linear
regularization_rate=1.0
annealing_schedule=100
epochs_per_component=400


for num_components in 2 4 8
do
    epochs=$((num_components * epochs_per_component))

    for dataset in mnist freyfaces omniglot caltech #cifar10
    do  
        for num_flows in 4 8 16
        do  
            for h_size in 256 #128 256 512
            do
                python main_experiment.py --dataset mnist \
                       --experiment_name ${experiment_name} \
                       --testing \
                       --no_cuda \
                       --num_workers 1 \
                       --no_lr_schedule \
                       --learning_rate ${learning_rate} \
                       --annealing_schedule ${annealing_schedule} \
                       --epochs_per_component ${epochs_per_component} \
                       --epochs ${epochs} \
                       --vae_layers ${vae_layers} \
                       --flow boosted \
                       --component_type realnvp \
                       --num_base_layers 0 \
                       --base_network tanh \
                       --h_size ${h_size} \
                       --num_components ${num_components} \
                       --regularization_rate ${regularization_rate} \
                       --num_flows ${num_flows} \
                       --z_size ${z_size} \
                       --batch_size ${batch_size} \
                       --manual_seed ${manual_seed} \
                       --plot_interval ${plotting} &
            done
        done
    done
    wait

done
wait
echo "Job complete"
