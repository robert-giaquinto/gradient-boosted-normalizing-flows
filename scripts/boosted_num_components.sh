cd /export/scratch/robert/ensemble-normalizing-flows

# activate virtual environment
module unload soft/python
module load soft/python/anaconda
source /soft/python/anaconda/Linux_x86_64/etc/profile.d/conda.sh
conda activate env

# Load defaults for all experiments
source ./scripts/experiment_config.sh

# define variable specific to this experiment
exp_name=num_components
epochs=600
vae_layers=linear
regularization_rate=1.0

for num_components in 2 4 8
do
    for num_flows in 1 2
    do
        for h_size in 64 128 256
        do
            for base_network in tanh relu
            do
                python main_experiment.py --dataset mnist \
                       --experiment_name ${experiment_name} \
                       --validation \
                       --no_cuda \
                       --num_workers ${num_workers} \
                       --no_lr_schedule \
                       --learning_rate 0.0005 \
                       --epochs ${epochs} \
                       --early_stopping_epochs 0 \
                       --burnin 0 \
                       --annealing_schedule 100 \
                       --vae_layers ${vae_layers} \
                       --flow boosted \
                       --component_type realnvp \
                       --num_base_layers 1 \
                       --base_network ${base_network} \
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
done
wait
echo "Job complete"
