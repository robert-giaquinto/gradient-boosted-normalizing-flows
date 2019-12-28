# Load defaults for all experiments
source ./scripts/experiment_config.sh

# activate virtual environment
source ./venv/bin/activate

# define variable specific to this experiment
exp_name=num_components
epochs=600
vae_layers=linear

for num_components in 2 4 8
do
    for regularization_rate in 0.9 0.95 1.0 1.05 1.1 1.25
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
               --base_network relu \
               --h_size 128 \
               --num_components ${num_components} \
               --regularization_rate ${regularization_rate} \
               --num_flows 1 \
               --z_size ${z_size} \
               --batch_size ${batch_size} \
               --manual_seed ${manual_seed} \
               --plot_interval ${plotting} &
    done
done
wait
echo "Job complete"
