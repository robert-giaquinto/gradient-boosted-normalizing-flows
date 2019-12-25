# Load defaults for all experiments
source ./scripts/experiment_config.sh

# activate virtual environment
source ./venv/bin/activate

# define variable specific to this experiment
exp_name=regularization
num_components=2
epochs=600
lr=0.0005

for reg in 0.1 0.25 0.5 0.75 0.9 1.0 1.25
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
           --flow boosted \
           --component_type realnvp \
           --num_base_layers 1 \
           --base_network relu \
           --h_size 128 \
           --num_components ${num_components} \
           --regularization_rate ${reg} \
           --num_flows 1 \
           --z_size ${z_size} \
           --batch_size ${batch_size} \
           --manual_seed ${manual_seed} \
           --plot_interval ${plotting} &

    python main_experiment.py --dataset mnist \
           --experiment_name ${experiment_name} \
           --validation \
           --no_cuda \
           --num_workers ${num_workers} \
           --no_lr_schedule \
           --learning_rate 0.001 \
           --epochs ${epochs} \
           --early_stopping_epochs 0 \
           --burnin 0 \
           --annealing_schedule 100 \
           --flow boosted \
           --component_type realnvp \
           --num_base_layers 1 \
           --base_network relu \
           --h_size 128 \
           --num_components ${num_components} \
           --regularization_rate ${reg} \
           --num_flows 1 \
           --z_size ${z_size} \
           --batch_size ${batch_size} \
           --manual_seed ${manual_seed} \
           --plot_interval ${plotting} &

done

