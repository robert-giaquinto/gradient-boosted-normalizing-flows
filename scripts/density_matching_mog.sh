# Load defaults for all experiments
source ./scripts/experiment_config_density.sh

# activate virtual environment
source ./venv/bin/activate

# variables specific to this experiment
num_steps=150001
exp_name=mog_density_matching
iters_per_component=25000
regularization_rate=0.4
plot_resolution=250


for sigma in 1.0 1.25 1.5
do

    for num_components in 2 4 6
    do

        # boosted model
        python density.py --dataset u5 \
               --mog_sigma ${sigma} \
               --mog_clusters 6 \
               --experiment_name ${exp_name} \
               --no_cuda \
               --num_workers ${num_workers} \
               --plot_resolution ${plot_resolution} \
               --num_steps ${num_steps} \
               --no_annealing \
               --iters_per_component ${iters_per_component} \
               --flow boosted \
               --component_type ${component_type} \
               --num_components ${num_components} \
               --regularization_rate ${regularization_rate} \
               --num_flows 1 \
               --z_size ${z_size} \
               --batch_size ${batch_size} \
               --manual_seed ${seed} \
               --log_interval 1000 \
               --plot_interval ${iters_per_component} &
    done

done
