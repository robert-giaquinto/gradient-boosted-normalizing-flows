# Load defaults for all experiments
source ./scripts/experiment_config_density.sh

# activate virtual environment
source ./venv/bin/activate

# variables specific to this experiment
num_steps=150001
exp_name=mog_density_matching
iters_per_component=25000
min_beta=1.0
regularization_rate=0.75
plot_resolution=500


for sigma in 1.0 1.25 1.5
do

    for flow_depth in 4 8 16
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
               --min_beta ${min_beta} \
               --iters_per_component ${iters_per_component} \
               --flow boosted \
               --component_type ${component_type} \
               --num_components 2 \
               --regularization_rate ${regularization_rate} \
               --num_flows ${flow_depth} \
               --z_size ${z_size} \
               --batch_size ${batch_size} \
               --manual_seed ${seed} \
               --log_interval ${iters_per_component} \
               --plot_interval ${iters_per_component} &

        # planar flow
        python density.py --dataset u5 \
               --mog_sigma ${sigma} \
               --mog_clusters 6 \
               --experiment_name ${exp_name} \
               --no_cuda \
               --num_steps ${num_steps} \
               --plot_resolution ${plot_resolution} \
               --min_beta ${min_beta} \
               --iters_per_component ${iters_per_component} \
               --num_workers ${num_workers} \
               --flow ${component_type} \
               --num_flows ${flow_depth} \
               --z_size ${z_size} \
               --batch_size ${batch_size} \
               --manual_seed ${seed} \
               --log_interval ${iters_per_component} \
               --plot_interval ${iters_per_component} ;


    done

done
