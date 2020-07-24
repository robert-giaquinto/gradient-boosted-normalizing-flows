cd /export/scratch/robert/ensemble-normalizing-flows

# activate virtual environment
module unload soft/python
module load soft/python/anaconda
source /soft/python/anaconda/Linux_x86_64/etc/profile.d/conda.sh
conda activate env

source ./scripts/experiment_config_density.sh

# variables specific to this experiment
num_steps=200001
exp_name=boosted_density_matching
iters_per_component=25000
logging=1000


for u in 1 2 3 4
do
    for num_components in 2 4
    do
        for regularization_rate in 0.6 0.8 0.9 1.0 1.1 1.2
        do
            # realnvp and iaf with various h_sizes
            for h_size in 64 128 256
            do
                # realnvp
                for num_flows in 1 2
                do
                    for network in tanh relu
                    do
                        for layers in 1 2
                        do
                            python -m toy_experiment --dataset u${u} \
                                   --experiment_name ${exp_name} \
                                   --no_cuda \
                                   --num_workers ${num_workers} \
                                   --num_steps ${num_steps} \
                                   --learning_rate ${learning_rate} \
                                   --flow boosted \
                                   --iters_per_component ${iters_per_component} \
                                   --num_components ${num_components} \
                                   --num_flows ${num_flows} \
                                   --component_type realnvp \
                                   --num_base_layers ${layers} \
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

            # for flow_depth in 1 2 4 8 16 32
            # do
            #     # non-linear squared flow
            #     python -m density_experiment --dataset u${u} \
            #            --experiment_name ${exp_name} \
            #            --no_cuda \
            #            --num_workers ${num_workers} \
            #            --num_steps ${num_steps} \
            #            --learning_rate 0.0001 \
            #            --no_lr_schedule \
            #            --flow boosted \
            #            --iters_per_component ${iters_per_component} \
            #            --regularization_rate ${regularization_rate} \
            #            --num_components ${num_components} \
            #            --component_type nlsq \
            #            --num_flows ${flow_depth} \
            #            --batch_size ${batch_size} \
            #            --manual_seed ${manual_seed} \
            #            --log_interval ${logging} \
            #            --plot_resolution ${plot_resolution} \
            #            --plot_interval ${iters_per_component} &
            # done
            
            # # affine
            # python -m density_experiment --dataset u${u} \
            #        --experiment_name ${exp_name} \
            #        --no_cuda \
            #        --num_workers ${num_workers} \
            #        --num_steps ${num_steps} \
            #        --learning_rate ${learning_rate} \
            #        --flow boosted \
            #        --iters_per_component ${iters_per_component} \
            #        --regularization_rate ${regularization_rate} \
            #        --num_components ${num_components} \
            #        --component_type affine \
            #        --num_flows 1 \
            #        --batch_size ${batch_size} \
            #        --manual_seed ${manual_seed} \
            #        --log_interval ${logging} \
            #        --plot_resolution ${plot_resolution} \
            #        --plot_interval ${iters_per_component} ;
        done
    done
done
wait
echo "Job complete"

