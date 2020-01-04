cd /export/scratch/robert/ensemble-normalizing-flows

# activate virtual environment
module unload soft/python
module load soft/python/anaconda
source /soft/python/anaconda/Linux_x86_64/etc/profile.d/conda.sh
conda activate env

# Load defaults for all experiments
source ./scripts/experiment_config_density.sh

# variables specific to this experiment
num_steps=150001
exp_name=baseline_density_matching
logging=1000
plotting=25000
plot_resolution=1000


for u in 1 2 3 4
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
                
                    python -m density_experiment --dataset u${u} \
                           --experiment_name ${exp_name} \
                           --no_cuda \
                           --num_workers ${num_workers} \
                           --num_steps ${num_steps} \
                           --learning_rate ${learning_rate} \
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
        
        # # iaf
        # for num_hidden_layers in 0 1 2
        # do
        #     python -m density_experiment --dataset u${u} \
        #            --experiment_name ${exp_name} \
        #            --no_cuda \
        #            --num_workers ${num_workers} \
        #            --num_steps ${num_steps} \
        #            --learning_rate ${learning_rate} \
        #            --flow iaf \
        #            --num_base_layers ${num_hidden_layers} \
        #            --num_flows 1 \
        #            --h_size ${h_size} \
        #            --batch_size ${batch_size} \
        #            --manual_seed ${manual_seed} \
        #            --log_interval ${logging} \
        #            --plot_resolution ${plot_resolution} \
        #            --plot_interval ${plotting} &
        # done
    done
    wait

    # # basic flows only need to tune num_flows
    # for flow in planar radial nlsq
    # do
    #     for flow_depth in 1 2 4 8 16
    #     do
    #         python -m density_experiment --dataset u${u} \
    #                --experiment_name ${exp_name} \
    #                --no_cuda \
    #                --num_steps ${num_steps} \
    #                --learning_rate ${learning_rate} \
    #                --num_workers ${num_workers} \
    #                --flow ${flow} \
    #                --num_flows ${flow_depth} \
    #                --batch_size ${batch_size} \
    #                --manual_seed ${manual_seed} \
    #                --log_interval ${logging} \
    #                --plot_resolution ${plot_resolution} \
    #                --plot_interval ${plotting} &
    #     done
    #     python -m density_experiment --dataset u${u} \
    #            --experiment_name ${exp_name} \
    #            --no_cuda \
    #            --num_steps ${num_steps} \
    #            --learning_rate ${learning_rate} \
    #            --num_workers ${num_workers} \
    #            --flow ${flow} \
    #            --num_flows 32 \
    #            --batch_size ${batch_size} \
    #            --manual_seed ${manual_seed} \
    #            --log_interval ${logging} \
    #            --plot_resolution ${plot_resolution} \
    #            --plot_interval ${plotting} &

    # done

    # # affine
    # python -m density_experiment --dataset u${u} \
    #        --experiment_name ${exp_name} \
    #        --no_cuda \
    #        --num_steps ${num_steps} \
    #        --learning_rate ${learning_rate} \
    #        --num_workers ${num_workers} \
    #        --flow affine \
    #        --num_flows 1 \
    #        --batch_size ${batch_size} \
    #        --manual_seed ${manual_seed} \
    #        --log_interval ${logging} \
    #        --plot_resolution ${plot_resolution} \
    #        --plot_interval ${plotting} &

done
wait
echo "Job complete"

    
