# Load defaults for all experiments
source ./scripts/experiment_config_density.sh

# activate virtual environment
source ./venv/bin/activate

# variables specific to this experiment
num_steps=150001
exp_name=baseline_density_matching
logging=1000
plotting=25000
plot_resolution=500


for u in 1 2 3 4
do

    # realnvp and iaf with various h_sizes
    for h_size in 16 32 64 128
    do
        # realnvp
        for num_flows in 1 2
        do
            python density.py --dataset u${u} \
                   --experiment_name ${exp_name} \
                   --no_cuda \
                   --num_workers ${num_workers} \
                   --num_steps ${num_steps} \
                   --learning_rate ${learning_rate} \
                   --no_annealing \
                   --no_lr_schedule \
                   --flow realnvp \
                   --num_flows ${num_flows} \
                   --num_base_layers 1 \
                   --base_network relu \
                   --h_size ${h_size} \
                   --batch_size ${batch_size} \
                   --manual_seed ${manual_seed} \
                   --log_interval ${logging} \
                   --plot_resolution ${plot_resolution} \
                   --plot_interval ${plotting} &
        done
        
        # iaf
        for num_hidden_layers in 0 1 2
        do
            python density.py --dataset u${u} \
                   --experiment_name ${exp_name} \
                   --no_cuda \
                   --num_workers ${num_workers} \
                   --num_steps ${num_steps} \
                   --learning_rate ${learning_rate} \
                   --no_annealing \
                   --no_lr_schedule \
                   --flow iaf \
                   --num_base_layers ${num_hidden_layers} \
                   --num_flows 1 \
                   --h_size ${h_size} \
                   --batch_size ${batch_size} \
                   --manual_seed ${manual_seed} \
                   --log_interval ${logging} \
                   --plot_resolution ${plot_resolution} \
                   --plot_interval ${plotting} ;
        done
    done

    # basic flows only need to tune num_flows
    for flow in planar radial nlsq
    do
        for flow_depth in 1 2 4 8 16
        do
            python density.py --dataset u${u} \
                   --experiment_name ${exp_name} \
                   --no_cuda \
                   --num_steps ${num_steps} \
                   --learning_rate ${learning_rate} \
                   --no_annealing \
                   --no_lr_schedule \
                   --num_workers ${num_workers} \
                   --flow ${flow} \
                   --num_flows ${flow_depth} \
                   --batch_size ${batch_size} \
                   --manual_seed ${manual_seed} \
                   --log_interval ${logging} \
                   --plot_resolution ${plot_resolution} \
                   --plot_interval ${plotting} &
        done
        python density.py --dataset u${u} \
               --experiment_name ${exp_name} \
               --no_cuda \
               --num_steps ${num_steps} \
               --learning_rate ${learning_rate} \
               --no_annealing \
               --no_lr_schedule \
               --num_workers ${num_workers} \
               --flow ${flow} \
               --num_flows 32 \
               --batch_size ${batch_size} \
               --manual_seed ${manual_seed} \
               --log_interval ${logging} \
               --plot_resolution ${plot_resolution} \
               --plot_interval ${plotting} ;

    done

    # affine
    python density.py --dataset u${u} \
           --experiment_name ${exp_name} \
           --no_cuda \
           --num_steps ${num_steps} \
           --learning_rate ${learning_rate} \
           --no_annealing \
           --no_lr_schedule \
           --num_workers ${num_workers} \
           --flow affine \
           --num_flows 1 \
           --batch_size ${batch_size} \
           --manual_seed ${manual_seed} \
           --log_interval ${logging} \
           --plot_resolution ${plot_resolution} \
           --plot_interval ${plotting} ;

done


    
