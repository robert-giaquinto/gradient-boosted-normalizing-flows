# Load defaults for all experiments
source ./scripts/experiment_config.sh

# activate virtual environment
source ./venv/bin/activate

# variables specific to this experiment
experiment_name=baseline


for dataset in mnist
do
    # realnvp and iaf with various h_sizes
    for h_size in 128 192 256
    do
        # realnvp
        for num_flows in 1 #2
        do
            python main_experiment.py --dataset ${dataset} \
                   --experiment_name ${experiment_name} \
                   --validation \
                   --no_cuda \
                   --manual_seed ${manual_seed} \
                   --num_workers ${num_workers} \
                   --epochs ${epochs} \
                   --learning_rate ${learning_rate} \
                   --no_lr_schedule \
                   --early_stopping_epochs ${early_stop} \
                   --annealing_schedule ${annealing_schedule} \
                   --batch_size ${batch_size} \
                   --num_flows ${num_flows} \
                   --flow realnvp \
                   --num_base_layers 1 \
                   --base_network relu \
                   --h_size ${h_size} \
                   --z_size ${z_size} \
                   --plot_interval ${plotting} &
        done
        
        # iaf
        for num_hidden_layers in 0 #1 2
        do
            python main_experiment.py --dataset ${dataset} \
                   --experiment_name ${experiment_name} \
                   --validation \
                   --no_cuda \
                   --manual_seed ${manual_seed} \
                   --num_workers ${num_workers} \
                   --epochs ${epochs} \
                   --learning_rate ${learning_rate} \
                   --no_lr_schedule \
                   --early_stopping_epochs ${early_stop} \
                   --annealing_schedule ${annealing_schedule} \
                   --batch_size ${batch_size} \
                   --num_flows ${num_flows} \
                   --flow iaf \
                   --num_base_layers ${num_hidden_layers} \
                   --num_flows 1 \
                   --h_size ${h_size} \
                   --plot_interval ${plotting} ;
        done
    done

    # basic flows only need to tune num_flows
    for flow_depth in 4 8 16 32
    do
        for flow in planar radial
        do
            python main_experiment.py --dataset ${dataset} \
                   --experiment_name ${experiment_name} \
                   --validation \
                   --no_cuda \
                   --manual_seed ${manual_seed} \
                   --num_workers ${num_workers} \
                   --epochs ${epochs} \
                   --learning_rate ${learning_rate} \
                   --no_lr_schedule \
                   --early_stopping_epochs ${early_stop} \
                   --annealing_schedule ${annealing_schedule} \
                   --batch_size ${batch_size} \
                   --flow ${flow} \
                   --num_flows ${flow_depth} \
                   --plot_interval ${plotting} &
        done

        # Non-Linear Squared flow (no annealing, lower LR to prevent collapse)
        python main_experiment.py --dataset ${dataset} \
               --experiment_name ${experiment_name} \
               --validation \
               --no_cuda \
               --manual_seed ${manual_seed} \
               --num_workers ${num_workers} \
               --epochs ${epochs} \
               --learning_rate 0.0005 \
               --no_lr_schedule \
               --no_annealing \
               --early_stopping_epochs ${early_stop} \
               --annealing_schedule ${annealing_schedule} \
               --batch_size ${batch_size} \
               --flow nlsq \
               --num_flows ${flow_depth} \
               --plot_interval ${plotting} &

        # sylvester orthogolonal flows
        python main_experiment.py --dataset ${dataset} \
               --experiment_name ${experiment_name} \
               --validation \
               --no_cuda \
               --manual_seed ${manual_seed} \
               --num_workers ${num_workers} \
               --epochs ${epochs} \
               --learning_rate ${learning_rate} \
               --no_lr_schedule \
               --early_stopping_epochs ${early_stop} \
               --annealing_schedule ${annealing_schedule} \
               --batch_size ${batch_size} \
               --flow orthogonal \
               --num_ortho_vecs 16 \
               --num_flows ${flow_depth} \
               --plot_interval ${plotting} ;
    done

    # affine
    python main_experiment.py --dataset ${dataset} \
           --experiment_name ${experiment_name} \
           --validation \
           --no_cuda \
           --manual_seed ${manual_seed} \
           --num_workers ${num_workers} \
           --epochs ${epochs} \
           --learning_rate ${learning_rate} \
           --no_lr_schedule \
           --early_stopping_epochs ${early_stop} \
           --annealing_schedule ${annealing_schedule} \
           --batch_size ${batch_size} \
           --flow affine \
           --num_flows 1 \
           --plot_interval ${plotting} ;

done


    
