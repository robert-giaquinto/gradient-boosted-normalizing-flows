cd /export/scratch/robert/ensemble-normalizing-flows

# activate virtual environment
module unload soft/python
module load soft/python/anaconda
source /soft/python/anaconda/Linux_x86_64/etc/profile.d/conda.sh
conda activate env

# Load defaults for all experiments
source ./scripts/experiment_config.sh

# variables specific to this experiment
experiment_name=baseline_linear_run1
vae_layers=linear
realnvp_iaf_hidden_layers=0
realnvp_iaf_activation=tanh

for dataset in mnist freyfaces omniglot caltech
do

    echo "Running models on ${dataset}"
    
    # no flow
    python main_experiment.py --dataset ${dataset} \
           --experiment_name ${experiment_name} \
           --validation \
           --no_cuda \
           --manual_seed ${manual_seed} \
           --num_workers 1 \
           --epochs ${epochs} \
           --learning_rate ${learning_rate} \
           --no_lr_schedule \
           --early_stopping_epochs ${early_stop} \
           --annealing_schedule ${annealing_schedule} \
           --vae_layers ${vae_layers} \
           --batch_size ${batch_size} \
           --flow no_flow \
           --plot_interval ${plotting} &    

    for flow_depth in 16 #4 8 16
    do
        # realnvp and iaf with various h_sizes
        for flow in realnvp iaf
        do
            for h_size in 512 #128 256 512
            do
                python main_experiment.py --dataset ${dataset} \
                       --experiment_name ${experiment_name} \
                       --validation \
                       --no_cuda \
                       --manual_seed ${manual_seed} \
                       --num_workers 1 \
                       --epochs ${epochs} \
                       --learning_rate ${learning_rate} \
                       --no_lr_schedule \
                       --early_stopping_epochs ${early_stop} \
                       --annealing_schedule ${annealing_schedule} \
                       --vae_layers ${vae_layers} \
                       --batch_size ${batch_size} \
                       --num_flows ${flow_depth} \
                       --flow ${flow} \
                       --num_base_layers ${realnvp_iaf_hidden_layers} \
                       --base_network ${realnvp_iaf_activation} \
                       --h_size ${h_size} \
                       --z_size ${z_size} \
                       --plot_interval ${plotting} &
            done
        done
        
        # planar and radial flows have no additional hyperparameters
        for flow in planar radial
        do
            python main_experiment.py --dataset ${dataset} \
                   --experiment_name ${experiment_name} \
                   --validation \
                   --no_cuda \
                   --manual_seed ${manual_seed} \
                   --num_workers 1 \
                   --epochs ${epochs} \
                   --learning_rate ${learning_rate} \
                   --no_lr_schedule \
                   --early_stopping_epochs ${early_stop} \
                   --annealing_schedule ${annealing_schedule} \
                   --vae_layers ${vae_layers} \
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
               --num_workers 1 \
               --epochs ${epochs} \
               --learning_rate 0.0001 \
               --no_lr_schedule \
               --early_stopping_epochs ${early_stop} \
               --annealing_schedule ${annealing_schedule} \
               --vae_layers ${vae_layers} \
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
               --num_workers 1 \
               --epochs ${epochs} \
               --learning_rate ${learning_rate} \
               --no_lr_schedule \
               --early_stopping_epochs ${early_stop} \
               --annealing_schedule ${annealing_schedule} \
               --vae_layers ${vae_layers} \
               --batch_size ${batch_size} \
               --flow orthogonal \
               --num_ortho_vecs 32 \
               --num_flows ${flow_depth} \
               --plot_interval ${plotting} &
    done
    wait
    
done
wait
echo "Job complete"


    
