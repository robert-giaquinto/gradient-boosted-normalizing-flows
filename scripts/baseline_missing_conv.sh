cd /export/scratch/robert/ensemble-normalizing-flows

# activate virtual environment
module unload soft/python
module load soft/python/anaconda
source /soft/python/anaconda/Linux_x86_64/etc/profile.d/conda.sh
conda activate env

# Load defaults for all experiments
source ./scripts/experiment_config.sh

# variables specific to this experiment
experiment_name=baseline_conv_run1
vae_layers=convolutional
realnvp_iaf_hidden_layers=0
realnvp_iaf_activation=tanh

for dataset in mnist freyfaces omniglot caltech
do

    echo "Running models on ${dataset}"
    
    for flow_depth in 16 #4 8 16
    do
        # realnvp with various h_sizes
        for h_size in 512 #128 256 512
        do
            python main_experiment.py --dataset ${dataset} \
                   --experiment_name ${experiment_name} \
                   --testing \
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
                   --flow realnvp \
                   --num_flows ${flow_depth} \
                   --num_base_layers ${realnvp_iaf_hidden_layers} \
                   --base_network ${realnvp_iaf_activation} \
                   --h_size ${h_size} \
                   --z_size ${z_size} \
                   --plot_interval ${plotting} ;
        done

        # radial flows have no additional hyperparameters
        python main_experiment.py --dataset ${dataset} \
               --experiment_name ${experiment_name} \
               --testing \
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
               --flow radial \
               --num_flows ${flow_depth} \
               --z_size ${z_size} \
               --plot_interval ${plotting} ;
    done
done
wait
echo "Job complete"

