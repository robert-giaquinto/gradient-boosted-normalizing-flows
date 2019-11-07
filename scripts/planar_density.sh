# Load defaults for all experiments
source ./scripts/experiment_config_density.sh

# activate virtual environment
source ./venv/bin/activate

# variables specific to this experiment
num_steps=100001
exp_name=baseline_density
logging=1000
min_beta=1.0


for flow_depth in 2 4 8 16 32
do
    for u in 1 2 3 4 5
    do
        python density.py --dataset u${u} \
               --experiment_name ${exp_name} \
               --no_cuda \
               --num_steps ${num_steps} \
               --plot_resolution ${plot_resolution} \
               --min_beta ${min_beta} \
               --num_workers ${num_workers} \
               --flow planar \
               --num_flows ${flow_depth} \
               --batch_size ${batch_size} \
               --manual_seed ${seed} \
               --log_interval ${logging} \
               --plot_interval ${logging} ;
    done

    for dataset in 8gaussians 2gaussians swissroll rings moons pinwheel cos 2spirals checkerboard line line-noisy circles joint_gaussian
    do
    python density.py --dataset ${dataset} \
           --experiment_name ${exp_name} \
           --no_cuda \
           --num_steps ${num_steps} \
           --plot_resolution ${plot_resolution} \
           --min_beta ${min_beta} \
           --num_workers ${num_workers} \
           --flow planar \
           --num_flows ${flow_depth} \
           --batch_size ${batch_size} \
           --manual_seed ${seed} \
           --log_interval ${logging} \
           --plot_interval ${logging} ;
    done
done


