# Load defaults for all experiments
source ./scripts/experiment_config.sh

# activate virtual environment
source ./venv/bin/activate

# variables specific to this experiment
num_steps=100000
exp_name=density
batch_size=128
logging=1000


for flow_depth in 1 4 8 16
do
    for u in 1 2 3 4 5
    do
        python density.py --dataset u${u} \
               --experiment_name ${exp_name} \
               --no_cuda \
               --num_steps ${num_steps} \
               --num_workers ${num_workers} \
               --flow planar \
               --num_flows ${flow_depth} \
               --batch_size ${batch_size} \
               --manual_seed ${seed} \
               --log_interval ${logging} \
               --plot_interval ${logging} ;
    done

    for dataset in 8gaussians swissroll moons pinwheel cos 2spirals checkerboard line line-noisy circles joint_gaussian
    do
    python density.py --dataset ${dataset} \
           --experiment_name ${exp_name} \
           --no_cuda \
           --num_steps ${num_steps} \
           --num_workers ${num_workers} \
           --flow planar \
           --num_flows ${flow_depth} \
           --batch_size ${batch_size} \
           --manual_seed ${seed} \
           --log_interval ${logging} \
           --plot_interval ${logging} ;
    done
done


