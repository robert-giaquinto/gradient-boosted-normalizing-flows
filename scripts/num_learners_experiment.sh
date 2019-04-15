# Hypothesis: ensemble flows with more components are better (up to a point)
# Experiment: Run ensemble flows with a fixed flow length,
#			  but varying number of weak learners.

# Load defaults for all experiments
source /export/scratch/robert/ensemble-normalizing-flows/scripts/experiment_config.sh

# activate virtual environment
cd /export/scratch/robert/ensemble-normalizing-flows
source ./venv/bin/activate

# define variable specific to this experiment
num_flows=8
boosting_reweighting=zk
out_dir=snapshots/num_learners_4_15_2019

#for num_learners in 1 2 3 4 8 16 32 64; do
for num_learners in 4 8 16 32; do
    # ensemble flow: boosting
    python main_experiment.py --dataset ${dataset} \
	   --validation \
           --batch_size ${bs} \
           --warmup ${warmup} \
           --epochs ${epochs} \
           --early_stopping_epochs ${early_stopping_epochs} \
           --z_size ${z_size} \
	   --flow boosted \
           --num_learners ${num_learners} \
	   --learner_type ${learner_type} \
           --aggregation_method ${aggregation_method} \
           --boosting_reweighting ${boosting_reweighting} \
	   --num_flows ${num_flows} \
	   --manual_seed ${seed} \
           --out_dir ${out_dir} \
	   --log_interval ${log_int} \
	   --plot_interval ${plot_int} ;

    # ensemble flow: bagging
    python main_experiment.py --dataset ${dataset} \
	   --validation \
           --batch_size ${bs} \
           --warmup ${warmup} \
           --epochs ${epochs} \
           --early_stopping_epochs ${early_stopping_epochs} \
           --z_size ${z_size} \
	   --flow bagged \
	   --num_learners ${num_learners} \
	   --learner_type ${learner_type} \
	   --num_flows ${num_flows} \
	   --manual_seed ${seed} \
           --out_dir ${out_dir} \
	   --log_interval ${log_int} \
	   --plot_interval ${plot_int} ;

done
