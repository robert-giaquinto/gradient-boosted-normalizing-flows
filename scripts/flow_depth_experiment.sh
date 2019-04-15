# Hypothesis: ensemble flows are relatively stronger for deeper flows
# Experiment: Run planar flows for varying flow lengths as baseline
# 			  Run ensemble flow as comparison


# Load defaults for all experiments
source /export/scratch/robert/ensemble-normalizing-flows/scripts/experiment_config.sh

# activate virtual environment
cd /export/scratch/robert/ensemble-normalizing-flows
source ./venv/bin/activate

# define variable specific to this experiment
num_learners=16
boosting_reweighting=zk
out_dir=snapshots/flow_depth_4_17_2019

#for flow_depth in 1 2 3 4 6 8 12 16 32; do
for flow_depth in 1 4 8 16; do
	# planar flow baseline
	python main_experiment.py --dataset ${dataset} \
	       --validation \
               --batch_size ${bs} \
               --warmup ${warmup} \
               --epochs ${epochs} \
               --early_stopping_epochs ${early_stopping_epochs} \
               --z_size ${z_size} \
	       --flow planar \
	       --num_flows ${flow_depth} \
	       --manual_seed ${seed} \
               --out_dir ${out_dir} \
	       --log_interval ${log_int} \
	       --plot_interval ${plot_int} ;

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
	       --num_flows ${flow_depth} \
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
	       --num_flows ${flow_depth} \
	       --manual_seed ${seed} \
               --out_dir ${out_dir} \
	       --log_interval ${log_int} \
	       --plot_interval ${plot_int} ;

done

# run no flow VAE for additional baseline
python main_experiment.py --dataset ${dataset} \
       --validation \
       --batch_size ${bs} \
       --warmup ${warmup} \
       --epochs ${epochs} \
       --early_stopping_epochs ${early_stopping_epochs} \
       --z_size ${z_size} \
       --flow no_flow \
       --manual_seed ${seed} \
       --out_dir ${out_dir} \
       --log_interval ${log_int} \
       --plot_interval ${plot_int} ;
