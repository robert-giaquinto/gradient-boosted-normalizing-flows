# Hypothesis: ensemble flows are relatively stronger for deeper flows
# Experiment: Run planar flows for varying flow lengths as baseline
# 			  Run ensemble flow as comparison

bs=256
log_int=0
plot_int=25
seed=1
num_learners=32

for flow_depth in 1 2 3 4 8 16 32; do
	# planar flow baseline
	python main_experiment.py -d mnist \
		--testing \
		--flow planar \
		--num_flows ${flow_depth} \
		--batch_size ${bs} \
		--manual_seed ${seed} \
		--log_interval ${log_int} \
		--plot_interval ${plot_int} ;

	# ensemble flow
	python main_experiment.py -d mnist \
		--testing \
		--flow boosted \
		--num_learners ${num_learners} \
		--learner_type planar \
		--num_flows ${flow_depth} \
		--batch_size ${bs} \
		--manual_seed ${seed} \
		--log_interval ${log_int} \
		--plot_interval ${plot_int} ;
done

# run no flow VAE for additional baseline
python main_experiment.py -d mnist \
	--testing \
	--flow no_flow \
	--batch_size ${bs} \
	--manual_seed ${seed} \
	--log_interval ${log_int} \
	--plot_interval ${plot_int} ;
