# Hypothesis: ensemble flows with more components are better (up to a point)
# Experiment: Run ensemble flows with a fixed flow length,
#			  but varying number of weak learners.

bs=256
log_int=0
plot_int=25
seed=1
flow_depth=4

for num_learners in 2 4 8 16 32 64 128; do

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
