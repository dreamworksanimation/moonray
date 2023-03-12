mkdir -p best_candidate_points
sys_loop 1-4096 -cpus 16 "./stratified_best_candidate %4d best_candidate_points/points%4d.dat"
cat ./best_candidate_points/points* > ./best_candidate_points/bc_1D_integrator_points_4096_sequences_of_1024.dat
./ascii_to_binary 0 1 best_candidate_points/bc_1D_integrator_points_4096_sequences_of_1024.dat best_candidate_points/bc_1D_integrator_points_4096_sequences_of_1024.bin
