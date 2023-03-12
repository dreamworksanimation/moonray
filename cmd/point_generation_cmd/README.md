To compile:
> scl enable devtoolset-6 bash # We need a newer GCC than we have by default
> make

The point generation library has several data structures for nearest-neighbor searching:
* DynamicHyperGrid: a dynamically-resizing N-dimensional grid that allows visiting neighbors in the hyper-torus space and projected space.
* StaticHyperGrid: a statically-allocated N-dimensional grid that allows visiting neighbors in the hyper-torus space and projected space. There is a specialization for pointers.
* PerfectPowerArray: a statically-allocated N-dimensional grid that allows visiting neighbors in the hyper-torus space. I believe this should be deprecated.

There are several applications:
* blue_noise_pd_progressive_generation: A variant of blue noise generation
  based on Projective Blue-Noise Sampling by Reinert, et al. http://resources.mpi-inf.mpg.de/ProjectiveBlueNoise/ProjectiveBlueNoise.pdf
* pd_generation: A simple dart-throwing Poisson disk generator
* progressive_pd_generation: Progessive Poisson disk generator
* reorder: Reorder points for progressive rendering
* ascii_to_binary: Convert ascii points to a binary file
* stratified_best_candidate: N-Dimensional stratified best candidate
* discrepancy: Measure star discrepancy of a set of points
* pmj02: Pixar's progressive multi-jittered sampling: https://graphics.pixar.com/library/ProgressiveMultiJitteredSampling/paper.pdf
