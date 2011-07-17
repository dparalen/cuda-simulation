#define A2 (1.0/4.0)
#define B2 (1.0/4.0)

#define A3 (3.0/8.0)
#define B3 (3.0/32.0)
#define C3 (9.0/32.0)

#define	A4 (12.0/13.0)
#define B4 (1932.0/2197.0)
#define C4 (-7200.0/2197.0)
#define D4 (7296.0/2197.0)

#define A5 1.0
#define B5 (439.0/216.0)
#define C5 (-8.0)
#define D5 (3680.0/513.0)
#define E5 (-845.0/4104.0)

#define A6 (1.0/2.0)
#define B6 (-8.0/27.0)
#define C6 2.0
#define D6 (-3544.0/2565.0)
#define E6 (1859.0/4104.0)
#define F6 (-11.0/40.0)

#define R1 (1.0/360.0)
#define R3 (-128.0/4275.0)
#define R4 (-2197.0/75240.0)
#define R5 (1.0/50.0)
#define R6 (2.0/55.0)

#define N1 (25.0/216.0)
#define N3 (1408.0/2565.0)
#define N4 (2197.0/4104.0)
#define N5 (-1.0/5.0)

#define MINIMUM_TIME_STEP 0.0000001
#define MAXIMUM_TIME_STEP 100
#define MINIMUM_SCALAR_TO_OPTIMIZE_STEP 0.00001
#define MAXIMUM_SCALAR_TO_OPTIMIZE_STEP 4.0

#define STATUS_OK 0
#define STATUS_TIMEOUT 1
#define STATUS_PRECISION 2
#include "./num_sim_kernel.h"


/* FILTERING
 *
 * Traces are being checked for "changes" in atomic propositions
 * before being processed any further. This happens in order
 * to minimize the space required for a trace storage as well as
 * to minimize the number of cycles of further processing on CPU
*/

/* some LIMITs on the amount of guards that can be used */
#ifndef _MAX_GUARDS
#define _MAX_GUARDS (2048)
#endif

#ifndef _MAX_CHANGED_VECTORS
#define _MAX_CHANGED_VECTORS (1024)
#endif

#ifndef _MAX_DIM_OLD_RECORDS
#define _MAX_DIM_OLD_RECORDS (1024)
#endif

#define _GE(a, b) ((a) >= (b))
#define _GT(a, b) ((a) > (b))
#define _LE(a, b) ((a) <= (b))
#define _LT(a, b) ((a) < (b))
/* a guard structure */
struct guard {
	size_t		dimension_id;	/* the dimension */
	float		value;		/* the guard value */
};


/* a structure storing all the guards */
/* it rather might be stored in a global space if more guards are needed */
static __constant__  __device__ struct {
	struct guard 	guards[_MAX_GUARDS];
	size_t		guards_len;
} FILTER_GUARDS;

static __device__ __shared__ struct {
	/* array of vectors (per block) that changed 
	 * threads synchronize the result of particular dimension here
	*/ 
	/* TODO: figure out how to avoid a constant here */
	unsigned short	vector[_MAX_CHANGED_VECTORS];
	unsigned short	dim_old[_MAX_DIM_OLD_RECORDS];

	/* assorted config stuff */ 
	struct {
		size_t 	simulations_per_block;
		size_t 	dimensions;
	} config;

} FILTER_CHECKED_VECTORS;

#define _ID_IN_BLOCK (blockIdx.x * threadIdx.y + threadIdx.x)
/* compute the BLOCK ID pointing to a vector change array
 * to report/read its modification from the guards pow.
*/
#define _FILTER_CHECKED_VECTOR_IDX(simulation_id) \
	((simulation_id) % \
	FILTER_CHECKED_VECTORS.config.simulations_per_block)
#define _FILTER_CHECKED_VECTOR(simulation_id) \
	(FILTER_CHECKED_VECTORS.vector\
		[_FILTER_CHECKED_VECTOR_IDX(simulation_id)])

#define _FILTER_DIM_OLD_INDEX _ID_IN_BLOCK
#define _FILTER_DIM_OLD_VALUE \
	(FILTER_CHECKED_VECTORS.dim_old[_FILTER_DIM_OLD_INDEX])

/* check whether there was any change on propositions */
/* a reduct per all the dimensions --- all thread observe same value */
#define _FILTER_VECTOR_CHANGED(simulation_id) \
	_FILTER_CHECKED_VECTOR(simulation_id)


/* initialize VECTOR FILTER; on-device part */
static __device__ void
filterDeviceInit (
	const size_t dimensions,
	const size_t simulations_per_block
){
	/* for threads within the vector--block size range
	 * init vector change state with false */
	if (threadIdx.y * blockDim.x + threadIdx.x  < \
		simulations_per_block * dimensions)
	{
		FILTER_CHECKED_VECTORS.vector \
			[threadIdx.y * blockDim.x + threadIdx.x] = false;
	}	
	/* stuff to be initialized only by the "first" thread */
	if(threadIdx.x != 0 || threadIdx.y != 0) return;
	FILTER_CHECKED_VECTORS.config.dimensions = dimensions;
	FILTER_CHECKED_VECTORS.config.simulations_per_block = \
		simulations_per_block;
}

/* evaluate and check the state of a vector entry
 *   - each thread computes a separate dimension
 *   - the "old" state is computed in a one shot togeter with change checking
*/
static void __device__ 
filterVectorEntryCheck (
	/* the simulation vector entry */
	const float
		vectentry,

	/* the vector field being calculated by this thread */
	const size_t
		dimension_id,

	/* the index of a simulation trace being computed */
	const size_t
		simulation_index
){
#define FS (FILTER_STATE)
#define FG (FILTER_GUARDS)

	register unsigned short ret = false;
	register size_t		i = 0;

	/* even though the cycle goes trough other dimensions' guards
	 * it has the benefit of the threads are kept in sync
	 * and the indexing easy and memory efficient
	 * the semantic is that if just one proposition changes its
	 * evaluation, a new "state" is entered with regard to trace
	 * model-checking
	*/
	/* please note that just the`>' inequality is checked here
	 * should be legitimate: TODO: prove
        */
	if (_ID_IN_BLOCK < FILTER_CHECKED_VECTORS.config.dimensions) {
		/* the base step; initialize "old" values */
		for ( i = 0; i < FG.guards_len ; i++) {
			if (dimension_id == FG.guards[i].dimension_id) {
				/* in case the dimension fits, initialize */
				_FILTER_DIM_OLD_VALUE |= FG.guards[i].value > \
					vectentry;
			}
		}
		/* first trace is always a change---prevents information loss*/
		ret = true;
	} else {
		/* induction step */
		for(i = 0; i < FG.guards_len; i++) {
			if (dimension_id == FG.guards[i].dimension_id) {
				/* in case the dimension fits, check&update */
				ret |= ((vectentry > FG.guards[i].value) ^ \
					_FILTER_DIM_OLD_VALUE);
				_FILTER_DIM_OLD_VALUE |= vectentry > \
					FG.guards[i].value;
			}
		}
	}
	/* update the status for the whole vector */
	FILTER_CHECKED_VECTORS.vector \
		[_FILTER_CHECKED_VECTOR_IDX(simulation_index)] |= ret;
#undef FG
#undef FS
}

__device__ void ode_function(
	const float 	time,
	const float* 	vector,
	const int	index,
	const int	vector_size,
	const float* 	coefficients,
	const int*	coefficient_indexes,
	const int*	factors,
	const int*	factor_indexes,
	float*		result
) {
	*result = 0;
	for(int c=coefficient_indexes[index]; c<coefficient_indexes[index+1]; c++) {
		float aux_result = coefficients[c];
		for(int f=factor_indexes[c]; f<factor_indexes[c+1]; f++) {
			aux_result *= vector[factors[f]];
		}
		*result += aux_result;
	}
}

extern "C"
void __global__ rkf45_kernel(
	const float 	time,
	const float 	time_step,
	const int	target_time,
	const float*	vectors,
	const int	number_of_vectors,
	const int	max_number_of_vectors,
	const int	vector_size,
	const float 	min_abs_divergency,
	const float 	max_abs_divergency,
	const float	min_rel_divergency,
	const float	max_rel_divergency,
	const int	max_number_of_steps,
	const int	simulation_max_size,
	const float* 	function_coefficients,
	const int*	function_coefficient_indexes,
	const int*	function_factors,
	const int*	function_factor_indexes,
	float* 		result_points,
	float*		result_times,
	int*		number_of_executed_steps,
	int*		return_code
) {
	// compute the number of simulations per block
	__shared__ int simulations_per_block;
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		simulations_per_block = (blockDim.x * blockDim.y) / vector_size;
	}
	__threadfence_block();
	__syncthreads();

#ifdef WITH_FILTERING
	filterDeviceInit(vector_size, simulations_per_block);
	__threadfence_block();
	__syncthreads();
#endif

	// compute the id within the block
	/* blockDim is the "size" of a Block within the Grid */
	int id_in_block = threadIdx.y * blockDim.x + threadIdx.x;

	// if the thread can't compute any simulation, exit
	if (id_in_block >= vector_size * simulations_per_block) return;

	// compute index of simulation
	/* gridDim is the size of the Grid containing Blocks */
	int simulation_id = (blockIdx.y * gridDim.x + blockIdx.x) * simulations_per_block + id_in_block / vector_size;

	// if the thread is out of the number of simulations, exit
	if (simulation_id >= number_of_vectors) return;

	// compute index of dimension
	int dimension_id  = id_in_block % vector_size;

	// reset number of executed steps and set the default time step
	number_of_executed_steps[simulation_id] = 0;	
	float h = time_step;

	// set current time and position
	float current_time 	= time;
	int position		= 0;

	// note the pointer on the vetor
	float* vector = &(result_points[(simulation_max_size + 1) * vector_size * simulation_id  + vector_size * position]);

	// copy init vector to the result
	vector[dimension_id] = vectors[vector_size * simulation_id + dimension_id];

	// perform the simulation
	int steps = 0;
	while(steps < max_number_of_steps && current_time < target_time && position < simulation_max_size) {
		__threadfence_block();
		__syncthreads();
		steps++;
		float k1, k2, k3, k4, k5, k6, dim_value;
		dim_value = vector[dimension_id];
		// K1
		ode_function(current_time + h, vector, dimension_id, vector_size, function_coefficients, function_coefficient_indexes, function_factors, function_factor_indexes, &k1);
		k1 = k1 * h;
		// K2
		vector[dimension_id] = vector[dimension_id] + B2 * k1;
		__threadfence_block();
		__syncthreads();
		ode_function(current_time + A2 * h, vector, dimension_id, vector_size, function_coefficients, function_coefficient_indexes, function_factors, function_factor_indexes, &k2);
		k2 = k2 * h;
		// K3
		vector[dimension_id] = dim_value + B3 * k1 + C3 * k2;
		__threadfence_block();
		__syncthreads();
		ode_function(current_time + A3 * h, vector, dimension_id, vector_size, function_coefficients, function_coefficient_indexes, function_factors, function_factor_indexes, &k3);		
		k3 = k3 * h;
		// K4
		vector[dimension_id] = dim_value + B4 * k1 + C4 * k2 + D4 * k3;
		__threadfence_block();
		__syncthreads();
		ode_function(current_time + A4 * h, vector, dimension_id, vector_size, function_coefficients, function_coefficient_indexes, function_factors, function_factor_indexes, &k4);		
		k4 = k4 * h;
		// K5
		vector[dimension_id] = dim_value + B5 * k1 + C5 * k2 + D5 * k3 + E5 * k4;
		__threadfence_block();
		__syncthreads();
		ode_function(current_time + A5 * h, vector, dimension_id, vector_size, function_coefficients, function_coefficient_indexes, function_factors, function_factor_indexes, &k5);
		k5 = k5 * h;
		// K6
		vector[dimension_id] = dim_value + B6 * k1 + C6 * k2 + D6 * k3 + E6 * k4 + F6 * k5;
		__threadfence_block();	
		__syncthreads();
		ode_function(current_time + A6 * h, vector, dimension_id, vector_size, function_coefficients, function_coefficient_indexes, function_factors, function_factor_indexes, &k6);
		k6 = k6 * h;

		// reset vector
		__syncthreads();
		vector[dimension_id] = dim_value;
		__threadfence_block();
		__syncthreads();

		// error
		float my_error = abs(R1 * k1 + R3 * k3 + R4 * k4 + R5 * k5 + R6 * k6);
		__syncthreads();
		result_points[(simulation_max_size + 1) * vector_size * simulation_id  + vector_size * (position+1) + dimension_id] = my_error;
		__threadfence_block();
		__syncthreads();
		float error = 0;
		for (int i=0; i<vector_size; i++) {
			if (result_points[(simulation_max_size + 1) * vector_size * simulation_id  + vector_size * (position+1) + i] > error) {
				error = result_points[(simulation_max_size + 1) * vector_size * simulation_id  + vector_size * (position+1) + i];
			}
		}
		__syncthreads();

		// check absolute error
		if (max_abs_divergency > 0 && error >= max_abs_divergency) {
			// compute a new time step
			h /= 2;
			// precision has been lost
			if (h < MINIMUM_TIME_STEP) {
				return_code[simulation_id] = STATUS_PRECISION;
				number_of_executed_steps[simulation_id] = position;
				return;
			}
		}
		else {
			// result
			float dim_result = vector[dimension_id] + N1 * k1 + N3 * k3 + N4 * k4 + N5 * k5;
			// relative error			
			float rel_error  = 0;
			if (max_rel_divergency > 0 || min_rel_divergency > 0) {	
				// compute relative error
				result_points[(simulation_max_size + 1) * vector_size * simulation_id  + vector_size * (position+1) + dimension_id] = abs(my_error/dim_result);
				__threadfence_block();
				__syncthreads();
				// sync relative error
				for(int i=0; i<vector_size; i++) {
					if (rel_error < result_points[(simulation_max_size + 1) * vector_size * simulation_id  + vector_size * (position+1) + i]) {
						rel_error = result_points[(simulation_max_size + 1) * vector_size * simulation_id  + vector_size * (position+1) + i];
					}
				}
				__syncthreads();
			}
			// check relative error
			if (max_rel_divergency > 0 && rel_error > max_rel_divergency) {
				// compute a new time step
				h /= 2;
				if (h < MINIMUM_TIME_STEP) {
					return_code[simulation_id] = STATUS_PRECISION;
					number_of_executed_steps[simulation_id] = position;
					return;
				}
			}
			// save result
			else {
				current_time += h;
				vector[dimension_id] = dim_result;
				if (current_time >= time_step * (position+1)) {
					result_times[simulation_id * simulation_max_size + position] = current_time;
#ifdef WITH_FILTERING
					filterVectorEntryCheck(vector[dimension_id], dimension_id, simulation_id);
					__threadfence_block();
					__syncthreads();

					if ( _FILTER_VECTOR_CHANGED(simulation_id) ) {
						/* save new point only if vector change detected */
						/* a reduct per all the dimensions --- all thread observe same value */
						position++;
					}
#else
					position++;
#endif
					vector = &(result_points[(simulation_max_size + 1) * vector_size * simulation_id  + vector_size * position]);
					vector[dimension_id] = dim_result;
					if (error <= min_abs_divergency) {
						h *= 2;
						if (h > time_step) h = time_step;
					}
				}				
			}
		}
	}
	if (steps >= max_number_of_steps) return_code[simulation_id] = STATUS_TIMEOUT;
	else return_code[simulation_id] = STATUS_OK;
	number_of_executed_steps[simulation_id] = position;
}


extern "C"
void __global__ euler_simple_kernel(
	const float 	time,
	const float 	time_step,
	const int	target_time,
	const float*	vectors,
	const int	number_of_vectors,
	const int	max_number_of_vectors,
	const int	vector_size,
	const float 	min_abs_divergency,
	const float 	max_abs_divergency,
	const float	min_rel_divergency,
	const float	max_rel_divergency,
	const int	max_number_of_steps,
	const int	simulation_max_size,
	const float* 	function_coefficients,
	const int*	function_coefficient_indexes,
	const int*	function_factors,
	const int*	function_factor_indexes,
	float* 		result_points,
	float*		result_times,
	int*		number_of_executed_steps,
	int*		return_code
) {
	// compute index of simulation
	int simulation_id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	// if the thread is out of the number of simulations, exit
	if (simulation_id >= number_of_vectors) return;

	// reset number of executed steps and set the default time step
	number_of_executed_steps[simulation_id] = 0;	
	float h = time_step;

	// set current time and position
	float current_time 	= time;
	int position		= 0;

	// note the pointer on the vetor
	float* previous = &(result_points[(simulation_max_size + 1) * vector_size * simulation_id  + vector_size * position]);
	float* next 	= &(result_points[(simulation_max_size + 1) * vector_size * simulation_id  + vector_size * (position + 1)]);

	// copy init vector to the result
	for(int i=0; i<vector_size; i++) {
		previous[i] = vectors[vector_size * simulation_id + i];
	}

	// perform the simulation
	int steps = 0;
	
	while(steps < max_number_of_steps && current_time < target_time && position < simulation_max_size) {
		steps++;
		for(int dim=0; dim<vector_size; dim++) {
			float dim_result;
			ode_function(current_time + h, previous, dim, vector_size, function_coefficients, function_coefficient_indexes, function_factors, function_factor_indexes, &dim_result);
			next[dim] = previous[dim] + h * dim_result;
		}
		current_time += h;
		if (current_time >= time_step * (position+1)) {
			result_times[simulation_id * simulation_max_size + position] = current_time;
			position++;
			previous = &(result_points[(simulation_max_size + 1) * vector_size * simulation_id  + vector_size * position]);
			next = &(result_points[(simulation_max_size + 1) * vector_size * simulation_id  + vector_size * (position + 1)]);
		}
	}
	if (steps >= max_number_of_steps) return_code[simulation_id] = STATUS_TIMEOUT;
	else return_code[simulation_id] = STATUS_OK;
	number_of_executed_steps[simulation_id] = position;
}

extern "C"
void __global__ euler_kernel(
	const float 	time,
	const float 	time_step,
	const int	target_time,
	const float*	vectors,
	const int	number_of_vectors,
	const int	max_number_of_vectors,
	const int	vector_size,
	const float 	min_abs_divergency,
	const float 	max_abs_divergency,
	const float	min_rel_divergency,
	const float	max_rel_divergency,
	const int	max_number_of_steps,
	const int	simulation_max_size,
	const float* 	function_coefficients,
	const int*	function_coefficient_indexes,
	const int*	function_factors,
	const int*	function_factor_indexes,
	float* 		result_points,
	float*		result_times,
	int*		number_of_executed_steps,
	int*		return_code
) {
	// compute the number of simulations per block
	__shared__ int simulations_per_block;
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		simulations_per_block = (blockDim.x * blockDim.y) / vector_size;
	}
	__threadfence_block();
	__syncthreads();

	// compute the id within the block
	int id_in_block = threadIdx.y * blockDim.x + threadIdx.x;

	// if the thread can't compute any simulation, exit
	if (id_in_block >= vector_size * simulations_per_block) return;

	// compute index of simulation
	int simulation_id = (blockIdx.y * gridDim.x + blockIdx.x) * simulations_per_block + id_in_block / vector_size;

	// if the thread is out of the number of simulations, exit
	if (simulation_id >= number_of_vectors) return;

	// compute index of dimension
	int dimension_id  = id_in_block % vector_size;

	// reset number of executed steps and set the default time step
	number_of_executed_steps[simulation_id] = 0;	
	float h = time_step;

	// set current time and position
	float current_time 	= time;
	int position		= 0;

	// note the pointer on the vetor
	float* vector = &(result_points[(simulation_max_size + 1) * vector_size * simulation_id  + vector_size * position]);

	// copy init vector to the result
	vector[dimension_id] = vectors[vector_size * simulation_id + dimension_id];

	// perform the simulation
	int steps = 0;
	while(steps < max_number_of_steps && current_time < target_time && position < simulation_max_size) {	
		__threadfence_block();
		__syncthreads();
		steps++;
		float dim_result;
		ode_function(current_time + h, vector, dimension_id, vector_size, function_coefficients, function_coefficient_indexes, function_factors, function_factor_indexes, &dim_result);
		dim_result = vector[dimension_id] + h * dim_result;
		current_time += h;
		if (current_time >= time_step * (position+1)) {
			vector[dimension_id] = dim_result;
			result_times[simulation_id * simulation_max_size + position] = current_time;
			position++;
			vector = &(result_points[(simulation_max_size + 1) * vector_size * simulation_id  + vector_size * position]);
			vector[dimension_id] = dim_result;
		}
	}
	if (steps >= max_number_of_steps) return_code[simulation_id] = STATUS_TIMEOUT;
	else return_code[simulation_id] = STATUS_OK;
	number_of_executed_steps[simulation_id] = position;
}
