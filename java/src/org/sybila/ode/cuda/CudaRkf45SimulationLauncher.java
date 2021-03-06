package org.sybila.ode.cuda;

import java.io.File;
import org.sybila.ode.MultiAffineFunction;

public class CudaRkf45SimulationLauncher extends AbstractCudaFilteredSimulationLauncher
{

	private static final String KERNEL_FILE = "c" + File.separator + "build" + File.separator + "num_sim_kernel.cubin";

	private static final String KERNEL_NAME = "rkf45_kernel";

	public CudaRkf45SimulationLauncher(int dimension, int maxNumberOfSimulations, int maxSimulationSize, MultiAffineFunction function) {
		super(dimension, maxNumberOfSimulations, maxSimulationSize, function);
	}

	public CudaRkf45SimulationLauncher(int dimension, int maxNumberOfSimulations, int maxSimulationSize, MultiAffineFunction function, float minAbsDivergency, float maxAbsDivergency, float minRelDivergency, float maxRelDivergency) {
		super(dimension, maxNumberOfSimulations, maxSimulationSize, function, minAbsDivergency, maxAbsDivergency, minRelDivergency, maxRelDivergency);
	}

	@Override
	protected String getKernelFile() {
		return KERNEL_FILE;
	}

	@Override
	protected String getKernelName() {
		return KERNEL_NAME;
	}
}
