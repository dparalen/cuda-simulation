package org.sybila.ode.cuda;

import jcuda.runtime.cudaStream_t;
import jcuda.utils.KernelLauncher;
import org.sybila.ode.AbstractSimulationLauncher;
import org.sybila.ode.MultiAffineFunction;
import org.sybila.ode.SimulationFilteredLauncher;
import org.sybila.ode.SimulationFilterGuards;

abstract public class AbstractCudaFilteredSimulationLauncher extends AbstractSimulationLauncher
implements SimulationFilteredLauncher
{

	protected static final int BLOCK_SIZE = 32;

	protected static final int MAX_NUMBER_OF_THREADS_IN_BLOCK = 512;

	private SimulationFilteredWorkspace workspace;

	private cudaStream_t stream;

	public AbstractCudaFilteredSimulationLauncher(int dimension, int maxNumberOfSimulations, int maxSimulationSize, MultiAffineFunction function) {
		super(dimension, maxNumberOfSimulations, maxSimulationSize, function);
	}

	public AbstractCudaFilteredSimulationLauncher(int dimension, int maxNumberOfSimulations, int maxSimulationSize, MultiAffineFunction function, float minAbsDivergency, float maxAbsDivergency, float minRelDivergency, float maxRelDivergency) {
		super(dimension, maxNumberOfSimulations, maxSimulationSize, function, minAbsDivergency, maxAbsDivergency, minRelDivergency, maxRelDivergency);
		if (dimension > 100) {
			throw new IllegalArgumentException("The dimension can't be greater than 100.");
		}
	}

	public SimulationResult launch(
		float	time,
		float	timeStep,
		float	targetTime,
		float[] vectors,
		int		numberOfSimulations
	) {
		if (numberOfSimulations <= 0 || numberOfSimulations > getMaxNumberOfSimulations()) {
			throw new IllegalArgumentException("The number of simulations [" + numberOfSimulations + "] is out of the range [1, " + getWorkspace().getMaxNumberOfSimulations() + "].");
		}

		// prepare data
		getWorkspace().bindVectors(vectors);

		// run kernel
		KernelLauncher launcher = KernelLauncher.load(getKernelFile(), getKernelName());
		int blockDim = BLOCK_SIZE * (BLOCK_SIZE/2) / getWorkspace().getDimension();
		int gridDim = (int) Math.ceil(Math.sqrt((float) numberOfSimulations / blockDim));
		launcher.setGridSize(gridDim, gridDim);
		launcher.setBlockSize(blockDim, getWorkspace().getDimension(), 1);
		System.out.println(getWorkspace().getGuards());
		System.out.println(getWorkspace().getGuardsLength());
		System.out.println(getWorkspace().getGuardIndexes());
		launcher.call(
			time,
			timeStep,
			targetTime,
			getWorkspace().getDeviceVectors(),
			numberOfSimulations,
			getWorkspace().getMaxNumberOfSimulations(),
			getWorkspace().getDimension(),
			getMinAbsDivergency(),
			getMaxAbsDivergency(),
			getMinRelDivergency(),
			getMaxRelDivergency(),
			MAX_NUMBER_OF_EXECUTED_STEPS,
			getWorkspace().getMaxSimulationSize(),
			getWorkspace().getDeviceFunctionCoefficients(),
			getWorkspace().getDeviceFunctionCoefficientIndexes(),
			getWorkspace().getDeviceFunctionFactors(),
			getWorkspace().getDeviceFunctionFactorIndexes(),
			getWorkspace().getGuards(),
			getWorkspace().getGuardsLength(),
			getWorkspace().getGuardIndexes(),
			getWorkspace().getDeviceResultPoints(),
			getWorkspace().getDeviceResultTimes(),
			getWorkspace().getDeviceExecutedSteps(),
			getWorkspace().getDeviceReturnCodes()
		);

		// get result
		return getWorkspace().getResult(numberOfSimulations);
	}

	@Override
	public void finalize() {
		clean();
	}

	public void clean() {
		if (workspace == null) {
			workspace.destroy();
		}
	}

	protected cudaStream_t getStream() {
		if (stream == null) {
			stream = new cudaStream_t();
		}
		return stream;
	}

	protected SimulationFilteredWorkspace getWorkspace() {
		if (workspace == null) {
			workspace = new 
				SimulationFilteredWorkspace(
						getDimension(),
						getMaxNumberOfSimulations(),
						getMaxSimulationSize(),
						getFunction());
		}
		return workspace;
	}
	public void setFilterGuards(SimulationFilterGuards guards) {
		SimulationFilteredWorkspace workspace = this.getWorkspace();
		workspace.setGuards(guards.getGuards());
		workspace.setGuardIndexes(guards.getGuardIndexes());
	}
	public SimulationFilterGuards getFilterGuards() {
		SimulationFilteredWorkspace workspace = this.getWorkspace();
		SimulationFilterGuards guards = new 
				CudaSimulationFilterGuards(
						workspace.getDimension(),
						workspace.getHostGuards(),
						workspace.getHostGuardIndexes());
		return guards;
	}

	abstract protected String getKernelFile();

	abstract protected String getKernelName();

}
