package org.sybila.ode.cuda;

public class CudaSimulationFilterGuards extends org.sybila.ode.AbstractSimulationFilterGuards {
	// currently, just a wrapper
	public CudaSimulationFilterGuards(int dimension) {
		super(dimension);
	}
	public CudaSimulationFilterGuards(int dimension, float[] guards, int[] guardIndexes) {
		super(dimension, guards, guardIndexes);
	}
}
