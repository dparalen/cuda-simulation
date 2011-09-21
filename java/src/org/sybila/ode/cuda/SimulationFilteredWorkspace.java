package org.sybila.ode.cuda;

import org.sybila.ode.MultiAffineFunction;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaStream_t;

public class SimulationFilteredWorkspace extends SimulationWorkspace {
	/* the guards and guardIndexes mustn't be null; sad that not everything is transparent */
	protected CUdeviceptr guards = null;
	protected int guardsLength = 0;
	
	protected CUdeviceptr guardIndexes = null;
	protected int guardIndexesLength = 0;

	public SimulationFilteredWorkspace(int dimension, int maxNumberOfSimulations, int maxSimulationSize, MultiAffineFunction function, cudaStream_t stream) {
		super(dimension, maxNumberOfSimulations, maxSimulationSize, function, stream);
	}

	public SimulationFilteredWorkspace(int dimension, int maxNumberOfSimulations, int maxSimulationSize, MultiAffineFunction function) {
		super(dimension, maxNumberOfSimulations, maxSimulationSize, function);
	}

	public void destroy() throws jcuda.CudaException {
		destroyGuards();
		destroyGuardIndexes();
		super.destroy();
	}

	private void destroyGuards() throws jcuda.CudaException {
		if (this.guards != null) {
			JCuda.cudaFree(guards);
			this.guards = null;
			this.guardsLength = 0;
		}
	}
	
	private void destroyGuardIndexes(){
		if (this.guardIndexes != null) {
			JCuda.cudaFree(guardIndexes);
			this.guardIndexes = null;
			this.guardIndexesLength = 0;
		}
	}
	public void setGuards(float[] guards) throws jcuda.CudaException {
		destroyGuards();
		this.guards = new CUdeviceptr();
		this.guardsLength = guards.length;
		JCuda.cudaMalloc(this.guards, this.guardsLength * Sizeof.FLOAT);
		copyHostToDevice(this.guards, Pointer.to(guards), this.guardsLength * Sizeof.FLOAT);
	}
	public void setGuards(CUdeviceptr guards){
		destroyGuards();
		this.guards = guards;
	}
	
	public void setGuardIndexes(int[] guardIndexes) throws jcuda.CudaException {
		destroyGuardIndexes();
		this.guardIndexes	= new CUdeviceptr();
		this.guardIndexesLength = guardIndexes.length;
		JCuda.cudaMalloc(this.guardIndexes, this.guardIndexesLength * Sizeof.INT);
		copyHostToDevice(this.guardIndexes, Pointer.to(guardIndexes), this.guardIndexesLength * Sizeof.INT);
	}
	public void setGuardIndexes(CUdeviceptr guardIndexes){
		destroyGuardIndexes();
		this.guardIndexes = guardIndexes;
	}
	
	public CUdeviceptr getGuards() {
		return this.guards;
	}
	public float[] getHostGuards() throws jcuda.CudaException {
		float[] guardsArray = new float[this.guardsLength];
		copyDeviceToHost(Pointer.to(guardsArray), this.guards, this.guardsLength * Sizeof.FLOAT);
		
		return guardsArray;
	}
	public int getGuardsLength(){
		return this.guardsLength;
	}

	public CUdeviceptr getGuardIndexes() {
		return this.guardIndexes;
	}
	public int[] getHostGuardIndexes() {
		int[] indexesArray = new int[this.guardIndexesLength];
		copyDeviceToHost(Pointer.to(indexesArray), this.guardIndexes, this.guardIndexesLength * Sizeof.INT);
		return indexesArray;
	}
	public int getGuardIndexesLength(){
		return this.guardIndexesLength;
	}
}
