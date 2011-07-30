package org.sybila.ode.cuda;

import jcuda.CudaException;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;
import jcuda.runtime.cudaMemcpyKind;
import jcuda.runtime.cudaStream_t;
import jcuda.runtime.cudaPointerAttributes;

/* please note that if guardsLen == 0, the Kernel doesn't process any guards, i.e. no filtering happens */

public class CudaSimulationFilterGuards {
	/* filter implementation/container for CUDA */
	private cudaStream_t		stream;
	public CUdeviceptr		guards;
	public CUdeviceptr		guardsIndexes;	
	public int			guardsLength;
	private int			dimension;

	public CudaSimulationFilterGuards (
		int dimension,
		float[] guards,
		int[] guardsIndexes, 
		cudaStream_t stream
	) throws IllegalArgumentException, ArrayIndexOutOfBoundsException, CudaException {
		this.dimension = dimension;
		this.stream = stream;
		if (guardsIndexes.length != 2 * dimension && guardsIndexes.length != 0) {
			throw new IllegalArgumentException ("GuardsIndexes of incompatible size; should be either 0 or 2*dimension");
		}
		if ( guards.length > 0 && guardsIndexes.length == 0 ) {
			throw new IllegalArgumentException ("GuardsIndexes.length == 0 while Guards.length > 0; should be 2*dimensions");
		}
		for (int i = 0; i < dimension; i++) {
			/* check that no index points further than the length of guards */
			if (guardsIndexes[2 * i] < 0 || guardsIndexes[2 * i] >= guards.length ||
				guardsIndexes[2 * i + 1] < 0 || guardsIndexes [ 2 * i + 1 ] >= guards.length) {
				throw new ArrayIndexOutOfBoundsException ("GuardsIndexes out of range");

			}
		}
		initGuards(guards);
		initGuardsIndexes(guardsIndexes);
	}

	private void initGuards(float[] hostGuards)  throws CudaException {
		int error;
		if (hostGuards == null) {
			this.guardsLength = 0;
			return;
		}
		// free if already allocated
		destroyGuards();
		this.guards = new CUdeviceptr();
		System.out.println("Allocating guards memory on device");
		JCuda.cudaMalloc(this.guards, hostGuards.length * Sizeof.FLOAT);
		error = JCuda.cudaGetLastError();
		if (error != cudaError.cudaSuccess) {
			throw new CudaException(JCuda.cudaGetErrorString(error));
		}
		if (this.stream == null) {
			JCuda.cudaMemcpy(this.guards, Pointer.to(hostGuards), hostGuards.length * Sizeof.FLOAT,
				cudaMemcpyKind.cudaMemcpyHostToDevice);
		} else {
			JCuda.cudaMemcpyAsync(this.guards, Pointer.to(hostGuards), hostGuards.length * Sizeof.FLOAT,
				cudaMemcpyKind.cudaMemcpyHostToDevice, stream);
		}
		error = JCuda.cudaGetLastError();
		if (error != cudaError.cudaSuccess) {
			throw new CudaException(JCuda.cudaGetErrorString(error));
		}
		this.guardsLength = hostGuards.length;
	}

	private void initGuardsIndexes(int[] hostGuardsIndexes) throws CudaException {
		int error;
		if ( hostGuardsIndexes == null) {
			return;
		}
		// free if already allocated
		destroyGuardIndexes();
		this.guardsIndexes = new CUdeviceptr();
		System.out.println("Allocating guardsIndexes on device");
		JCuda.cudaMalloc(this.guardsIndexes, hostGuardsIndexes.length * Sizeof.INT);
		error = JCuda.cudaGetLastError();
		if (error != cudaError.cudaSuccess) {
			throw new CudaException(JCuda.cudaGetErrorString(error));
		}
		if (this.stream == null) {
			JCuda.cudaMemcpy(this.guardsIndexes, Pointer.to(hostGuardsIndexes), hostGuardsIndexes.length * Sizeof.INT,
				cudaMemcpyKind.cudaMemcpyHostToDevice);
		} else {
			JCuda.cudaMemcpyAsync(this.guardsIndexes, Pointer.to(hostGuardsIndexes), hostGuardsIndexes.length * Sizeof.INT,
				cudaMemcpyKind.cudaMemcpyHostToDevice, this.stream);
		}
		error = JCuda.cudaGetLastError();
		if (error != cudaError.cudaSuccess) {
			throw new CudaException(JCuda.cudaGetErrorString(error));
		}
	}

	private void destroy() {
		destroyGuards();
		destroyGuardIndexes();
	}

	private void destroyGuards(){
		if (this.guards == null) {
			return;
		}
		JCuda.cudaFree(this.guards);
		System.out.println("Destroyed guards on device");
	}

	private void destroyGuardIndexes(){
		if (this.guardsIndexes == null) {
			return;
		}
		JCuda.cudaFree(this.guardsIndexes);
		System.out.println("Destroyed guardsIndexes on device");
	}

	public  CUdeviceptr getGuardsPtr() {
		return this.guards;
	}

	public int getGuardsLengt() {
		return this.guardsLength;
	}

	public CUdeviceptr getGuardsIndexesPtr() {
		return this.guardsIndexes;
	}
}
