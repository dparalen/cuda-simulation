package org.sybila.ode.cpu;

public class CpuSimulationFilterGuards
	extends org.sybila.ode.AbstractSimulationFilterGuards
	implements org.sybila.ode.SimulationFilterGuards
{
	public CpuSimulationFilterGuards ()
	{
		super();
	}
	public CpuSimulationFilterGuards (
			int dimension,
			float[] guards,
			int[] guardsIndexes
			)
	{
		super(dimension, guards, guardsIndexes);
	}
	public CpuSimulationFilterGuards(org.sybila.ode.SimulationFilterGuards guards)
	{
		super(guards.getDimension(), guards.getGuards(), guards.getGuardIndexes());
	}
	public float[] getDimensionGuards(int dimensionId)
		throws ArrayIndexOutOfBoundsException
	{
		guardsIndexes = this.getGuardIndexes();
		guards = this.getGuards();
		int begin = guardsIndexes[2 * dimensionId];
		int end = guardsIndexes[2 * dimensionId + 1];
		float[] ret = new float[end - begin];
		for(int i = begin; i <= end; i++){
			ret[i - begin] = guards[i];
		}
		return ret;
	}
}
