package org.sybila.ode.cpu;

public class CpuSimulationFilterGuards
	// a stub
	extends org.sybila.ode.AbstractSimulationFilterGuards
	implements org.sybila.ode.SimulationFilterGuards
{
	public CpuSimulationFilterGuards (int dimension)
	{
		super(dimension);
	}
	public CpuSimulationFilterGuards (
			int dimension,
			float[] guards,
			int[] guardIndexes
			)
	{
		super(dimension, guards, guardIndexes);
	}
	public CpuSimulationFilterGuards(org.sybila.ode.SimulationFilterGuards guards)
	{
		super(guards.getDimension(), guards.getGuards(), guards.getGuardIndexes());
	}
}
