package org.sybila.ode;

public interface SimulationFilterGuards
{
	public float[] getGuards();
	public int[] getGuardIndexes();
	public int getDimension();
	public boolean empty();
}