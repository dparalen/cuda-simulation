package org.sybila.ode;

public abstract class AbstractSimulationFilterGuards
	implements SimulationFilterGuards
{
	protected int guardsLength;
	protected float[] guards;
	protected int[] guardsIndexes;
	protected int dimension;

	public AbstractSimulationFilterGuards()
	{
		/* no filtering case */
		this.guards = new float[0];
		this.guardsIndexes = new int[0];
	}

	public AbstractSimulationFilterGuards(
			int dimension,
			float[] guards,
			int[] guardsIndexes
			)
		throws IllegalArgumentException,
					ArrayIndexOutOfBoundsException
	{
		this.dimension = dimension;
		if (guardsIndexes.length != 2 * dimension &&
				guardsIndexes.length != 0) {
			throw new IllegalArgumentException("GuardsIndexes of incompatible size; should be either 0 or 2*dimension");
		}
		if (guards.length > 0 && guardsIndexes.length == 0) {
			throw new IllegalArgumentException ("GuardsIndexes.lengt == 0 " +
					"while Guards.length > 0; should be 2*dimensions");
		}
		for (int i = 0; i < dimension; i++) {
			/* check that no index points further than the length of
			 * guards */
			if ((guardsIndexes[2 * i] < 0 || guardsIndexes[2 * i] >= guards.length) ||
					(guardsIndexes[2 * i + 1] < 0) ||
					(guardsIndexes [ 2 * i + 1 ] >= guards.length)) {
				throw new ArrayIndexOutOfBoundsException ("GuardsIndexes out of range");
			}
		}
		this.guards = guards;
		this.guardsIndexes = guardsIndexes;
	}
	public float[] getGuards()
	{
		return this.guards;
	}
	public int[] getGuardIndexes()
	{
		return this.guardsIndexes;
	}
	public int getDimension()
	{
		return this.dimension;
	}
	public boolean empty()
	{
		return this.guards.length == 0;
	}
}

