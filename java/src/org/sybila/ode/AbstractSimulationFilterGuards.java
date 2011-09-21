package org.sybila.ode;

public abstract class AbstractSimulationFilterGuards
	implements SimulationFilterGuards
{
	protected int guardsLength;
	protected float[] guards;
	protected int[] guardIndexes;
	protected int dimension;
	public AbstractSimulationFilterGuards(int dimension) {
		this.dimension = dimension;
		this.guardIndexes = new int[2 * dimension];
		reset();
	}
	public AbstractSimulationFilterGuards(
			int dimension,
			float[] guards,
			int[] guardIndexes
			)
		throws IllegalArgumentException,
					ArrayIndexOutOfBoundsException
	{
		this.dimension = dimension;
		if (guardIndexes.length != 2 * dimension &&
				guardIndexes.length != 0) {
			throw new IllegalArgumentException("GuardIndexes of incompatible size; should be either 0 or 2*dimension");
		}
		if (guards.length > 0 && guardIndexes.length == 0) {
			throw new IllegalArgumentException ("GuardsIndexes.lengt == 0 " +
					"while Guards.length > 0; should be 2*dimensions");
		}
		for (int i = 0; i < dimension; i++) {
			/* check that no index points further than the length of
			 * guards */
			if ((guardIndexes[2 * i] < 0 || guardIndexes[2 * i] >= guards.length) ||
					(guardIndexes[2 * i + 1] < 0) ||
					(guardIndexes [ 2 * i + 1 ] >= guards.length)) {
				//throw new ArrayIndexOutOfBoundsException ("GuardsIndexes out of range");
				;
			}
		}
		this.guards = guards;
		this.guardIndexes = guardIndexes;
	}
	public float[] getGuards()
	{
		return this.guards;
	}
	public int[] getGuardIndexes()
	{
		return this.guardIndexes;
	}
	public int getDimension()
	{
		return this.dimension;
	}
	public boolean empty()
	{
		return this.guards.length == 0;
	}
	public int dimGuardsBegin(int dimensionId)
		throws ArrayIndexOutOfBoundsException
	{
		return this.guardIndexes[2 * dimensionId];
	}
	public int dimGuardsEnd(int dimension)
		throws ArrayIndexOutOfBoundsException
	{
		return this.guardIndexes[2 * dimension + 1];
	}
	public float[] getDimensionGuards(int dimensionId)
		// get a list of guards of a particular dimension ID
		throws ArrayIndexOutOfBoundsException
	{
		// FIXME: System.out.println("Getting guards: " + dimensionId);
		// FIXME: System.out.println(this.toString());
		int begin = dimGuardsBegin(dimensionId);
		int end = dimGuardsEnd(dimensionId);
		float[] ret = new float[end - begin + 1];
		for(int i = begin; i <= end; i++){
			ret[i - begin] = this.guards[i];
		}
		return ret;
	}
	public void setDimensionGuards(int dimensionId, float[] guards)
		// throws "ArrayOut..." in case wrong dimensionId provided
		throws ArrayIndexOutOfBoundsException
	{
		System.out.println(this.toString());
		int l = 0; // the last index
		int begin = dimGuardsBegin(dimensionId);
		int end = dimGuardsEnd(dimensionId);
		// create new array of appropriate length
		float[] newGuards = new float[this.guards.length - (end - begin + 1) + guards.length];
		for(int i = 0; i < dimensionId; i++) {
			// go through "previous" dimensions copying elements to a new
			// guards array
			begin = dimGuardsBegin(i);
			end = dimGuardsEnd(i);
			for (int j = begin; j <= end; j++, l++) {
				newGuards[j] = this.guards[j];
			}
		}
		// the dimension being updated starts here
		this.guardIndexes[2 * dimensionId] = l;
		this.guardIndexes[2 * dimensionId + 1] = l + guards.length - 1;
		for(int j = l; l < j + guards.length; l++) {
			// copy new dimension guards
			newGuards[l] = guards[l - j];
		}
		for(int i = dimensionId + 1; i < this.dimension; i++) {
			// copy the rest of the dimension guards to the new array
			// "shifting" the indexes
			begin = dimGuardsBegin(i);
			end = dimGuardsEnd(i);
			this.guardIndexes[2 * i] = l;
			this.guardIndexes[2 * i + 1] = l + (end - begin);
			for (int j = begin; j <= end; j++, l++) {
				newGuards[l] = this.guards[j];
			}
		}
		this.guards = newGuards;
		System.out.println(this.toString());
	}
	public String toString() {
		String ret = new String("Length: " + this.guards.length);
		for (int k = 0; k < this.dimension; k++) {
			ret += "\n  " + k + ": " + this.guardIndexes[2 * k] + "," + this.guardIndexes[2 * k + 1];
			int begin = dimGuardsBegin(k);
			int end = dimGuardsEnd(k);
			if (begin <= end) {
				ret += "\n    ";
				for (int l = begin; l <= end; l++) {
					ret += this.guards[l] + ", ";
				}
			}
		}
		return ret;
	}
	public void reset(){
		// sets all indexes and guards so as no guards in any dimension are
		// indexed i.e. begin index greater than end index in each dimension
		// field
		for(int i = 0; i < this.dimension; i++) {
			this.guardIndexes[2 * i ] = 1;
			this.guardIndexes[2 * i + 1 ] = 0;
		}
		this.guards = new float[0];
	}
}
