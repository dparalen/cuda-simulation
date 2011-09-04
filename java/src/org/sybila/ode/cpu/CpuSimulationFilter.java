package org.sybila.ode.cpu;

//import org.sybila.ode.cpu.CpuSimulationFilterGuards;
public class CpuSimulationFilter
{
	private CpuSimulationFilterGuards guards;
	private Point oldPoint;
	private int dimension;

	public CpuSimulationFilter(int dimension)
	// empty filter case
	{
		this.guards = new CpuSimulationFilterGuards(dimension);
		this.dimension = dimension;
		this.reset();
	}
	public CpuSimulationFilter(CpuSimulationFilterGuards guards)
	{
		this.guards = guards;
		this.dimension = guards.getDimension();
		this.reset();
	}
	public CpuSimulationFilter(org.sybila.ode.SimulationFilterGuards guards) {
		this.guards = new CpuSimulationFilterGuards(guards);
		this.dimension = guards.getDimension();
		this.reset();
	}
	public void reset(){
	  // dummy point prior to the reset call
	  float[] data = new float [this.dimension];
	  this.oldPoint = new Point(this.dimension, 0f, data);
	}
	public void reset(Point point)
	{
	  this.oldPoint = point;
	}
	public boolean changed(Point point)
		throws ArrayIndexOutOfBoundsException
	// check whether the provided point differs in evaluation from the last
	// changed point
	{
		if (this.guards.empty()) {
			// if no guards, filter always changed
			return true;
		}
		// true if at least one dimension of a point changed 
		// it's evaluation
		for(int dimId = 0; dimId < this.dimension; dimId++)
		{
			float[] dimGuards = this.guards.getDimensionGuards(dimId);
			for(int guardId = 0; guardId < dimGuards.length; guardId++)
			{
				if ((point.getValue(dimId) < dimGuards[guardId]) ^
						(this.oldPoint.getValue(dimId) <
						 dimGuards[guardId]))
				{
					return true;
				}
			}
		}
		return false;
	}
	public boolean pass(Point point)
		throws ArrayIndexOutOfBoundsException
	// check the point and update the filter based on the outcome
	{
		if (changed(point)) {
			this.reset(point);
			return true;
		}
		return false;
	}
	public boolean block(Point point)
		throws ArrayIndexOutOfBoundsException
	// check the point and update the filter based on the outcome
	{
		 return ! this.pass(point);
	}
	public CpuSimulationFilterGuards getGuards()
	{
		 return this.guards;
	}
	public void setGuards(CpuSimulationFilterGuards guards)
	{
		this.guards = guards;
		this.dimension = guards.getDimension();
		this.reset();
	}
}
