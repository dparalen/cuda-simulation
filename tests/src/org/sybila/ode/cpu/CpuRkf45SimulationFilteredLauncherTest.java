package org.sybila.ode.cpu;

import org.sybila.ode.AbstractSimulationFilteredLauncherTest;
import org.sybila.ode.MultiAffineFunction;
import org.sybila.ode.SimulationFilteredLauncher;
import org.sybila.ode.SimulationLauncher;
import org.sybila.ode.cpu.CpuRkf45SimulationLauncherTest;
import org.sybila.ode.cpu.CpuSimulationFilterGuards;

public class CpuRkf45SimulationFilteredLauncherTest extends AbstractSimulationFilteredLauncherTest
{

	@Override
	protected SimulationLauncher createLauncher(int dimension, int numberOfSimulations, int maxSimulationSize, MultiAffineFunction function) {
		SimulationFilteredLauncher launcher = new CpuRkf45SimulationLauncher(dimension, numberOfSimulations, maxSimulationSize, function);
		//CpuSimulationFilterGuards guards = new CpuSimulationFilterGuards(dimension, this.GUARDS, this.GUARDINDEXES);
		CpuSimulationFilterGuards guards = new CpuSimulationFilterGuards(dimension);
		guards.setDimensionGuards(1, this.GUARDS_DIM1);
		guards.setDimensionGuards(3, this.GUARDS_DIM3);
		// a test of the simulation param settings method
		float[] tmp = {0.2f};
		guards.setDimensionGuards(3, tmp);
		tmp = new float[0];
		guards.setDimensionGuards(3, tmp);
		guards.setDimensionGuards(3, this.GUARDS_DIM3);
		launcher.setFilterGuards(guards);
		return (SimulationLauncher) launcher;
	}

}
