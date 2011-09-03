package org.sybila.ode.cpu;

import org.sybila.ode.AbstractSimulationFilteredLauncherTest;
import org.sybila.ode.MultiAffineFunction;
import org.sybila.ode.SimulationLauncher;
import org.sybila.ode.cpu.CpuRkf45SimulationLauncherTest;
import org.sybila.ode.cpu.CpuSimulationFilterGuards;

public class CpuRkf45SimulationFilteredLauncherTest extends AbstractSimulationFilteredLauncherTest
{

	@Override
	protected SimulationLauncher createLauncher(int dimension, int numberOfSimulations, int maxSimulationSize, MultiAffineFunction function) {
		SimulationLauncher launcher = new CpuRkf45SimulationLauncher(dimension, numberOfSimulations, maxSimulationSize, function);
		CpuSimulationFilterGuards guards = new CpuSimulationFilterGuards(dimension, this.GUARDS, this.GUARDINDEXES);
		launcher.setFilterGuards(guards);
		return launcher;
	}

}
