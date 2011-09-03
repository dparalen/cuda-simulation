package org.sybila.ode;

import org.sybila.ode.SimulationFilterGuards;

public abstract class AbstractSimulationFilteredLauncherTest extends AbstractSimulationLauncherTest
{
	protected static final float[] GUARDS = {0.51f, 0.52f};
	protected static final int[] GUARDINDEXES = {0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
}
