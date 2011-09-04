package org.sybila.ode;

import org.sybila.ode.SimulationFilterGuards;

public abstract class AbstractSimulationFilteredLauncherTest extends AbstractSimulationLauncherTest
{
	protected static final float[] GUARDS = {0.4f, 0.3f};
	protected static final int[] GUARDINDEXES = {0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
	protected static final float[] GUARDS_DIM1 = {0.4f, 0.3f};
	protected static final float[] GUARDS_DIM3 = {0.5f, 0.6f};
}
