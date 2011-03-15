package org.sybila.ode;

public interface SimulationLauncher
{

	public SimulationResult launch(
			float time,
			float timeStep,
			float targetTime,
			float[] vectors,
			int	 numberOfSimulations,
			float absDivergency,
			int maxNumberOfExecutedSteps);

}
