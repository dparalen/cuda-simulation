/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package org.sybila.ode.cuda;

import org.sybila.ode.AbstractSimulationLauncherTest;
import org.sybila.ode.MultiAffineFunction;
import org.sybila.ode.SimulationLauncher;

public class CudaSimulationLauncherTest extends AbstractSimulationLauncherTest
{

	@Override
	protected SimulationLauncher createLauncher(int dimension, int numberOfSimulations, int maxSimulationSize, MultiAffineFunction function) {
		return new CudaSimulationLauncher(dimension, numberOfSimulations, maxSimulationSize, function);
	}


}
