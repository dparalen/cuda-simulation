package org.sybila.ode.cpu;

import java.util.Arrays;

public class Point implements org.sybila.ode.Point
{

	private float[] data;

	private int dimension;

	private float time;

	public Point(int dimension, float time, float[] data) {
		this.dimension = dimension;
		this.data = data;
		this.time = time;
	}

	public int getDimension() {
		return dimension;
	}

	public float getValue(int index) {
		if (index < 0 || index >= dimension) {
			throw new IndexOutOfBoundsException("The index [" + index + "] is out of the range [0," + dimension + "].");
		}
		return data[index];
	}

	public float getTime() {
		return time;
	}

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append(time + " ");
		builder.append("[");
		boolean notFirst = false;
		for(int i=0; i<dimension; i++) {
			if (notFirst) {
				builder.append(", " + data[i]);
			}
			else {
				builder.append(data[i]);
				notFirst = true;
			}
		}
		builder.append("]");
		return builder.toString();
	}

	public float[] toArray() {
		return Arrays.copyOf(data, data.length);
	}

}
