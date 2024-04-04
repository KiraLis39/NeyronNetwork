package network.net;

import lombok.Getter;

@Getter
public final class Layer {
    private final int size;
    public final double[] neurons;
    public final double[] biases;
    public double[][] weights;

    public Layer(int size, int nextSize) {
        this.size = size;
        this.neurons = new double[size];
        this.biases = new double[size];
        this.weights = new double[size][nextSize];
    }
}
