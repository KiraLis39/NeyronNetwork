package network.net;

import lombok.Getter;

@Getter
public final class Layer {
    private final int size;
    private final double[] neurons;
    private final double[] biases;
    public double[][] weights;

    public Layer(int size, int nextSize) {
        this.size = size;
        this.neurons = new double[size];
        this.biases = new double[size];
        this.weights = new double[size][nextSize];
    }

    public void setNeuron(int j, double value) {
        this.neurons[j] = value;
    }

    public void setBiase(int j, double value) {
        this.biases[j] = value;
    }
}
