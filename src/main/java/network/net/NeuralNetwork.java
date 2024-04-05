package network.net;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import java.util.function.UnaryOperator;


@Slf4j
public class NeuralNetwork {
    private final UnaryOperator<Double> activation = x -> 1 / (1 + Math.exp(-x));
    private final UnaryOperator<Double> derivative = y -> y * (1 - y);
    private final int[] sizes = new int[] {
            2, // 2
            4, // -
            16, // 8
            32, // 10
            2, // 2
    };
    private final Layer[] layers = new Layer[sizes.length];

    @Setter
    @Getter
    private double learningRate = 0.0020D;

    @Setter
    private double decay = 0.0025D;
    private int iterationCount;


    public NeuralNetwork() {
        for (int i = 0; i < sizes.length; i++) {
            int nextSize = 0;
            if(i < sizes.length - 1) {
                nextSize = sizes[i + 1];
            }

            layers[i] = new Layer(sizes[i], nextSize);

            for (int j = 0; j < sizes[i]; j++) {
                layers[i].setBiase(j, Math.random() * 2.0d - 1.0d);
                for (int k = 0; k < nextSize; k++) {
                    layers[i].weights[j][k] = Math.random() * 2.0 - 1.0;
                }
            }
        }
    }

    public double[] feedForward(double[] inputs) {
        System.arraycopy(inputs, 0, layers[0].getNeurons(), 0, inputs.length);

        for (int i = 1; i < layers.length; i++)  {
            Layer currentLayer = layers[i - 1];
            Layer nextLayer = layers[i];

            for (int j = 0; j < nextLayer.getSize(); j++) {
                nextLayer.setNeuron(j, 0);
                for (int k = 0; k < currentLayer.getSize(); k++) {
                    nextLayer.setNeuron(j, nextLayer.getNeurons()[j] + currentLayer.getNeurons()[k] * currentLayer.weights[k][j]);
                }
                nextLayer.setNeuron(j, nextLayer.getNeurons()[j] + nextLayer.getBiases()[j]);
                nextLayer.setNeuron(j, activation.apply(nextLayer.getNeurons()[j]));
            }
        }

        return layers[layers.length - 1].getNeurons();
    }

    public void backpropagation(double[] targets) {
        double dlearningRate = learningRate * (1.0d / (1.0d + decay * iterationCount));
        if (dlearningRate < 0.001) {
            dlearningRate = 0.001; // защита от переполнения long.
        }

        double[] errors = new double[layers[layers.length - 1].getSize()];
        for (int i = 0; i < layers[layers.length - 1].getSize(); i++) {
            errors[i] = targets[i] - layers[layers.length - 1].getNeurons()[i];
        }

        for (int k = layers.length - 2; k >= 0; k--) {
            Layer previousLayer = layers[k];
            Layer currentLayer = layers[k + 1];
            double[] errorsNext = new double[previousLayer.getSize()];
            double[] gradients = new double[currentLayer.getSize()];

            // формирование градиента biases:
            for (int i = 0; i < currentLayer.getSize(); i++) {
                gradients[i] = errors[i] * derivative.apply(layers[k + 1].getNeurons()[i]);
                gradients[i] *= dlearningRate;
            }

            double[][] deltas = new double[currentLayer.getSize()][previousLayer.getSize()];
            for (int i = 0; i < currentLayer.getSize(); i++) {
                for (int j = 0; j < previousLayer.getSize(); j++) {
                    deltas[i][j] = gradients[i] * previousLayer.getNeurons()[j];
                }
            }

            for (int i = 0; i < previousLayer.getSize(); i++) {
                errorsNext[i] = 0;
                for (int j = 0; j < currentLayer.getSize(); j++) {
                    errorsNext[i] += previousLayer.weights[i][j] * errors[j];
                }
            }

            errors = new double[previousLayer.getSize()];
            System.arraycopy(errorsNext, 0, errors, 0, previousLayer.getSize());
            double[][] weightsNew = new double[previousLayer.weights.length][previousLayer.weights[0].length];
            for (int i = 0; i < currentLayer.getSize(); i++) {
                for (int j = 0; j < previousLayer.getSize(); j++) {
                    weightsNew[j][i] = previousLayer.weights[j][i] + deltas[i][j];
                }
            }

            previousLayer.weights = weightsNew;
            for (int i = 0; i < currentLayer.getSize(); i++) {
                currentLayer.setBiase(i, currentLayer.getBiases()[i] + gradients[i]);
            }
        }

        iterationCount++;
    }
}
