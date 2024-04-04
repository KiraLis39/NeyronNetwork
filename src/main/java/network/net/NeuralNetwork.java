package network.net;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import network.gui.iWorkFrame;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.function.UnaryOperator;


@Slf4j
public class NeuralNetwork {
    private final ObjectMapper mapper = new ObjectMapper();
    private final UnaryOperator<Double> activation;
    private final UnaryOperator<Double> derivative;
    private Layer[] layers;

    private int iterationCount;
    @Setter
    private double learningRate;
    @Setter
    private double decay;

    @Setter
    private boolean showLRateInfo;
    private long was;
    private iWorkFrame form;


    public NeuralNetwork(iWorkFrame frame, double learningRate, double decay, UnaryOperator<Double> activation, UnaryOperator<Double> derivative, int... sizes) {
        this.learningRate = learningRate;
        this.decay = decay;
        this.activation = activation;
        this.derivative = derivative;
        this.layers = new Layer[sizes.length];
        this.form = frame;

        this.was = System.currentTimeMillis();

        for (int i = 0; i < sizes.length; i++) {
            int nextSize = 0;
            if(i < sizes.length - 1) {nextSize = sizes[i + 1];}

            layers[i] = new Layer(sizes[i], nextSize);

            for (int j = 0; j < sizes[i]; j++) {
                layers[i].biases[j] = Math.random() * 2.0 - 1.0;
                for (int k = 0; k < nextSize; k++) {
                    layers[i].weights[j][k] = Math.random() * 2.0 - 1.0;
                }
            }

        }
    }

    public synchronized double[] feedForward(double[] inputs) {
        System.arraycopy(inputs, 0, layers[0].neurons, 0, inputs.length);

        for (int i = 1; i < layers.length; i++)  {
            Layer currentLayer = layers[i - 1];
            Layer nextLayer = layers[i];

            for (int j = 0; j < nextLayer.getSize(); j++) {
                nextLayer.neurons[j] = 0;
                for (int k = 0; k < currentLayer.getSize(); k++) {
                    nextLayer.neurons[j] += currentLayer.neurons[k] * currentLayer.weights[k][j];
                }
                nextLayer.neurons[j] += nextLayer.biases[j];
                nextLayer.neurons[j] = activation.apply(nextLayer.neurons[j]);
            }
        }

        return layers[layers.length - 1].neurons;
    }

    public synchronized void backpropagation(double[] targets) {
        double dlearningRate = learningRate * (1.0d / (1.0d + decay * iterationCount));
        if (dlearningRate < 0.001) {
            dlearningRate = 0.001; // защита от переполнения long.
        }

        double[] errors = new double[layers[layers.length - 1].getSize()];
        for (int i = 0; i < layers[layers.length - 1].getSize(); i++) {
            errors[i] = targets[i] - layers[layers.length - 1].neurons[i];
        }

        for (int k = layers.length - 2; k >= 0; k--) {
            Layer previousLayer = layers[k];
            Layer currentLayer = layers[k + 1];
            double[] errorsNext = new double[previousLayer.getSize()];
            double[] gradients = new double[currentLayer.getSize()];

            // формирование градиента biases:
            for (int i = 0; i < currentLayer.getSize(); i++) {
                gradients[i] = errors[i] * derivative.apply(layers[k + 1].neurons[i]);
                gradients[i] *= dlearningRate;
            }

            double[][] deltas = new double[currentLayer.getSize()][previousLayer.getSize()];
            for (int i = 0; i < currentLayer.getSize(); i++) {
                for (int j = 0; j < previousLayer.getSize(); j++) {
                    deltas[i][j] = gradients[i] * previousLayer.neurons[j];
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
                currentLayer.biases[i] += gradients[i];
            }
        }

        iterationCount++;
    }

    public void save(String newName) {
//        form.out("Saving net...");

        String homeDir = "./dat/";
        try {
            if (!Files.exists(Path.of(homeDir))) {Files.createDirectory(Paths.get(homeDir));}
            Files.createFile(Paths.get(homeDir + "/" + newName + ".dat"));
        } catch (IOException e) {
            log.error("Error (002): {}", e.getMessage());
//            form.out("!!! Не удалось записать сеть !!!");
            return;
        }

        try (OutputStreamWriter osw = new OutputStreamWriter(new FileOutputStream(homeDir + "/" + newName + ".dat"))) {
            String gsonDump = mapper.writeValueAsString(layers);
            osw.write(gsonDump);
//            form.out("Net was saved.");
        } catch (IOException e) {
            log.error("Error (001): {}", e.getMessage());
        }
    }

    public void load(Path pathToDatFile) {
        try (BufferedReader br = new BufferedReader(new FileReader(pathToDatFile.toFile()))) {
            layers = mapper.readValue(br, Layer[].class);
//            form.out("Net was loaded.");
        } catch (Exception e) {
            log.error("Error (003): {}", e.getMessage());
//            form.out("!!! Не удалось загрузить сеть из файла !!!");
        }
    }
}
