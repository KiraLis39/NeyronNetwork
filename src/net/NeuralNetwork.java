package net;

import com.FormDigits;
import com.google.gson.Gson;

import java.io.*;
import java.math.BigDecimal;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.function.UnaryOperator;


public class NeuralNetwork {
    private static Layer[] layers;
    private static UnaryOperator<Double> activation;
    private static UnaryOperator<Double> derivative;

    private static int iterationCount;
    private static double learningRate;
    private static double decay;

    private static boolean showLRateInfo;
    private static long was = System.currentTimeMillis();

    private Gson g = new Gson();


    public NeuralNetwork(double learningRate, double decay, UnaryOperator<Double> activation, UnaryOperator<Double> derivative, int... sizes) {
        this.learningRate = learningRate;
        this.decay = decay;
        this.activation = activation;
        this.derivative = derivative;
        this.layers = new Layer[sizes.length];

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

            for (int j = 0; j < nextLayer.size; j++) {
                nextLayer.neurons[j] = 0;
                for (int k = 0; k < currentLayer.size; k++) {
                    nextLayer.neurons[j] += currentLayer.neurons[k] * currentLayer.weights[k][j];
                }
                nextLayer.neurons[j] += nextLayer.biases[j];
                nextLayer.neurons[j] = activation.apply(nextLayer.neurons[j]);
            }
        }

        return layers[layers.length - 1].neurons;
    }

    public synchronized void backpropagation(double[] targets) {

        double dlearningRate = learningRate * (1.0 / (1.0 + decay * iterationCount));
        if (showLRateInfo && secondPass()) {
            FormDigits.out("NLR #" + iterationCount + ": " + (float)dlearningRate);}
        if (dlearningRate < 0.001) {dlearningRate = 0.001;} // защита от переполнения long.

        double[] errors = new double[layers[layers.length - 1].size];
        for (int i = 0; i < layers[layers.length - 1].size; i++) {
            errors[i] = targets[i] - layers[layers.length - 1].neurons[i];
        }

        for (int k = layers.length - 2; k >= 0; k--) {
            Layer previousLayer = layers[k];
            Layer currentLayer = layers[k + 1];
            double[] errorsNext = new double[previousLayer.size];
            double[] gradients = new double[currentLayer.size];

            // формирование градиента biases:
            for (int i = 0; i < currentLayer.size; i++) {
                gradients[i] = errors[i] * derivative.apply(layers[k + 1].neurons[i]);
                gradients[i] *= dlearningRate;
            }

            double[][] deltas = new double[currentLayer.size][previousLayer.size];
            for (int i = 0; i < currentLayer.size; i++) {
                for (int j = 0; j < previousLayer.size; j++) {
                    deltas[i][j] = gradients[i] * previousLayer.neurons[j];
                }
            }

            for (int i = 0; i < previousLayer.size; i++) {
                errorsNext[i] = 0;
                for (int j = 0; j < currentLayer.size; j++) {
                    errorsNext[i] += previousLayer.weights[i][j] * errors[j];
                }
            }

            errors = new double[previousLayer.size];
            System.arraycopy(errorsNext, 0, errors, 0, previousLayer.size);
            double[][] weightsNew = new double[previousLayer.weights.length][previousLayer.weights[0].length];
            for (int i = 0; i < currentLayer.size; i++) {
                for (int j = 0; j < previousLayer.size; j++) {
                    weightsNew[j][i] = previousLayer.weights[j][i] + deltas[i][j];
                }
            }

            previousLayer.weights = weightsNew;
            for (int i = 0; i < currentLayer.size; i++) {
                currentLayer.biases[i] += gradients[i];
            }
        }

        iterationCount++;
    }

    private boolean secondPass() {
        if (System.currentTimeMillis() - was > 2000) {
            was = System.currentTimeMillis();
            return true;
        }
        return false;
    }

    public void setShowLRateInfo(boolean showLRateInfo) {
        this.showLRateInfo = showLRateInfo;
    }


    public void save(String newName) {
        FormDigits.out("Saving net...");

        String homeDir = "./dat/";
        try {
            if (!Files.exists(Path.of(homeDir))) {Files.createDirectory(Paths.get(homeDir));}
            Files.createFile(Paths.get(homeDir + "/" + newName + ".dat"));
        } catch (IOException e) {
            e.printStackTrace();
            FormDigits.out("!!! Не удалось записать сеть !!!");
            return;
        }

        String gsonDump = g.toJson(layers);
        try (OutputStreamWriter osw = new OutputStreamWriter(new FileOutputStream(homeDir + "/" + newName + ".dat"))) {
            osw.write(gsonDump);
            FormDigits.out("Net was saved.");
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public void load(Path pathToDatFile) {
        try (BufferedReader br = new BufferedReader(new FileReader(pathToDatFile.toFile()))) {
            layers = g.fromJson(br, Layer[].class);
            FormDigits.out("Net was loaded.");
        } catch (Exception e) {
            e.printStackTrace();
            FormDigits.out("!!! Не удалось загрузить сеть из файла !!!");
        }
    }


    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public void setDecay(double decay) {
        this.decay = decay;
    }
}
