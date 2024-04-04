package network.gui;

import lombok.extern.slf4j.Slf4j;
import network.enums.Status;
import network.net.NeuralNetwork;

import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.plaf.nimbus.NimbusLookAndFeel;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.math.BigDecimal;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.Stream;

@Slf4j
public class DigitsFrame implements iWorkFrame {
    private static Status curStatus = Status.NOT_STARTED;

    private final int w = 28, h = w, scale = 32;

    private final int mThreadsCount = Runtime.getRuntime().availableProcessors();
    private final UnaryOperator<Double> sigmoid = x -> 1 / (1 + Math.exp(-x));
    private final UnaryOperator<Double> dsigmoid = y -> y * (1 - y);

    private NeuralNetwork nn;
    private ExecutorService initExecutors;

    private JFrame showFrame;

    private final int inputCount = w * h; // колличество входных данных.
    private final int outputsCount = 21; // колличество вариантов ответа.
    private final int[] sizes = new int[]{
            inputCount, // 784

            512, // 533
            256, // 137
            64, // 46

            outputsCount // 21
    };

    private double[] drawOutputs;
    private boolean showLRateInfo = true, trainBrakeFlag = false, showInitTestFrameFlag = false;

    private int mousePressed = 0, mx = 0, my = 0;
    private double[][] colors;

    private Long was;
    private int maxDigit = 0, success, epochNow;
    private double errors;
    private int[] correctAnswers;
    private double[][] inputs;
    private double[] drawInputs;

    private BufferedImage img;
    private List<Path> imagesFilesPaths;
    private JPanel drawPane, rightInfoPane, centerPane;
    private JButton initButton, trainButton, saveButton, loadButton;
    private JTextArea outArea;
    private JScrollPane outScrollPane;
    private JLabel imagesCountLabel, currentStatusLabel, progressLabel;
    private JTextField enterRightAnswerField;
    private JSpinner epoSpinner, bchSpinner, lnrSpinner, dcySpinner;

    private Thread trainThread;
    private boolean isActive = true;

    private final ArrayList<String> letters = new ArrayList<>();
    private final int BLACK_PIXEL = -16777216; // -16777216;
    private final int GRAY_PIXEL = -9500000; // -9650000;


    public DigitsFrame() {
        try {
            UIManager.setLookAndFeel(new NimbusLookAndFeel());
//			SwingUtilities.updateComponentTreeUI(frame);
        } catch (Exception e) {
            log.error("Couldn't get specified look and feel, for some reason: {}", e.getMessage());
        }

        new JFrame() {
            {
                setTitle("Prepares frame:");
                setMinimumSize(new Dimension(Toolkit.getDefaultToolkit().getScreenSize().height + 475, Toolkit.getDefaultToolkit().getScreenSize().height));
                setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                getContentPane().setBackground(Color.DARK_GRAY);

                centerPane = new JPanel(new BorderLayout(3, 3)) {
                    {
                        setOpaque(false);
                        setBorder(new EmptyBorder(3, 3, 3, 3));

                        JPanel upInfoPane = new JPanel(new BorderLayout(3, 3)) {
                            {
                                setOpaque(false);
                                setFocusable(false);
                                setBorder(BorderFactory.createCompoundBorder(
                                        new EmptyBorder(3, 3, 3, 3),
                                        BorderFactory.createLineBorder(Color.BLACK, 2, true)
                                ));

                                JPanel infoPane = new JPanel(new GridLayout(3, 3, 24, 3)) {
                                    {
                                        setOpaque(false);
                                        setBorder(new EmptyBorder(3, 3, 3, 3));

                                        currentStatusLabel = new JLabel("Current status: ") {
                                            {
                                                setForeground(Color.GRAY);
                                            }
                                        };

                                        imagesCountLabel = new JLabel("Images count: ") {
                                            {
                                                setForeground(Color.WHITE);
                                            }
                                        };

                                        progressLabel = new JLabel("Progress: ") {
                                            {
                                                setForeground(Color.WHITE);
                                            }
                                        };


                                        // количество эпох:
                                        SpinnerNumberModel epoModel = new SpinnerNumberModel(300d, 1d, 1000d, 1d);
                                        epoSpinner = new JSpinner(epoModel);

                                        // количество проходов за эпоху:
                                        SpinnerNumberModel bchModel = new SpinnerNumberModel(150d, 10d, 1000d, 1d);
                                        bchSpinner = new JSpinner(bchModel);

                                        // learning rate:
                                        SpinnerNumberModel lnrModel = new SpinnerNumberModel(0.012D, 0.001D, 1D, 0.001D);
                                        lnrSpinner = new JSpinner(lnrModel) {
                                            {setEditor(new JSpinner.NumberEditor(this, "0.000"));}};

                                        // learning decay:
                                        SpinnerNumberModel dcyModel = new SpinnerNumberModel(0.0000225D, 0.000001D, 0.01D, 0.000001D);
                                        dcySpinner = new JSpinner(dcyModel) {
                                            {setEditor(new JSpinner.NumberEditor(this, "0.0000000"));}};


                                        add(currentStatusLabel);
                                        add(imagesCountLabel);
                                        add(progressLabel);

                                        add(new JPanel(new BorderLayout(3, 3)) {{
                                            setOpaque(false);
                                            add(new JLabel("Epoch count:") {{
                                                setForeground(Color.WHITE);
                                            }});
                                            add(epoSpinner, BorderLayout.EAST);
                                        }});
                                        add(new JPanel(new BorderLayout(3, 3)) {{
                                            setOpaque(false);
                                            add(new JLabel("Batch count:") {{
                                                setForeground(Color.WHITE);
                                            }});
                                            add(bchSpinner, BorderLayout.EAST);
                                        }});
                                        add(new JLabel("n/a"));

                                        add(new JPanel(new BorderLayout(3, 3)) {{
                                            setOpaque(false);
                                            add(new JLabel("Learning rate:") {{
                                                setForeground(Color.WHITE);
                                            }});
                                            add(lnrSpinner, BorderLayout.EAST);
                                        }});
                                        add(new JPanel(new BorderLayout(3, 3)) {{
                                            setOpaque(false);
                                            add(new JLabel("Learning decay:") {{
                                                setForeground(Color.WHITE);
                                            }});
                                            add(dcySpinner, BorderLayout.EAST);
                                        }});
                                        add(new JLabel("n/a"));
                                    }
                                };

                                add(infoPane, BorderLayout.NORTH);
                            }
                        };

                        drawPane = new JPanel() {
                            @Override
                            public void paintComponent(Graphics g) {
                                if (img == null) {
                                    img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
                                }
                                updateImage();
                                g.drawImage(img, 0, 0, w * scale, h * scale, this);
                            }

                            private void updateImage() {
                                drawInputs = new double[w * h];
                                for (int i = 0; i < w; i++) {
                                    for (int j = 0; j < h; j++) {
                                        if (mousePressed > 0 && curStatus != Status.LEARNING) {
                                            double dist = (i - mx) * (i - mx) + (j - my) * (j - my);

                                            if (dist < 1) {
                                                dist = 1;
                                            }
                                            dist *= dist;

                                            if (mousePressed == 1) {
                                                colors[i][j] -= 0.1 / dist;
                                            } else if (mousePressed == 3) {
                                                colors[i][j] += 0.1 / dist;
                                            }

                                            if (colors[i][j] > 1) {
                                                colors[i][j] = 1D;
                                            }
                                            if (colors[i][j] < 0) {
                                                colors[i][j] = 0D;
                                            }
                                        }

                                        int color = (int) (colors[i][j] * 255);
                                        color = (color << 16) | (color << 8) | color;
                                        img.setRGB(i, j, color);
                                        drawInputs[i + j * w] = colors[i][j];
                                    }
                                }

                                if (mousePressed > 0 && curStatus != Status.LEARNING) {
                                    drawOutputs = nn.feedForward(drawInputs);
                                    maxDigit = 0;
                                    double maxDigitWeight = -1;
                                    for (int i = 0; i < outputsCount; i++) {
                                        if (drawOutputs[i] > maxDigitWeight) {
                                            maxDigitWeight = drawOutputs[i];
                                            maxDigit = i;
                                        }
                                    }
                                }

                            }

                            {
                                setOpaque(false);
                                setPreferredSize(new Dimension(w, h));
                                setFocusable(true);

                                addMouseListener(new MouseAdapter() {
                                    @Override
                                    public void mousePressed(MouseEvent e) {
                                        mousePressed = e.getButton();
                                    }

                                    @Override
                                    public void mouseReleased(MouseEvent e) {
                                        mousePressed = 0;
                                    }
                                });
                                addMouseMotionListener(new MouseMotionAdapter() {
                                    @Override
                                    public void mouseDragged(MouseEvent e) {
                                        mx = e.getX() / scale;
                                        my = e.getY() / scale;
                                    }

                                    @Override
                                    public void mouseMoved(MouseEvent e) {
                                        mx = e.getX() / scale;
                                        my = e.getY() / scale;
                                    }
                                });
                            }
                        };

                        rightInfoPane = new JPanel(new BorderLayout(3, 3)) {
                            @Override
                            public void paintComponent(Graphics g) {
                                g.setColor(Color.DARK_GRAY);
                                g.fillRect(0, 0, getWidth(), getHeight());
                                g.setFont(new Font("TimesRoman", Font.BOLD, 36));

                                int vertSpace = 40;
                                for (int i = 0; i < 10; i++) {
                                    if (maxDigit == i) {
                                        g.setColor(Color.RED);
                                    } else {
                                        g.setColor(Color.GRAY);
                                    }
                                    g.drawString(i + ":", 20, 34 + i * vertSpace);

                                    if (drawOutputs != null) {
                                        Color rectColor = new Color(0, (float) drawOutputs[i], 0);
                                        int rectWidth = (int) (drawOutputs[i] * 100);
                                        g.setColor(rectColor);
                                        g.fillRoundRect(65, 10 + i * vertSpace, rectWidth, 26, 3, 3);
                                        g.setColor(Color.BLACK);
                                        g.drawRoundRect(65, 10 + i * vertSpace, rectWidth, 26, 3, 3);
                                    }
                                }

                                for (int i = 0; i < letters.size(); i++) {
                                    if (maxDigit == i + letters.size() - 1) {
                                        g.setColor(Color.RED);
                                        setTitle("Result draw frame: " + maxDigit + " (" + letters.get(i) + ")");
                                    } else {
                                        g.setColor(Color.GRAY);
                                    }
                                    g.drawString(letters.get(i) + ":", 20, 442 + i * vertSpace);

                                    if (drawOutputs != null) {
                                        Color rectColor = new Color(0, (float) drawOutputs[i], 0);
                                        int rectWidth = (int) (drawOutputs[i] * 100);
                                        g.setColor(rectColor);
                                        g.fillRoundRect(65, 420 + i * vertSpace, rectWidth, 26, 3, 3);

                                        g.setColor(Color.BLACK);
                                        g.drawRoundRect(65, 420 + i * vertSpace, rectWidth, 26, 3, 3);
                                    }
                                }

                            }

                            {
                                setPreferredSize(new Dimension(250, 0));

                                JPanel handleCorrectFieldPane = new JPanel(new BorderLayout(3, 3)) {
                                    {
                                        setOpaque(false);

                                        enterRightAnswerField = new JTextField() {
                                            {
                                                addKeyListener(new KeyAdapter() {
                                                    @Override
                                                    public void keyPressed(KeyEvent e) {
                                                        if (e.getKeyCode() == KeyEvent.VK_SPACE) {
                                                            setText(null);
                                                        }
                                                    }

                                                    @Override
                                                    public void keyReleased(KeyEvent e) {
                                                        if (e.getKeyCode() == KeyEvent.VK_SPACE) {
                                                            setText(null);
                                                        }
                                                    }
                                                });
                                            }
                                        };

                                        JButton correctNetBtn = new JButton("Correct") {
                                            {
                                                setFocusable(false);
                                                addActionListener(new ActionListener() {
                                                    @Override
                                                    public void actionPerformed(ActionEvent e) {

                                                        if (enterRightAnswerField.getText().isBlank()) {
                                                            return;
                                                        }

                                                        if (curStatus != Status.LEARNING && nn != null) {
                                                            out("Learn this sign is '" + enterRightAnswerField.getText() + "'");
                                                            String hca = enterRightAnswerField.getText().trim();
                                                            enterRightAnswerField.setText(null);
                                                            enterRightAnswerField.setText("\r");
                                                            backPropagationBySignHandle(hca);
                                                        } else {
                                                            out("Current curStatus: " + curStatus);
                                                        }

                                                        transferFocus();
                                                    }
                                                });
                                            }
                                        };

                                        add(enterRightAnswerField, BorderLayout.CENTER);
                                        add(correctNetBtn, BorderLayout.EAST);
                                    }
                                };

                                add(handleCorrectFieldPane, BorderLayout.SOUTH);
                            }
                        };

                        drawPane.setEnabled(false);
                        rightInfoPane.setEnabled(false);

                        add(upInfoPane, BorderLayout.NORTH);
                        add(drawPane, BorderLayout.CENTER);
                        add(rightInfoPane, BorderLayout.EAST);
                    }
                };

                outArea = new JTextArea() {
                    {
                        setEditable(false);
                        setLineWrap(true);
                        setWrapStyleWord(true);
                        setBackground(Color.BLACK);
                        setForeground(Color.green);
                        setBorder(new EmptyBorder(3, 6, 0, 3));
                        append("Log:\n*** *** ***\n");
                    }
                };
                outScrollPane = new JScrollPane(outArea) {
                    {
                        setBorder(null);
                        setViewportBorder(null);
                        setPreferredSize(new Dimension(250, 0));

                        setVerticalScrollBarPolicy(ScrollPaneConstants.VERTICAL_SCROLLBAR_AS_NEEDED);
                        getVerticalScrollBar().setUnitIncrement(h * scale / 10);

                        setAutoscrolls(true);
                        getVerticalScrollBar().setAutoscrolls(true);
                    }
                };

                JPanel downButtonsPane = new JPanel(new GridLayout(1, 8, 3, 3)) {
                    {
                        setOpaque(false);
                        setBorder(new EmptyBorder(1, 1, 3, 1));

                        initButton = new JButton("Reset net") {
                            {
                                setFocusable(false);
                                setBackground(Color.orange.darker());
                                addActionListener(new ActionListener() {
                                    @Override
                                    public void actionPerformed(ActionEvent e) {
                                        int a = JOptionPane.showConfirmDialog(null, "Are You shure?", "Reset net?", JOptionPane.OK_CANCEL_OPTION);
                                        if (a == 0) {
                                            if (trainThread != null) {
                                                while (trainThread.isAlive()) {
                                                    trainBrakeFlag = true;
                                                }
                                            }
                                            reinit();
                                        }
                                    }
                                });
                            }
                        };

                        trainButton = new JButton("Start/stop train") {
                            {
                                setFocusable(false);
                                setBackground(Color.CYAN.darker());
                                addActionListener(new ActionListener() {
                                    @Override
                                    public void actionPerformed(ActionEvent e) {
                                        if (trainThread != null && trainThread.isAlive()) {
                                            trainBrakeFlag = true;
                                            setEnabled(imagesFilesPaths != null);
                                            centerPane.setEnabled(true);
                                            rightInfoPane.setEnabled(true);
                                        } else {
                                            centerPane.setEnabled(false);
                                            rightInfoPane.setEnabled(false);
                                            train();
                                        }
                                    }
                                });

                                setEnabled(imagesFilesPaths != null);
                            }
                        };

                        saveButton = new JButton("Save net") {
                            {
                                setFocusable(false);
                                setBackground(Color.GREEN);
                                addActionListener(new ActionListener() {
                                    @Override
                                    public void actionPerformed(ActionEvent e) {
                                        int a = JOptionPane.showConfirmDialog(null, "Are You sure?", "Save this net?", JOptionPane.OK_CANCEL_OPTION);
                                        if (a == 0) {
                                            String name = JOptionPane.showInputDialog(null, "Как ты хочешь назвать файл?", "Имя файла:", JOptionPane.QUESTION_MESSAGE);
                                            nn.save(name);
                                        }
                                    }
                                });
                            }
                        };

                        loadButton = new JButton("Load net") {
                            {
                                setFocusable(false);
                                setBackground(Color.YELLOW);
                                addActionListener(new ActionListener() {
                                    @Override
                                    public void actionPerformed(ActionEvent e) {

                                        JFileChooser datChooser = new JFileChooser("./");
                                        datChooser.setDialogTitle("Выбери файл сети для загрузки:");
                                        datChooser.setDialogType(JFileChooser.OPEN_DIALOG);
                                        datChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
                                        datChooser.setMultiSelectionEnabled(false);
                                        datChooser.showOpenDialog(null);

                                        if (datChooser.getSelectedFile() != null) {
                                            nn.load(datChooser.getSelectedFile().toPath());
                                        }
                                    }
                                });
                            }
                        };

                        add(initButton);
                        add(trainButton);
                        add(saveButton);
                        add(loadButton);
                    }
                };

                add(centerPane, BorderLayout.CENTER);
                add(outScrollPane, BorderLayout.WEST);
                add(downButtonsPane, BorderLayout.SOUTH);

                setFocusable(true);
                addKeyListener(new KeyAdapter() {
                    @Override
                    public void keyPressed(KeyEvent e) {
                        if (e.getKeyCode() == KeyEvent.VK_SPACE) {
                            clearCanvas();
                        }
                    }
                });

                pack();
                setLocationRelativeTo(null);
                setVisible(true);
            }
        };

        Thread.startVirtualThread(() -> {
            while (isActive) {
                if (mousePressed > 0 && centerPane.isEnabled()) {
                    centerPane.repaint();
                }

                if (curStatus.equals(Status.AWAIT)) {
                    updateInfo();
                }

                try {
                    Thread.sleep(13);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        });

        reinit();
    }

    private void reinit() {
        curStatus = Status.INITIALIZATION;
//        decay = new BigDecimal((double) lnrSpinner.getValue() * (double) epoSpinner.getValue() * (double) dcySpinner.getValue()); // скорость затухания модификатора обучения (0.00005 / 0.000034).

        letters.addAll(Arrays.stream(new String[] {"A", "B", "C", "E", "H", "K", "M", "P", "T", "X", "Y"}).toList());

        nn = new NeuralNetwork(
                this,
                (double) lnrSpinner.getValue(),
                BigDecimal.valueOf((double) lnrSpinner.getValue() * (int) epoSpinner.getValue() * (double) dcySpinner.getValue()).doubleValue(), sigmoid, dsigmoid, sizes);
        nn.setShowLRateInfo(showLRateInfo);

        clearCanvas();

        out(String.format("""
                Начата инициализация нейросети с параметрами:
                Потоков: %h
                Эпох: %d
                Батч: %d
                LR: %s
                Decay: %s
                Вход: %s
                Вывод: %s
                """, mThreadsCount, epoSpinner.getValue(), bchSpinner.getValue(), lnrSpinner.getValue(),
                String.format("%,.08f", BigDecimal.valueOf((double) lnrSpinner.getValue() * (int) epoSpinner.getValue() * (double) dcySpinner.getValue()).doubleValue()), inputCount, outputsCount));

        was = System.nanoTime();
        try {
            imagesFilesPaths = listFiles(Paths.get("./train/"));
            out("Exists images: " + imagesFilesPaths.size());

            BufferedImage[] images = new BufferedImage[imagesFilesPaths.size()];
            correctAnswers = new int[imagesFilesPaths.size()];
            inputs = new double[imagesFilesPaths.size()][w * h];

            out("Reading right answers...");
            initExecutors = Executors.newFixedThreadPool(mThreadsCount);
            for (int i = 0; i < mThreadsCount; i++) {
                int tmp = i;
                initExecutors.execute(new Runnable() {
                    @Override
                    public void run() {
                        try {
                            initAI(tmp, mThreadsCount);
                        } catch (IOException e) {
                            log.error("Error (019): {}", e.getMessage());
                        }
                    }

                    void initAI(int chunkCount, int chunkExists) throws IOException {
                        int from = imagesFilesPaths.size() / chunkExists * chunkCount;
                        int by = imagesFilesPaths.size() / chunkExists * (chunkCount + 1) < imagesFilesPaths.size() && chunkCount + 1 == chunkExists ?
                                imagesFilesPaths.size() : imagesFilesPaths.size() / chunkExists * (chunkCount + 1);

                        out("Thread #" + chunkCount + " get chunk from " + (imagesFilesPaths.size() / chunkExists * chunkCount) + " to " + by);

                        for (int i = from; i < by; i++) {
                            correctAnswers[i] = getNumberOfImageId(i);

                            images[i] = wrapToCanvas(imagesFilesPaths.get(i).toFile());
                            images[i] = removeGraySpots(images[i]);
//                        images[i] = createBWImage(images[i]);
//                        images[i] = removeDirtVoid(images[i]);

                            for (int x = 0; x < w; x++) {
                                for (int y = 0; y < h; y++) {
                                    inputs[i][x + y * w] = (images[i].getRGB(x, y) & 0xff) / 255.0;
                                }
                            }

                            if (showInitTestFrameFlag) {
                                showTestDialogFrame(i, inputs); // 5884
                            }
                        }
                    }
                });
            }
            initExecutors.shutdown();

        } catch (IOException e) {
            log.error("Error (016): {}", e.getMessage());
        }

        try {
            while (!initExecutors.awaitTermination(250, TimeUnit.MILLISECONDS)) {
            }
        } catch (InterruptedException e) {
            log.error("Error (005): {}", e.getMessage());
        }

        trainButton.setEnabled(true);
        out("Init finished for a " + ((System.nanoTime() - was) / 1000000000.0) + " sec.\n");
        curStatus = Status.AWAIT;
    }

    private void showTestDialogFrame(int imIndex, double[][] inputs) {
        if (showFrame != null && showFrame.isVisible()) {
            return;
        }

        showFrame = new JFrame() {
            final BufferedImage image = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);

            {
                setTitle("Image of number " +
                        (getNumberOfImageId(imIndex) > 9 ? getLetterOfImageId(imIndex) : getNumberOfImageId(imIndex)) +
                        " (index: " + getNumberOfImageId(imIndex) + ")");

                setPreferredSize(new Dimension(w * scale * 2 + 18, h * scale));
                setDefaultCloseOperation(JDialog.DISPOSE_ON_CLOSE);

                JPanel srcImagePane = new JPanel() {
                    @Override
                    public void paint(Graphics g) {
                        try {
                            g.drawImage(ImageIO.read(imagesFilesPaths.get(imIndex).toFile()), 0, 0, getWidth(), getHeight(), this);
                            g.setColor(Color.GREEN);
                            g.drawRoundRect(3, 3, getWidth() - 6, getHeight() - 6, 6, 6);

                            g.dispose();
                        } catch (IOException e) {
                            log.error("Error (008): {}", e.getMessage());
                        }
                    }

                    {
                        setPreferredSize(new Dimension(w * scale, h * scale));
                    }
                };

                JPanel dstImagePane = new JPanel() {
                    @Override
                    public void paint(Graphics g) {

                        for (int x = 0; x < w; x++) {
                            for (int y = 0; y < h; y++) {
                                int color = (int) (inputs[imIndex][x + y * w] * 255);
                                color = (color << 16) | (color << 8) | color;
                                image.setRGB(x, y, color);
                            }
                        }

                        g.drawImage(image, 0, 0, getWidth(), getHeight(), this);

                        g.setColor(Color.BLUE);
                        g.drawRoundRect(3, 3, getWidth() - 6, getHeight() - 6, 6, 6);

                        g.dispose();
                    }

                    {
                        setPreferredSize(new Dimension(w * scale, h * scale));
                    }
                };

                add(srcImagePane, BorderLayout.WEST);
                add(dstImagePane, BorderLayout.EAST);

                pack();
                setLocationRelativeTo(null);
                setVisible(true);
            }
        };
    }

    private void train() {
        curStatus = Status.LEARNING;

        nn.setLearningRate((double) lnrSpinner.getValue());
        nn.setDecay(BigDecimal.valueOf((double) lnrSpinner.getValue() * (int) epoSpinner.getValue() * (double) dcySpinner.getValue()).doubleValue());

        trainThread = new Thread(() -> {
            try {
                out("\nTraining started!");
                was = System.nanoTime();
                epochNow = 0;
                trainBrakeFlag = false;

                for (int i = 0; i < (int) epoSpinner.getValue(); i++) {
                    success = 0;
                    errors = 0;

                    for (int j = 0; j < (int) bchSpinner.getValue(); j++) {
                        // берем случайное изображение:
                        int randomPicIndex = (int) (Math.random() * imagesFilesPaths.size());
                        int correctAnswer = correctAnswers[randomPicIndex]; // верный ответ.

                        // устанавливаем 1 в правильный выход нейросети:
                        double[] targets = new double[outputsCount];
                        targets[correctAnswer] = 1;

                        // скармливаем массив пискелей выбранного изображения в нейросеть:
                        double[] outputs = nn.feedForward(inputs[correctAnswer]);

                        // вычисляем какой выход "загорелся" как верный ответ:
                        int maxDigit = 0;
                        double maxDigitWeight = -1;
                        for (int k = 0; k < outputs.length; k++) {
                            if (outputs[k] > maxDigitWeight) {
                                maxDigitWeight = outputs[k];
                                maxDigit = k;
                            }

                            errors += (targets[k] - outputs[k]) * (targets[k] - outputs[k]);
                        }

                        // проверяем, корректно ли угадано число:
                        if (correctAnswer == maxDigit) {
                            success++;
                        }

                        // отправляем данные для коррекции весов:
                        nn.backpropagation(targets);
                    }

                    out("(Ep.: " + (i + 1) + "; Cor.: " + success + "; Err.: " + errors + ")");
                    epochNow++;
                    if (errors < 1) {
                        out("ERRORS COUNT LESS THAN ZERO. THAN ENOUGHT.");
                        trainBrakeFlag = true;
                    }
                    if (trainBrakeFlag) {
                        break;
                    }
                }

                out("Training finished for a " + ((System.nanoTime() - was) / 1000000000.0 / 60.0) + " min.");

            } catch (Exception e) {
                log.error("Error (007): {}", e.getMessage());
            } finally {
                trainButton.setEnabled(true);
                curStatus = Status.AWAIT;
                drawPane.setEnabled(true);
                rightInfoPane.setEnabled(true);
            }

        });
        trainThread.start();
    }

    private void backPropagationBySignHandle(String correctAnswer) {
        int correctNumber;

        try {
            System.out.println("Seek '" + correctAnswer.toUpperCase() + "' into letters-array...");
            correctNumber = letters.contains(correctAnswer) ? letters.indexOf(correctAnswer) + 10 : Integer.parseInt(correctAnswer);

            // устанавливаем 1 в правильный выход нейросети:
            double[] targets = new double[outputsCount];
            targets[correctNumber] = 1;

            // отправляем данные для коррекции весов:
            out("Correct sign to index " + correctNumber + (correctNumber > 9 ? " (" + letters.get(correctNumber) + ")" : "."));
            nn.backpropagation(targets);
        } catch (Exception e) {
            log.error("Error (009): {}", e.getMessage());
        }
    }

    private int getNumberOfImageId(int id) {
        return Integer.parseInt(imagesFilesPaths.get(id).getParent().toFile().getName().split("\\.")[0]);
    }

    private String getLetterOfImageId(int id) {
        return letters.get(getNumberOfImageId(id) - 10);
    }

    public List<Path> listFiles(Path path) throws IOException {
        try (Stream<Path> stream = Files.walk(path)) {
            return stream
                    .filter(Files::isRegularFile)
                    .collect(Collectors.toList());
        } catch (Exception e) {
            log.error("Error (014): {}", e.getMessage());
            return Collections.emptyList();
        }
    }

    private void clearCanvas() {
        colors = new double[w][h];
        for (double[] row : colors) {
            Arrays.fill(row, 1);
        }

        drawPane.repaint();
    }


    private BufferedImage wrapToCanvas(File imageFile) {
        BufferedImage im = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);

        try {
            Graphics2D g = im.createGraphics();
//            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
//            g.setRenderingHint(RenderingHints.KEY_DITHERING, RenderingHints.VALUE_DITHER_ENABLE);
//            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR);
//            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);

            g.drawImage(ImageIO.read(imageFile), 0, 0, w, h, null);

//            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);
//            g.setRenderingHint(RenderingHints.KEY_DITHERING, RenderingHints.VALUE_DITHER_DISABLE);
            g.dispose();
        } catch (IOException e) {
            log.error("Error (010): {}", e.getMessage());
        }

        return im;
    }

    private BufferedImage removeGraySpots(BufferedImage grayImage) {

        for (int y = 0; y < grayImage.getHeight(); y++) {
            for (int x = 0; x < grayImage.getWidth(); x++) {
                if (grayImage.getRGB(x, y) <= GRAY_PIXEL) {
                    continue;
                }

                try {
                    grayImage.setRGB(x, y, Color.WHITE.getRGB());
                } catch (Exception e) {
                    System.err.println("Fail removing gray pixel: " + e.getMessage());
                    log.error("Error (011): {}", e.getMessage());
                }

            }
        }

        return grayImage;
    }

    private BufferedImage createBWImage(BufferedImage srcImage) {
        BufferedImage image = new BufferedImage(srcImage.getWidth(), srcImage.getHeight(), BufferedImage.TYPE_BYTE_BINARY);
//        BufferedImage image = new BufferedImage(srcImage.getWidth(), srcImage.getHeight(), BufferedImage.TYPE_BYTE_INDEXED);
//        BufferedImage image = new BufferedImage(srcImage.getWidth(), srcImage.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        image.createGraphics().drawImage(srcImage, 0, 0, null);
//        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
//        g.setRenderingHint(RenderingHints.KEY_DITHERING, RenderingHints.VALUE_DITHER_ENABLE);
//        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR);
//        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);

        return image;
    }

    private BufferedImage removeDirtVoid(BufferedImage dirtyImage) {
        int[] voidMatrix = new int[4];

        for (int y = 0; y < dirtyImage.getHeight(); y++) {
            for (int x = 0; x < dirtyImage.getWidth(); x++) {
                if (dirtyImage.getRGB(x, y) > GRAY_PIXEL) {
                    continue;
                }

                try {
                    voidMatrix[0] = x + 1 < dirtyImage.getWidth() ? (dirtyImage.getRGB(x + 1, y) == BLACK_PIXEL ? 1 : 0) : 0;
                    voidMatrix[1] = x - 1 >= 0 ? (dirtyImage.getRGB(x - 1, y) == BLACK_PIXEL ? 1 : 0) : 0;
                    voidMatrix[2] = y + 1 < dirtyImage.getHeight() ? (dirtyImage.getRGB(x, y + 1) == BLACK_PIXEL ? 1 : 0) : 0;
                    voidMatrix[3] = y - 1 >= 0 ? (dirtyImage.getRGB(x, y - 1) == BLACK_PIXEL ? 1 : 0) : 0;

                    if (voidMatrix[0] + voidMatrix[1] + voidMatrix[2] + voidMatrix[3] <= 1) {
                        dirtyImage.setRGB(x, y, Color.WHITE.getRGB());
                    }
                } catch (Exception e) {
                    System.err.println("Fail clean void pixel: " + e.getMessage());
                    log.error("Error (012): {}", e.getMessage());
                }

            }
        }

        return dirtyImage;
    }


    private void scrollDown() {
        new Thread(() -> {
            try {
                Thread.sleep(500);
            } catch (Exception e) {/* SLEEP IGNORE */}
            int current = outScrollPane.getVerticalScrollBar().getValue();
            int max = outScrollPane.getVerticalScrollBar().getMaximum();
            if (current < max - 250) {
                outScrollPane.getVerticalScrollBar().setValue(max);
            }
        }).start();
    }

    private void updateInfo() {
        try {
            currentStatusLabel.setText("Current status: " + curStatus);
            currentStatusLabel.setForeground(curStatus == Status.AWAIT ? Color.GRAY : curStatus == Status.INITIALIZATION ? Color.YELLOW : Color.GREEN);
            imagesCountLabel.setText("Images count: " + imagesFilesPaths.size());
            progressLabel.setText("Progress: " + ((int) (1f / (int)epoSpinner.getValue() * epochNow * 100f)) + "%");
        } catch (NullPointerException e) {
            log.error("Error (013): {}", e.getMessage());
        } catch (Exception e) {
            log.error("Error (014): {}", e.getMessage());
        }
    }

    public void out(String mes) {
        outArea.append(mes + "\n");
        scrollDown();
    }
}
