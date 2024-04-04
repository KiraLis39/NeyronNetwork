package network.gui;

import lombok.extern.slf4j.Slf4j;
import network.CustomPoint;
import network.net.NeuralNetwork;

import javax.swing.*;
import javax.swing.border.EtchedBorder;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.FocusAdapter;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;
import java.util.ArrayList;
import java.util.function.UnaryOperator;

import static javax.swing.SwingConstants.VERTICAL;


@Slf4j
public class DotsFrame extends JFrame implements iWorkFrame {
    private final int w = 1440, h = 900;
    private int drawQuality = 24;

    private final double pointDiameter = 16D;
    private double learningRate = 0.0020D;
    private double decay = 0.0025D;
    private int workSpeed = 1000;

    private final UnaryOperator<Double> sigmoid = x -> 1 / (1 + Math.exp(-x));
    private final UnaryOperator<Double> dsigmoid = y -> y * (1 - y);
    private final int[] sizes = new int[] {
            2, // 2
            4, // -
            16, // 8
            32, // 10
            2, // 2
    };

    private final NeuralNetwork nn;
    private BufferedImage background;
    private final List<CustomPoint> markers = new ArrayList<>();

    private static JPanel drawPane, controlPane;
    private JSpinner dQualitySpinner;

    private boolean warnMes = false, renderOn = false, useGradient = true, isActive = true;

    private final Color color_01 = Color.ORANGE, color_02 = Color.GREEN;

    public DotsFrame() {
//        if (w % drawQuality != 0 || h % drawQuality != 0) {
//            warnMes = true;
//        }

        nn = new NeuralNetwork(this, learningRate, decay, sigmoid, dsigmoid, sizes);

        setPreferredSize(new Dimension(w, h));
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        drawPane = new JPanel() {
            @Override
            public void paintComponent(Graphics g) {
                if (warnMes) {
                    g.setColor(Color.RED);
                    g.drawString("Width and Height must be multiplied by " + drawQuality,
                            getWidth() / 2 - 32, getHeight() / 2);
                    return;
                }

                Graphics2D g2D = (Graphics2D) g;
                render(g2D);

                g2D.drawImage(background, 0, 0, getWidth(), getHeight(), this);

                for (CustomPoint p : markers) {
                    g2D.setColor(p.color());
                    g2D.fillOval(
                            (int) (p.x() - pointDiameter / 2D),
                            (int) (p.y() - pointDiameter / 2D),
                            (int) pointDiameter,
                            (int) pointDiameter);
                }

                g2D.setColor(Color.RED);
                g2D.drawString("Rendering:", 12, 30);
                g2D.drawString(renderOn ? "ON" : "OFF", 96, 30);

                g2D.drawString("Delete last:", 12, 55);
                g2D.drawString("Backspace", 96, 55);

                g2D.drawString("Clear all:", 12, 75);
                g2D.drawString("F3", 96, 75);

                g2D.dispose();
            }

            private void render(Graphics2D g2D) {
                if (drawQuality <= 8) {
                    g2D.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_SPEED);
                    g2D.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);
                    g2D.setRenderingHint(RenderingHints.KEY_FRACTIONALMETRICS, RenderingHints.VALUE_FRACTIONALMETRICS_OFF);
                    g2D.setRenderingHint(RenderingHints.KEY_STROKE_CONTROL, RenderingHints.VALUE_STROKE_DEFAULT);
                    renderOn = false;
                    return;
                }

                g2D.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                g2D.setRenderingHint(RenderingHints.KEY_FRACTIONALMETRICS, RenderingHints.VALUE_FRACTIONALMETRICS_ON);
                g2D.setRenderingHint(RenderingHints.KEY_STROKE_CONTROL, RenderingHints.VALUE_STROKE_NORMALIZE);
                g2D.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
                renderOn = true;
            }

            {
                addMouseListener(new MouseAdapter() {
                    @Override
                    public void mousePressed(MouseEvent e) {
                        CustomPoint aNewMarker = new CustomPoint(
                                e.getPoint().getX(),
                                e.getPoint().getY(),
                                e.getButton() == 3 ? 1 : 0,
                                e.getButton() == 3 ? color_01 : color_02);

                        for (CustomPoint existMarker : markers) {
                            if (aNewMarker.x() >= existMarker.x() - pointDiameter / 2d
                                    && aNewMarker.x() <= existMarker.x() + pointDiameter / 2d
                                    && aNewMarker.y() >= existMarker.y() - pointDiameter / 2d
                                    && aNewMarker.y() <= existMarker.y() + pointDiameter / 2d
                            ) {
                                return;
                            }
                        }
                        markers.addLast(aNewMarker);
                    }
                });

                addKeyListener(new KeyAdapter() {
                    @Override
                    public void keyPressed(KeyEvent e) {
                        switch (KeyEvent.getKeyText(e.getKeyCode())) {
                            case "Backspace" -> markers.removeLast();
                            case "F3" -> markers.clear();
                            default -> log.info("KeyEvent.getKeyText(e.getKeyCode()): {}", KeyEvent.getKeyText(e.getKeyCode()));
                        }
                    }
                });
            }
        };

        controlPane = new JPanel(new GridLayout(0, 12, 3, 3)) {
            {
                SpinnerModel sm = new SpinnerNumberModel(learningRate, 0.0020D, 1.0000D, 0.0005D);
                JSpinner lRateSpiner = new JSpinner(sm) {
                    {
                        setToolTipText("Learning rate");
                        setFocusable(false);
                        setRequestFocusEnabled(false);
                        addChangeListener(e -> learningRate = (Double) getModel().getValue());
                        setEditor(new JSpinner.NumberEditor(this, "0.0000"));
                    }
                };


                SpinnerModel sm2 = new SpinnerNumberModel(workSpeed, 1, 100_000, 1000);
                JSpinner wSpeedSpiner = new JSpinner(sm2) {
                    {
                        setToolTipText("Work speed");
                        setFocusable(false);
                        setRequestFocusEnabled(false);
                        addChangeListener(e -> workSpeed = (int) getModel().getValue());
                    }
                };

                SpinnerModel sm3 = new SpinnerNumberModel(drawQuality, 2, h, 2);
                dQualitySpinner = new JSpinner(sm3) {
                    {
                        setToolTipText("Draw hardness (quality)");
                        setFocusable(false);
                        setRequestFocusEnabled(false);
                        addChangeListener(e -> recreateBackground());
                    }
                };

                add(new JLabel("learningRate: ") {{setHorizontalAlignment(RIGHT); setBorder(new EtchedBorder(EtchedBorder.LOWERED)); setFocusable(false);}});
                add(lRateSpiner);
                add(new JSeparator(VERTICAL));
                add(new JLabel("workSpeed: ") {{setHorizontalAlignment(RIGHT); setBorder(new EtchedBorder(EtchedBorder.LOWERED)); setFocusable(false);}});
                add(wSpeedSpiner);
                add(new JSeparator(VERTICAL));
                add(new JLabel("drawQuality: ") {{setHorizontalAlignment(RIGHT); setBorder(new EtchedBorder(EtchedBorder.LOWERED)); setFocusable(false);}});
                add(dQualitySpinner);

                setFocusable(false);
            }
        };

        add(drawPane, BorderLayout.CENTER);
        add(controlPane, BorderLayout.SOUTH);

        pack();
        setLocationRelativeTo(null);
        setVisible(true);

        recreateBackground();

        drawPane.setFocusable(true);
        drawPane.setRequestFocusEnabled(true);
        drawPane.requestFocusInWindow();

        Thread.startVirtualThread(() -> {
            while (isActive) {
                if (!markers.isEmpty()) {
                    recheck();
                }
                drawPane.repaint();

                try {
                    Thread.sleep(13);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        });
    }

    private void recreateBackground() {
        drawQuality = h / 4 / (int) dQualitySpinner.getModel().getValue();
        background = new BufferedImage(w / drawQuality, h / drawQuality, BufferedImage.TYPE_INT_RGB);
        log.info("Background width: {}, height: {}", background.getWidth(), background.getHeight());
    }

    private void recheck() {
        try {
            for (int k = 0; k < workSpeed; k++) {
                if (markers.isEmpty()) {
                    break; // if pressed F3
                }

                CustomPoint p = markers.get((int) (Math.random() * markers.size()));
                double nx = p.x() / w;
                double ny = p.y() / h;
                nn.feedForward(new double[]{nx, ny});

                double[] targets = new double[2];
                if (p.type() == 0) {
                    targets[0] = 1;
                } else {
                    targets[1] = 1;
                }

                // отправляем данные для коррекции весов:
                nn.backpropagation(targets);
            }

            for (int i = 0; i < w / drawQuality; i++) {
                for (int j = 0; j < h / drawQuality; j++) {
                    double nx = (double) i / w * drawQuality;
                    double ny = (double) j / h * drawQuality;

                    double[] outputs = nn.feedForward(new double[]{nx, ny}); // Кормим нейросеть данными...
                    background.setRGB(i, j, getColor(outputs)); // и рисуем цвет по полученному от сети ответу!
                }
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            log.error("Error (032): {}", e.getMessage());
        }
    }

    private int getColor(double[] outputs) {
        double green = Math.max(0, Math.min(1, outputs[0] - outputs[1] + 0.25)); // от 0 до 1

        if (useGradient) {
            double red = 127;
            double blue = 1 - green; // либо 1, либо 0

            green = 0.3d + green * 0.5d;
            blue = 0.5d + blue * 0.5d;
            return (((int)red << 16) | ((int)(green * 255) << 8) | (int)(blue * 255));
        } else {
            return green > 0.51d ? color_01.darker().getRGB() : green < 0.49d ? color_02.darker().getRGB() : Color.WHITE.getRGB();
        }
    }
}
