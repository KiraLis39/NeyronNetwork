package network.gui;

import lombok.extern.slf4j.Slf4j;
import network.CustomPoint;
import network.net.NeuralNetwork;

import javax.swing.*;
import javax.swing.border.EtchedBorder;
import java.awt.*;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.util.List;
import java.util.ArrayList;
import java.util.function.UnaryOperator;

import static javax.swing.SwingConstants.VERTICAL;


@Slf4j
public class DotsFrame extends JFrame {
    private final int w = 1440, h = 900;
    private int drawQuality = 24;

    private final double pointDiameter = 16D;
    private int workSpeed = 1000;

    private final NeuralNetwork nn;
    private BufferedImage background, markersLayer;
    private final List<CustomPoint> markers = new ArrayList<>();

    private static JPanel drawPane, controlPane;
    private JSpinner dQualitySpinner;

    private boolean renderOn = false, useGradient = false, isActive = true;

    private final Color color_01 = Color.ORANGE, color_02 = Color.GREEN;

    public DotsFrame() {
        nn = new NeuralNetwork();

        setPreferredSize(new Dimension(w, h));
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setResizable(false);

        drawPane = new JPanel() {
            @Override
            public void paintComponent(Graphics g) {
                Graphics2D g2D = (Graphics2D) g;
                render(g2D);

                g2D.drawImage(background, 0, 0, getWidth(), getHeight(), this);
                g2D.drawImage(markersLayer, 0, 0, getWidth(), getHeight(), this);

                g2D.setColor(Color.RED);
                g2D.drawString("Rendering:", 12, 30);
                g2D.drawString(renderOn ? "ON" : "OFF", 96, 30);

                g2D.drawString("Gradient:", 12, 50);
                g2D.drawString(useGradient ? "ON" : "OFF", 96, 50);

                g2D.drawString("Delete last:", 12, 75);
                g2D.drawString("Backspace", 96, 75);

                g2D.drawString("Clear all:", 12, 95);
                g2D.drawString("F3", 96, 95);

                g2D.drawString("Use gradient:", 12, 115);
                g2D.drawString("F5", 96, 115);

                g2D.dispose();
            }

            {
                setFocusable(true);
                setRequestFocusEnabled(true);
                requestFocusInWindow();

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
                        createMarkersLayer();
                    }
                });

                addKeyListener(new KeyAdapter() {
                    @Override
                    public void keyPressed(KeyEvent e) {
                        switch (KeyEvent.getKeyText(e.getKeyCode())) {
                            case "Backspace" -> {
                                markers.removeLast();
                                createMarkersLayer();
                            }
                            case "F3" -> {
                                markers.clear();
                                createMarkersLayer();
                            }
                            case "F5" -> useGradient = !useGradient;
                            default -> log.info("KeyEvent.getKeyText(e.getKeyCode()): {}", KeyEvent.getKeyText(e.getKeyCode()));
                        }
                    }
                });
            }
        };

        controlPane = new JPanel(new GridLayout(0, 12, 3, 3)) {
            {
                SpinnerModel sm = new SpinnerNumberModel(nn.getLearningRate(), 0.0020D, 1.0000D, 0.0005D);
                JSpinner lRateSpiner = new JSpinner(sm) {
                    {
                        setToolTipText("Learning rate");
                        setFocusable(false);
                        setRequestFocusEnabled(false);
                        addChangeListener(e -> nn.setLearningRate((Double) getModel().getValue()));
                        setEditor(new JSpinner.NumberEditor(this, "0.0000"));
                    }
                };


                SpinnerModel sm2 = new SpinnerNumberModel(workSpeed, 1, 100_000, 1000);
                JSpinner wSpeedSpinner = new JSpinner(sm2) {
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
                add(wSpeedSpinner);
                add(new JSeparator(VERTICAL));
                add(new JLabel("drawQuality: ") {{setHorizontalAlignment(RIGHT); setBorder(new EtchedBorder(EtchedBorder.LOWERED)); setFocusable(false);}});
                add(dQualitySpinner);

                setFocusable(false);
            }
        };

        add(drawPane, BorderLayout.CENTER);
        add(controlPane, BorderLayout.SOUTH);

        recreateBackground();

        pack();
        setLocationRelativeTo(null);
        setVisible(true);

        createMarkersLayer();

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

    private void createMarkersLayer() {
        log.info("Пересоздание слоя маркеров...");
        markersLayer = new BufferedImage(drawPane.getWidth(), drawPane.getHeight(), BufferedImage.TYPE_INT_ARGB);
        for (CustomPoint point : markers) {
            Graphics2D g2D = (Graphics2D) markersLayer.getGraphics();
            g2D.setColor(point.color());
            g2D.fillOval(
                    (int) (point.x() - pointDiameter / 2D),
                    (int) (point.y() - pointDiameter / 2D),
                    (int) pointDiameter,
                    (int) pointDiameter);
            g2D.dispose();
        }
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

            for (int i = 0; i < background.getWidth(); i++) {
                for (int j = 0; j < background.getHeight(); j++) {
                    double nx = (double) i / background.getWidth();
                    double ny = (double) j / background.getHeight();

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

    private void recreateBackground() {
        drawQuality = (int) dQualitySpinner.getModel().getValue();
        background = new BufferedImage(w / 100 * drawQuality, h / 100 * drawQuality, BufferedImage.TYPE_INT_RGB);
        log.info("Background width: {}, height: {}. Draw quality: {}", background.getWidth(), background.getHeight(), drawQuality);
    }

    private void render(Graphics2D g2D) {
        if (drawQuality <= 25) {
            if (renderOn) {
                log.info("Отключение качественного рендеринга...");
            }
            g2D.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_SPEED);
            g2D.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF);
            g2D.setRenderingHint(RenderingHints.KEY_FRACTIONALMETRICS, RenderingHints.VALUE_FRACTIONALMETRICS_OFF);
            g2D.setRenderingHint(RenderingHints.KEY_STROKE_CONTROL, RenderingHints.VALUE_STROKE_DEFAULT);
            renderOn = false;
            return;
        }

        if (!renderOn) {
            log.info("Активация качественного рендеринга...");
        }
        g2D.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2D.setRenderingHint(RenderingHints.KEY_FRACTIONALMETRICS, RenderingHints.VALUE_FRACTIONALMETRICS_ON);
        g2D.setRenderingHint(RenderingHints.KEY_STROKE_CONTROL, RenderingHints.VALUE_STROKE_NORMALIZE);
        g2D.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        renderOn = true;
    }
}
