package network.gui;

import network.CustomPoint;
import network.net.NeuralNetwork;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.util.List;
import java.util.ArrayList;
import java.util.function.UnaryOperator;


public class DotsFrame extends JFrame implements iWorkFrame {
    private final int w = 1440, h = 900;

    private final double pointDiameter = 26D;
    private double learningRate = 0.0025D;
    private double decay = 0.0025D;
    private int drawQuality = 36;
    private int workSpeed = 1000;

    private final UnaryOperator<Double> sigmoid = x -> 1 / (1 + Math.exp(-x));
    private final UnaryOperator<Double> dsigmoid = y -> y * (1 - y);
    private final int[] sizes = new int[] {2, 8, 10, 2};

    private final NeuralNetwork nn;
    private final Thread drawThread;
    private BufferedImage pimg;
    private final List<CustomPoint> markers = new ArrayList<>();

    private static JPanel drawPane, controlPane;
    private JSpinner dQualitySpiner;

    private boolean warnMes = false, renderOn = false, useGradient = true;


    public DotsFrame() {
        if (w % drawQuality != 0 || h % drawQuality != 0) {warnMes = true;}

        nn = new NeuralNetwork(this, learningRate, decay, sigmoid, dsigmoid, sizes);

        setPreferredSize(new Dimension(w, h));
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        drawPane = new JPanel() {
            @Override
            public void paintComponent(Graphics g) {
                if (warnMes) {
                    g.setColor(Color.RED);
                    g.drawString("width and height must be multiplice by " + drawQuality, getWidth() / 2, getHeight() / 2);
                    return;
                }

                Graphics2D g2D = (Graphics2D) g;
                render(g2D);

                g2D.drawImage(pimg, 0, 0, getWidth(), getHeight(), this);

                for (CustomPoint p : markers) {
                    if (p.type() == 0) {g2D.setColor(Color.GREEN);
                    } else {g2D.setColor(Color.BLUE);}

                    g2D.fillOval((int) (p.x() - pointDiameter / 2D), (int) (p.y() - pointDiameter / 2D), (int) pointDiameter, (int) pointDiameter);
                }

                g.setColor(Color.RED);
                g.drawString("Rendering: " + (renderOn ? "ON" : "OFF"), 12, 30);

                g2D.dispose();
            }

            private void render(Graphics2D g2D) {
                if (drawQuality <= 4) {
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
                setFocusable(true);

                addMouseListener(new MouseAdapter() {
                    @Override
                    public void mousePressed(MouseEvent e) {
                        int type;
                        if(e.getButton() == 3) {type = 1;
                        } else {type = 0;}

                        CustomPoint aNewMarker = new CustomPoint(e.getPoint().getX(), e.getPoint().getY(), type);
                        for (CustomPoint existMarker : markers) {
                            if (aNewMarker.x() >= existMarker.x() - pointDiameter
                                    && aNewMarker.x() <= existMarker.x() + pointDiameter
                                    && aNewMarker.y() >= existMarker.y() - pointDiameter
                                    && aNewMarker.y() <= existMarker.y() + pointDiameter
                            ) {return;}
                        }
                        markers.add(aNewMarker);
                    }
                });

            }
        };

        controlPane = new JPanel(new GridLayout(0, 12, 3, 3)) {
            {
                SpinnerModel sm = new SpinnerNumberModel(learningRate, 0.0025D, 1.0000D, 0.0025D);
                JSpinner lRateSpiner = new JSpinner(sm) {
                    {
                        setToolTipText("Learning rate");
                        addChangeListener(e -> learningRate = (Double) getModel().getValue());
                        setEditor(new JSpinner.NumberEditor(this, "0.0000"));
                    }
                };


                SpinnerModel sm2 = new SpinnerNumberModel(workSpeed, 1, 100_000, 1000);
                JSpinner wSpeedSpiner = new JSpinner(sm2) {
                    {
                        setToolTipText("Work speed");
                        addChangeListener(e -> workSpeed = (int) getModel().getValue());
                    }
                };

                SpinnerModel sm3 = new SpinnerNumberModel(drawQuality, 2, h, 2);
                dQualitySpiner = new JSpinner(sm3) {
                    {
                        setToolTipText("Draw hardness (quality)");
                        addChangeListener(new ChangeListener() {
                            @Override
                            public void stateChanged(ChangeEvent e) {
                                drawQuality = h / 4 / (int) getModel().getValue();
                                pimg = new BufferedImage(w / drawQuality, h / drawQuality, BufferedImage.TYPE_INT_RGB);
                            }
                        });
                    }
                };

                add(lRateSpiner);
                add(wSpeedSpiner);
                add(dQualitySpiner);
            }
        };

        add(drawPane, BorderLayout.CENTER);
        add(controlPane, BorderLayout.SOUTH);

        pack();
        setLocationRelativeTo(null);
        setVisible(true);

        drawQuality = h / 4 / (int) dQualitySpiner.getModel().getValue();
        pimg = new BufferedImage(w / drawQuality, h / drawQuality, BufferedImage.TYPE_INT_RGB);

        drawThread = new Thread(() -> {
            while (true) {
                recheck();
                drawPane.repaint();

              try {Thread.sleep(13);
              } catch (InterruptedException e) {
                  Thread.currentThread().interrupt();
              }
            }
        });
        drawThread.start();
    }

    private void recheck() {
        try {
            if (!markers.isEmpty()) {
                for (int k = 0; k < workSpeed; k++) {
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
                        pimg.setRGB(i, j, getColor(outputs)); // и рисуем цвет по полученному от сети ответу!
                    }
                }

            }
        } catch (ArrayIndexOutOfBoundsException e) {
            /* IGNORE BY NOW... */
        }
    }

    private int getColor(double[] outputs) {
        double green = Math.max(0, Math.min(1, outputs[0] - outputs[1] + 0.25)); // от 0 до 1

        if (useGradient) {
            double red = 100;
            double blue = 1 - green; // либо 1, либо 0

            green = 0.3 + green * 0.5;
            blue = 0.5 + blue * 0.5;
            return (((int)red << 16) | ((int)(green * 255) << 8) | (int)(blue * 255));
        } else {
            return green >= 0.5 ? Color.green.darker().getRGB() : Color.CYAN.darker().getRGB();
        }
    }
}
