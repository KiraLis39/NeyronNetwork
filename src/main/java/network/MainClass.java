package network;

import network.gui.DigitsFrame;
import network.gui.DotsFrame;

public class MainClass {

    public static void main(String[] args) {
        dots();
//        digits();
    }

    private static void dots() {
        new DotsFrame();
    }

    private static void digits() {
        new DigitsFrame();
    }
}
