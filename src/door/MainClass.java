package door;

import com.FormDigits;
import com.Frame;
import java.io.IOException;


public class MainClass {

    public static void main(String[] args) throws IOException {
//        dots();
        digits();
    }

    private static void dots() {new Frame();}

    private static void digits() throws IOException {new FormDigits();}
}
