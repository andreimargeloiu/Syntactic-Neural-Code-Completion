public class Example {
    int a, b, c;
    private static final int PROTEINS = 0;

	public void for_loop() {
        for (int i; i<=100; ++i) {
            a++;
        }
	}

    public void increase_b() {
        b++;
    }

    public void method_ifelse() {
        if (a > b) {
            b++;
        } else {
            a++;
        }
    }

    public void method_throws() throws Exception {
        if (a > 5) {
        	throw new Exception();
        }
    }
}

