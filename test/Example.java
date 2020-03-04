import java.util.Collections;
import java.util.ArrayList;
import java.util.List;

public class Example {
	private ArrayList<Integer> validation_error;

    private void check_convergence(ArrayList<Integer> list) {
        for (int i=0; i<10; ++i) {
            if (i > 0) {
                break;
            }
        }
	}

	public void add_error(int error){
	    validation_error.add(error);
		check_convergence(validation_error);
	}
}