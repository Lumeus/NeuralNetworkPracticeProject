package apps;

import java.io.File;
import java.io.IOException;

import gui.GUI;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.NumberedFileInputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

public class UI {
	public static void main(String[] args) throws Exception {
		GUI gui = new GUI();
		gui.setVisible(true);
	}
}
