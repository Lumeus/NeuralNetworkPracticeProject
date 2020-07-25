package apps;

import java.io.File;
import java.io.IOException;

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
		String path = "C:\\Users\\Lumeus\\Desktop\\учёба\\Практика\\Networks\\nnpp_22-07_19-51.zip";
		String filePath = "C:\\Users\\Lumeus\\Desktop\\учёба\\Практика\\Data\\гречка\\DSC_2977.JPG";
		File net = new File(path);
		//File file = new File(filePath);
		MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(net);
		
		NativeImageLoader loader = new NativeImageLoader(50, 50, 3);
        INDArray image = loader.asMatrix(filePath);
        INDArray output = model.output(image, false);
        System.out.println(output);
	}

}
