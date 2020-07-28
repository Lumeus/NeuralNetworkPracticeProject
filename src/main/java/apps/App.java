package apps;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import nnpp.DataSetLoader;
import nnpp.Network;

public class App {

	public static void main(String[] args) throws Exception {

		String path = "C:\\Users\\Lumeus\\Desktop\\учёба\\Практика\\New_data";
		String folder = "C:\\Users\\Lumeus\\Desktop\\учёба\\Практика\\Networks";


		DataSetLoader data = new DataSetLoader();
		data.loadData(path); // path - папка с данными

		DataSetIterator trainIter = data.getTrainIter();
		DataSetIterator testIter = data.getTestIter();
		
		Network network = new Network();
		network.train(folder, trainIter, testIter); // folder - папка для сохранения

	}

}
