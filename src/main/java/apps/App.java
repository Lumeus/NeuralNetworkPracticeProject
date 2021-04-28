// Приложение для тренировки нейронной сети с нуля

package apps;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import nnpp.DataSetLoader;
import nnpp.Network;

public class App {

	public static void main(String[] args) throws Exception {
		// Путь к набору данных (для каждого класса отдельная папка с изображениями)
		//String path = "C:\\Users\\Lumeus\\Desktop\\учёба\\Практика\\New_data";
		String path = args[0];
		// Путь к папке, в которую сохраняется итоговая конфигурация сети
		//String folder = "C:\\Users\\Lumeus\\Desktop\\учёба\\Практика\\Networks";
		String folder = args[1];

		// Загрузка данных
		DataSetLoader data = new DataSetLoader();
		data.loadData(path);
		
		// Получение тренировочного и тестового итераторов
		DataSetIterator trainIter = data.getTrainIter();
		DataSetIterator testIter = data.getTestIter();
		
		// Тренировка сети
		Network network = new Network();
		network.train(folder, trainIter, testIter);






	}

}
