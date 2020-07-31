// Класс, загружающий данные и возвращающий итераторы

package nnpp;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.CropImageTransform;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.datavec.image.transform.ScaleImageTransform;
import org.datavec.image.transform.ShowImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.util.Random;

public class DataSetLoader {

    private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;

    private static final long seed = 12345; // Начальное значение генератора псевдослучайных чисел

    private static final Random randNumGen = new Random(seed);

    private static final int height = 50; // Высота и ширна, к которым
    private static final int width = 50; // приводятся изображения
    private static final int channels = 3; // Цветность изображений
    
    // Итераторы
    private DataSetIterator trainIter;
    private DataSetIterator testIter;
    
    // Расположение данных
    public String dataLocalPath;
    
    public void loadData(String path) throws Exception {
    	
    	dataLocalPath = path;
    	
    	// Открытие директории
    	File parentDir=new File(dataLocalPath);
        
    	// Чтение файлов
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);
        
        // Объект, распознающий классы
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        
        // Создание сбалансированной выборки
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);
        
        // Разделение датасета на тренировочную и тестовую подвыборки
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];
        
        // Объекты, сопоставляющие изображения и классы
        ImageRecordReader trainRecordReader = new ImageRecordReader(height,width,channels,labelMaker);
        ImageRecordReader testRecordReader = new ImageRecordReader(height,width,channels,labelMaker);

        // Случайные трансформации изображений
        ImageTransform transform = new MultiImageTransform(randNumGen,
        		new CropImageTransform(10), new FlipImageTransform(),
        		new ScaleImageTransform(10), new WarpImageTransform(10),
        		new ShowImageTransform("Display - before "));
        
        // Подготовка данных
        trainRecordReader.initialize(trainData,transform);
        testRecordReader.initialize(testData,transform);
        
        int outputNum = trainRecordReader.numLabels(); // Число классов
        int batchSize = 32; // Размер минибатча
        int labelIndex = 1; // Уникальный идентификатор набора данных
        
        // Создание итераторов
        trainIter = new RecordReaderDataSetIterator(trainRecordReader, batchSize, labelIndex, outputNum);
        testIter = new RecordReaderDataSetIterator(testRecordReader, batchSize, labelIndex, outputNum);
    }

    // Получение тренировочного итератора
    public DataSetIterator getTrainIter() {
    	return trainIter;
    }

    // Получение тестового итератора
    public DataSetIterator getTestIter() {
    	return testIter;
    }
	
}
