package apps;

import nnpp.DataSetLoader;
import org.apache.commons.io.FilenameUtils;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class SparkMain {
    public static void main(String[] args) throws Exception {
        // Путь к набору данных (для каждого класса отдельная папка с изображениями)
        //String path = "C:\\Users\\Lumeus\\Desktop\\учёба\\Практика\\New_data";
        String path = "D:\\учёба\\Практика\\New_data";
        // Путь к папке, в которую сохраняется итоговая конфигурация сети
        //String folder = "C:\\Users\\Lumeus\\Desktop\\учёба\\Практика\\Networks";
        String folder = "D:\\учёба\\Практика\\Networks";

        // Загрузка данных
        DataSetLoader data = new DataSetLoader();
        data.loadData(path);

        // Получение тренировочного и тестового итераторов
        DataSetIterator trainIter = data.getTrainIter();
        // DataSetIterator testIter = data.getTestIter();


        JavaSparkContext sc = new JavaSparkContext("local[2]", "NNPP");


//        SparkSession spark = SparkSession.builder().sparkContext(sc.sc()).getOrCreate();
//        Dataset<Row> imagesDF = spark.read().format("image").option("dropInvalid", true).load("data/mllib/images/origin/kittens");
//        imagesDF.select("image.origin", "image.width", "image.height").show(false);
//        JavaRDD<DataSet> trainingData = imagesDF;

        List<DataSet> list = new ArrayList<DataSet>();
        while (trainIter.hasNext()) {
            list.add(trainIter.next());
        }

        JavaRDD<DataSet> trainingData = sc.parallelize(list);

        int nChannels = 3; // Число входных каналов (зависит от цветности изображения)
        int outputNum = 4; // Число выходов, т.е. классов
        int nEpochs = 1; // Число эпох тренировки
        int seed = 123; // Начальное значение генератора псевдослучайных чисел

//Model setup as on a single node. Either a MultiLayerConfiguration or a ComputationGraphConfiguration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.0005) // Регуляризация
                .weightInit(WeightInit.XAVIER) // Начальное распределение весов
                .updater(new Adam(1e-3)) // Метод обновления весов
                .list() // Слои
                // Свёрточный с ядром 5х5, шагом 1 и 20 выходными каналами
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nIn(nChannels)
                        .stride(1,1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                // Пулинговый с ядром 2х2, шагом 2
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                // Свёрточный с ядром 3х3, шагом 1 и 50 выходными каналами
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .stride(1,1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                // Пулинговый с ядром 2х2, шагом 1
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(1,1)
                        .build())
                // Свёрточный с ядром 5х5, шагом 2 и 100 выходными каналами
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .stride(2,2)
                        .nOut(100)
                        .activation(Activation.IDENTITY)
                        .build())
                // Пулинговый с ядром 2х2, шагом 1
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(1,1)
                        .build())
                // Полносвязный с 500 выходными каналами
                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                // Выходной
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                // Установка параметров входных данных
                .setInputType(InputType.convolutionalFlat(50,50,3))
                .build();

//Create the TrainingMaster instance
        int examplesPerDataSetObject = 1;
        TrainingMaster trainingMaster = new ParameterAveragingTrainingMaster.Builder(examplesPerDataSetObject)
                .build();

//Create the SparkDl4jMultiLayer instance and fit the network using the training data:
        SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sc, conf, trainingMaster);

//Execute training:
        for (int i = 0; i < nEpochs; i++) {
            sparkNetwork.fit(trainingData);
        }

        MultiLayerNetwork model = sparkNetwork.getNetwork();

        String savePath = FilenameUtils.concat(folder, "nnpp.zip");
        model.save(new File(savePath), true);
    }
}
