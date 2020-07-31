// Класс, тренирующий нейронную сеть

package nnpp;

import java.io.File;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Network {
	
	private static final Logger log = LoggerFactory.getLogger(Network.class);

	public void train(String folder, DataSetIterator train, DataSetIterator test) throws Exception {
		
        int nChannels = 3; // Число входных каналов (зависит от цветности изображения)
        int outputNum = 4; // Число выходов, т.е. классов
        int nEpochs = 10; // Число эпох тренировки
        int seed = 123; // Начальное значение генератора псевдослучайных чисел
        
        log.info("Build model....");
        // Конфигурация модели
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
                        //Note that nIn need not be specified in later layers
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
                        //Note that nIn need not be specified in later layers
                        .kernelSize(2,2)
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
        
        // Создание модели
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        
        log.info("Train model...");
        // Тренировка модели
        model.setListeners(new ScoreIterationListener(10), new EvaluativeListener(test, 1, InvocationType.EPOCH_END)); //Print score every 10 iterations and evaluate on test set every epoch
        model.fit(train, nEpochs);
        
        // Полное имя архива для сохранения модели
        String path = FilenameUtils.concat(folder, "nnpp.zip");
        
        log.info("Saving model to "+path);
        // Сохранение модели
        model.save(new File(path), true);
        
        log.info("****************Training finished********************");
	}
	
}
