package nnpp;

//import org.deeplearning4j.datapipelineexamples.utils.DownloaderUtility;
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
//import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.util.Random;

public class DataSetLoader {

    private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;

    private static final long seed = 12345;

    private static final Random randNumGen = new Random(seed);

    private static final int height = 50;//290;//
    private static final int width = 50;//360;//
    private static final int channels = 3;

    private DataSetIterator trainIter;
    private DataSetIterator testIter;
    
    public String dataLocalPath;
    
    public void loadData(String path) throws Exception {
    	
    	dataLocalPath = path;
    	
    	File parentDir=new File(dataLocalPath);
        
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);

        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];
        
        ImageRecordReader trainRecordReader = new ImageRecordReader(height,width,channels,labelMaker);
        ImageRecordReader testRecordReader = new ImageRecordReader(height,width,channels,labelMaker);

        //ImageTransform transform = new MultiImageTransform(randNumGen,new ShowImageTransform("Display - before "));
        ImageTransform transform = new MultiImageTransform(randNumGen,
        		new CropImageTransform(10), new FlipImageTransform(),
        		new ScaleImageTransform(10), new WarpImageTransform(10),
        		new ShowImageTransform("Display - before "));
        
        trainRecordReader.initialize(trainData,transform);
        testRecordReader.initialize(testData,transform);
        int outputNum = trainRecordReader.numLabels();
        int batchSize = 32; // Minibatch size. Here: The number of images to fetch for each call to dataIter.next().
        int labelIndex = 1; // Index of the label Writable (usually an IntWritable), as obtained by recordReader.next()
        
        trainIter = new RecordReaderDataSetIterator(trainRecordReader, batchSize, labelIndex, outputNum);
        testIter = new RecordReaderDataSetIterator(testRecordReader, batchSize, labelIndex, outputNum);
    }

    public DataSetIterator getTrainIter() {
    	return trainIter;
    }

    public DataSetIterator getTestIter() {
    	return testIter;
    }
	
}
