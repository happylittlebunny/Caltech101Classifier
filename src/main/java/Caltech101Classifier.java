
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;

import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.Random;

public class Caltech101Classifier {
    private static Logger log = LoggerFactory.getLogger(Caltech101Classifier.class);
    private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    protected static int height = 224;
    protected static int width = 224;
    protected static int channels = 3;
    protected static int seed = 123;
    private static final Random randNumGen = new Random(seed);
    protected static int batchSize = 10;
    protected static int numLabels = 101;
    protected static int numEpochs = 10;

    public static void main(String[] args) throws Exception {

        File parentDir= new File(args[0]); // "C:\\Users\\Yuyi\\Desktop\\bigdata2017Fall\\skymind\\data\\train"
        FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
        InputSplit train = filesInDirSplit[0];
        InputSplit test = filesInDirSplit[1];

        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);

        recordReader.initialize(train);

        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,numLabels);
        dataIter.setPreProcessor( new VGG16ImagePreProcessor());
        MultipleEpochsIterator trainIter;
        trainIter = new MultipleEpochsIterator(numEpochs, dataIter);

        log.info("**** Build Model ****");

        ZooModel zooModel = new VGG16(numLabels, seed, 1);
        ComputationGraph vgg16 = (ComputationGraph)zooModel.initPretrained(PretrainedType.IMAGENET);
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .learningRate(5e-5)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .seed(seed)
                .build();
        ComputationGraph model = new TransferLearning.GraphBuilder(vgg16)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("fc2")
                .removeVertexKeepConnections("predictions")
                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(4096).nOut(numLabels)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.SOFTMAX).build(), "fc2")
                .build();

        log.info(model.summary());

        model.init();

        log.info("Model build complete");

        log.info("*****TRAIN MODEL********");
        model.fit(trainIter);

        log.info("******EVALUATE MODEL******");

        recordReader.reset();

        recordReader.initialize(test);
        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,numLabels);
        testIter.setPreProcessor( new VGG16ImagePreProcessor());

        log.info(recordReader.getLabels().toString());

        Evaluation eval = model.evaluate(testIter);

        log.info(eval.stats());

        log.info("******SAVE TRAINED MODEL******");

        File locationToSave = new File("trained_caltech101_model.zip");

        boolean saveUpdater = false;

        ModelSerializer.writeModel(model,locationToSave,saveUpdater);
    }

}
