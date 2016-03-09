package de.pmenke.neurophtest;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.LockSupport;

import org.neuroph.core.data.DataSet;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.ImmutableList;

/**
 * <br/><br/>
 * Created by pmenke on 09.03.16.
 */
public class Main {
    private static final Logger LOG = LoggerFactory.getLogger(Main.class);
    public static void main(String[] args) {
        final MultiLayerPerceptron perceptron = new MultiLayerPerceptron(ImmutableList.of(2, 3, 1),
                TransferFunctionType.SIGMOID);
        final DataSet dataSet = new DataSet(2, 1);
        dataSet.addRow(a(0,0), a(0));
        dataSet.addRow(a(1,0), a(1));
        dataSet.addRow(a(0,1), a(1));
        dataSet.addRow(a(1,1), a(0));

        final BackPropagation learningRule = new BackPropagation();
        perceptron.setLearningRule(learningRule);
        learningRule.setMaxError(0.001);
        learningRule.setMaxIterations(100000);
        perceptron.learnInNewThread(dataSet);
        final Thread learningThread = perceptron.getLearningThread();
        while (learningThread.isAlive()) {
            LOG.info("Iteration {}/{}: MSE {}", learningRule.getCurrentIteration(), learningRule.getMaxIterations(),
                    learningRule.getTotalNetworkError());
            LockSupport.parkNanos(TimeUnit.MILLISECONDS.toNanos(500));
        }
        LOG.info("Iteration {}/{}: MSE {}", learningRule.getCurrentIteration(), learningRule.getMaxIterations(),
                learningRule.getTotalNetworkError());

        perceptron.setInput(0, 0);
        perceptron.calculate();
        LOG.info("0,0 => {}", perceptron.getOutput());
        perceptron.setInput(1, 0);
        perceptron.calculate();
        LOG.info("1,0 => {}", perceptron.getOutput());
        perceptron.setInput(0, 1);
        perceptron.calculate();
        LOG.info("0,1 => {}", perceptron.getOutput());
        perceptron.setInput(1, 1);
        perceptron.calculate();
        LOG.info("1,1 => {}", perceptron.getOutput());
    }

    private static double[] a(double... data){
        return data;
    }
}
