package main

import java.io.FileWriter
/*
* NaiveBayes Classifier
* Available models: "multinomial", "bernoulli"
* lambda > 0
 */
object Main {
  def main(args: Array[String]) {
    val kValue = "5"

//    val inputFile = "/home/christos/BigDataProject/data/train/vectors2000.txt"
//    val outFileName = "/home/christos/Dropbox/results/DecTree_results_2000.txt"
//    val maxDepth = "5"
//    DecisionTreeClassifier.main(Array(inputFile, outFileName, kValue, maxDepth))

//    val inputFile = "/home/christos/BigDataProject/data/train/vectors2000.txt"
//    val outFileName = "/home/christos/Dropbox/results/GradBoostTree_results_2000.txt"
//    val iterations1 = "3"
//    val maxDepth = "5"
//    GradientBoostedTreeClassifier.main(Array(inputFile, outFileName, kValue, iterations1, maxDepth))

    val inputFile = "/home/christos/BigDataProject/data/train/vectors200.txt"
    val outFileName = "/home/christos/Dropbox/results/LogReg_results_200.txt"
    LogisticRegressionClassifier.main(Array(inputFile, outFileName, kValue))

//    val inputFile = "/home/christos/BigDataProject/data/train/vectors2000.txt"
//    val outFileName = "/home/christos/Dropbox/results/NaiveBayes_results_2000.txt"
//    val modelValue = "multinomial"
//    val lValue = "1"
//    NaiveBayesClassifier.main(Array(inputFile, outFileName, kValue, modelValue, lValue))

//    val inputFile = "/home/christos/BigDataProject/data/train/vectors2000.txt"
//    val outFileName = "/home/christos/Dropbox/results/RandForest_results_2000.txt"
//    val maxDepth1 = "5"
//    val treeCount = "5"
//    RandomForestClassifier.main(Array(inputFile, outFileName, kValue, maxDepth1, treeCount))

//    val inputFile = "/home/christos/BigDataProject/data/train/vectors5000.txt"
//    val outFileName = "/home/christos/Dropbox/results/SVM_results_5000.txt"
//    val iterations = Array ("5","10","20","50","100")
//    for (iters <- iterations) {
//      SVMClassifier.main(Array(inputFile, outFileName, kValue, iters))
//    }

  }
}
