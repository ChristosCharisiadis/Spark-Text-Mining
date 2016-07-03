package main


/*
* NaiveBayes Classifier
* Available models: "multinomial", "bernoulli"
* lambda > 0
 */
object Main {
  def main(args: Array[String]) {

    val inputFile = "/home/christos/BigDataProject/data/train/vectors2000.txt"
    val outFileName = "/home/christos/Dropbox/results/results.txt"
    val kValue = "5"
//    val maxDepth = "5"
//    DecisionTreeClassifier.main(Array(inputFile, outFileName, kValue, maxDepth))

//    val inputFile = "/home/christos/BigDataProject/data/train/vectors10000.txt"
//    val outFileName = "/home/christos/BigDataProject/data/train/results/LogisticRegression_Results.txt"
//    val kValue = "5"
//    val iterations1 = "3"
//    val maxDepth = "5"
//    GradientBoostedTreeClassifier.main(Array(inputFile, outFileName, kValue, iterations1, maxDepth))

//    val inputFile = "/home/christos/BigDataProject/data/train/vectors10000.txt"
//    val outFileName = "/home/christos/BigDataProject/data/train/results/LogisticRegression_Results.txt"
//    val kValue = "5"
//    LogisticRegressionClassifier.main(Array(inputFile, outFileName, kValue))

//    val inputFile = "/home/christos/BigDataProject/data/train/vectors10000.txt"
//    val outFileName = "/home/christos/BigDataProject/data/train/results/LogisticRegression_Results.txt"
//    val kValue = "5"
//    val modelValue = "multinomial"
//    val lValue = "1"
//    NaiveBayesClassifier.main(Array(inputFile, outFileName, kValue, modelValue, lValue))

//    val inputFile = "/home/christos/BigDataProject/data/train/vectors10000.txt"
//    val outFileName = "/home/christos/BigDataProject/data/train/results/LogisticRegression_Results.txt"
//    val kValue = "5"
    val maxDepth1 = "5"
    val treeCount = "5"
    RandomForestClassifier.main(Array(inputFile, outFileName, kValue, maxDepth1, treeCount))

//    val inputFile = "/home/christos/BigDataProject/data/train/vectors10000.txt"
//    val outFileName = "/home/christos/BigDataProject/data/train/results/LogisticRegression_Results.txt"
//    val kValue = "5"
//    val iterations = "10"
//    SVMClassifier.main(Array(inputFile, outFileName, kValue, iterations))
  }
}
