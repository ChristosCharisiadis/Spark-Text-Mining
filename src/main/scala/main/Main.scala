package main

object Main {
  def main(args: Array[String]) {
    // modify the parameters and select the classifiers you want to train
    val kValue = "5"

    val inputFile1 = "vectors200.txt"
    val outFileName1 = "results_200.txt"
    val maxDepths1 = Array("3","5","10")
    for (maxDepth <- maxDepths1) {
      DecisionTreeClassifier.main(Array(inputFile1, outFileName1, kValue, maxDepth))
    }

    val inputFile2 = "vectors200.txt"
    val outFileName2 = "GradBoostTree_results_200.txt"
    val iterations1 = Array("5","10","20")
    val maxDepths2 = Array("3","5","10")
    for (iterations1 <- iterations1) {
      for (maxDepth <- maxDepths2) {
        GradientBoostedTreeClassifier.main(Array(inputFile2, outFileName2, kValue, iterations1, maxDepth))
      }
    }

    val inputFile3 = "vectors200.txt"
    val outFileName3 = "LogReg_results_200.txt"
    LogisticRegressionClassifier.main(Array(inputFile3, outFileName3, kValue))

    val inputFile4 = "vectors200.txt"
    val outFileName4 = "NaiveBayes_results_200.txt"
    val modelValue = "multinomial"
    val lValue = Array("0.1","1","10","20")
    for (lval <- lValue) {
      NaiveBayesClassifier.main(Array(inputFile4, outFileName4, kValue, modelValue, lval))
    }

    val inputFile5 = "/vectors200.txt"
    val outFileName5 = "RandForest_results_200.txt"
    val maxDepths3 = Array("3","5","10")
    val treeCounts = Array("5","10","20","50")
    for (treeCount <- treeCounts) {
      for (maxDepth1 <- maxDepths3) {
        RandomForestClassifier.main(Array(inputFile5, outFileName5, kValue, maxDepth1, treeCount))
      }
    }

    val inputFile6 = "vectors200.txt"
    val outFileName6 = "SVM_results_200.txt"
    val iterations2 = Array ("10","20","50")
    for (iters <- iterations2) {
      SVMClassifier.main(Array(inputFile6, outFileName6, kValue, iters))
    }

  }
}
