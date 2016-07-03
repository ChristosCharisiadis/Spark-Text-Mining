package main

import java.io.FileWriter
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils.kFold

/*
* NaiveBayes Classifier
* Available models: "multinomial", "bernoulli"
* lambda > 0
 */
object NaiveBayesClassifier {
  def main(args: Array[String]) {
    // get Input values
    val inputFile = args(0)
    val outFileName = args(1)
    val kValue = args(2).toInt
    val modelValue = args(3)
    val lValue = args(4).toDouble

    // Spark Context
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[4]")
    val sc = new SparkContext(conf)

    // load input File
    val vectors = sc.textFile(inputFile)

    // create LabeledPoint
    val parsedData = vectors.map { line =>
    val parts = line.split(",")
    LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(" ").map(_.toDouble)))
    }

    // start k-Fold cross validation
    val cvData = kFold(parsedData, kValue, 42)
    val accuracies = cvData.map { case (train, test) => {
      val model = NaiveBayes.train(train, lambda = lValue, modelType = modelValue)
      val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
      val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
      accuracy
    }}

    // find average accuracy
    val avgAccuracy = accuracies.sum / accuracies.length

    // write result to file
    val outFile = new FileWriter(outFileName, true)
    outFile.write("NaiveBayes - Accuracy = " + avgAccuracy + ", model = " + modelValue + ", λ = " + lValue + "\n")
    outFile.close()

    sc.stop()
  }
}
