package main

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity.Gini
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.configuration.Strategy

object RandomForestClassifier {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[4]")
    val sc = new SparkContext(conf)

    val vectors = sc.textFile("/home/christos/BigDataProject/data/train/vectors2000.txt")


    val parsedData = vectors.map { line =>
      val parts = line.split(",")
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(" ").map(_.toDouble)))
    }

    val splits = parsedData.randomSplit(Array(0.8, 0.2), seed = 11L)
    val training = splits(0)
    val test = splits(1)

    val algorithm = Classification
    val impurity = Gini
    val maximumDepth = 3
    val treeCount = 20
    val featureSubsetStrategy = "auto"
    val seed = 5043

    val model = RandomForest.trainClassifier(training, new Strategy(algorithm, impurity, maximumDepth), treeCount, featureSubsetStrategy, seed)


    val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

    println (accuracy)


    sc.stop()

  }

}

