package main

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy

object GradientBoostedTreeClassifier {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[4]")
    val sc = new SparkContext(conf)

    val vectors = sc.textFile("/home/christos/BigDataProject/data/train/vectors2000.txt")

    val parsedData = vectors.map { line =>
      val parts = line.split(",")
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(" ").map(_.toDouble)))
    }

    val splits = parsedData.randomSplit(Array(0.7, 0.3), seed = 11L)
    val training = splits(0)
    val test = splits(1)

    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.numIterations = 3
    boostingStrategy.treeStrategy.numClasses = 2
    boostingStrategy.treeStrategy.maxDepth = 5
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

    val model = GradientBoostedTrees.train(training, boostingStrategy)


    val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
    val a = test.map {point =>
    val prediction = model.predict(point.features)
      (point.label, prediction)}
    val err = a.filter(r => r._1 != r._2).count.toDouble / test.count()
    println("error" + err)
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

    println (accuracy)


    sc.stop()

  }

}

