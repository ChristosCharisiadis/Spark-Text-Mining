package main

import java.io.PrintWriter

import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.feature.IDF
import org.apache.spark.SparkContext._
import org.apache.spark.{SparkConf, SparkContext}
import math.log10
import java.io.PrintWriter
import java.io.FileWriter
import java.io.File

object FeatureSelector {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[1]")
    val sc = new SparkContext(conf)

    val documents = sc.textFile("/home/christos/BigDataProject/data/train/headinput.txt")
    val documentsN: Int = documents.count().toInt
    val terms = documents.flatMap(x => x.split(" ")).distinct().collect().sortBy(identity)
    val termsN: Int = terms.length

    // calculate idf
    var idfMap: Map[String, Double] = Map()
    for (term <- terms) {
      val idf = log10(documentsN / documents.filter(x => x.contains(term)).count().toDouble)
      idfMap += (term -> idf)
    }

    val outFileName = "/home/christos/BigDataProject/data/train/vectors.txt"
    new File(outFileName).delete()

    val vectorMap: Array[Array[Double]] = new Array(documentsN)
    var i = 0
    for (document <- documents) {
      var outputVector = ""
      val tfArray: Array[Double] = new Array(termsN)
      var j = 0
      for (term <- terms) {
        val tf = document.mkString.split(" ").count(x=>x.equals(term))
        tfArray(j) = tf
        j += 1
      }
      vectorMap(i) = tfArray
      i += 1
      if (i <= 10) {
        outputVector = "1 " + tfArray.mkString(" ")
      }
      else {
        outputVector = "0 " + tfArray.mkString(" ")
      }

      val outFile = new FileWriter(outFileName, true)
      outFile.write(outputVector + "\n")
      outFile.close()
    }

    sc.stop()

  }

}

