package main

import org.apache.spark.{SparkConf, SparkContext}
import math.log10
import java.io.FileWriter
import java.io.File

/*
* FeatureSelector will calculate the tfidf values for all the terms of the corpus for each document, and create a
* text file with the label and the tfidf vectors.
 */
object FeatureSelector {
  def main(args: Array[String]) {
    // only use one core
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

    val outFileName = "/home/christos/BigDataProject/data/train/vectors2.txt"
    new File(outFileName).delete()

    // calculate tf and tfidf and write them to the output file
    var i = 0
    for (document <- documents) {
      var outputVector = ""
      val tfidfArray: Array[Double] = new Array(termsN)
      var j = 0
      for (term <- terms) {
        val tf = document.mkString.split(" ").count(x=>x.equals(term))
        tfidfArray(j) = tf * idfMap(term)
        j += 1
      }
      i += 1
      if (i <= 100) {
        outputVector = "1," + tfidfArray.mkString(" ")
      }
      else {
        outputVector = "0," + tfidfArray.mkString(" ")
      }
      val outFile = new FileWriter(outFileName, true)
      outFile.write(outputVector + "\n")
      outFile.close()
      println(i)
    }

    sc.stop()
  }
}
