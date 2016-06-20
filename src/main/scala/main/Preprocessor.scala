package main

import java.io.PrintWriter

import collection.mutable.HashMap
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

import scala.collection.JavaConverters._

object Preprocessor {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[1]")
    val sc = new SparkContext(conf)

    processInputFiles("/home/christos/BigDataProject/data/train/pos", "/home/christos/BigDataProject/data/train/posFile.txt")
    processInputFiles("/home/christos/BigDataProject/data/train/neg", "/home/christos/BigDataProject/data/train/negFile.txt")
  }

  def processInputFiles (dirName: String, outFileName: String) {
    val inFiles = new java.io.File(dirName).listFiles()
    val outFile = new PrintWriter(outFileName)

    for (file <- inFiles) {
      val inFile = scala.io.Source.fromFile(file)
      val text = try inFile.mkString finally inFile.close()
      val tokenizer = new Tokenizer(text, "english")
      var words = ""

      tokenizer.eachWord().foreach(word => words += " " + word)
      outFile.write(words + "\n")
    }
    outFile.close()
  }
}

