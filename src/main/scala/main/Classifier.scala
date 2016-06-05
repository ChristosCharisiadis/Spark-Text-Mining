package main

import collection.mutable.HashMap
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object Classifier {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Simple Application")
    val sc = new SparkContext(conf)

    val tokenizer = new Tokenizer("i am reading this liking reading likes reads read. i have many books(). I like book and booking", "english")
    val wordCount = HashMap[String, Int]()

    tokenizer.eachWord().foreach(word => wordCount.put(word, 1 + wordCount.get(word).getOrElse(0)))
    println(wordCount)
  }
}

