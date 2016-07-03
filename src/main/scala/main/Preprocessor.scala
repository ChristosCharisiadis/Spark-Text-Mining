package main

import java.io.PrintWriter
import org.apache.spark.{SparkConf, SparkContext}

/*
* Preprocessor will tokenize, remove stopwords and perform stemming, to create a file that can then be used to find the
* tfidf values.
 */
object Preprocessor {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[1]")
    val sc = new SparkContext(conf)

    //process the directory with the positive reviews
    processInputFiles("/home/christos/BigDataProject/data/train/pos", "/home/christos/BigDataProject/data/train/posFile.txt")
    //process the directory with the negative reviews
    processInputFiles("/home/christos/BigDataProject/data/train/neg", "/home/christos/BigDataProject/data/train/negFile.txt")
  }

  /*
  * Input: Directory with text files
  * Output: Output file with the results
   */
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
