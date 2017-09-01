/**
  * Created by itim on 22.04.2017.
  */

import org.apache.spark.SparkConf

import org.apache.spark.SparkContext

import org.apache.spark.rdd._

import org.apache.spark.mllib.recommendation.{ALS, Rating, MatrixFactorizationModel}

object MovieLensALS {

  def main(args: Array[String]) {
    // set up environment

    val conf = new SparkConf()

      .setAppName("MovieLensALS")

      .set("spark.executor.memory", "2g")

      .setMaster("local[*]")

    val sc = new SparkContext(conf)

    // load ratings and movie titles

    val movieLensHomeDir = "hdfs://10.8.41.146:9000/dataset/movielens/";

    //RATINGS

    val ratings = sc.textFile(movieLensHomeDir + "ratings.dat").map { line =>

      val fields = line.split("::")

      // format: (timestamp % 10, Rating(userId, movieId, rating))

      (fields(3).toLong % 10, Rating(fields(0).toInt, fields(1).toInt,

        fields(2).toDouble))

    }

    //MOVIES

    val movies = sc.textFile(movieLensHomeDir + "movies.dat").map { line =>

      val fields = line.split("::")

      // format: (movieId, movieName)

      (fields(0).toInt, fields(1))

    }.collect.toMap

    //MY RATING

    val myRating = sc.textFile("personalRatings.txt").map {

      line =>

        val fields = line.split("::")

        Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)

    }.filter(t => t.rating > 0.0)

    // your code here

    val numRatings = ratings.count()

    val numUsers = ratings.map(t => t._2.user).distinct().count()

    val numMovies = ratings.map(t => t._2.product).distinct().count()

    println("Got " + numRatings + " ratings from "

      + numUsers + " users on " + numMovies + " movies.")


    val numPartitions = 4

    val training = ratings.filter(x => x._1 < 6)

      .values

      .union(myRating)

      .repartition(numPartitions)

      .cache()

    val validation = ratings.filter(x => x._1 >= 6 && x._1 < 8)

      .values

      .repartition(numPartitions)

      .cache()

    val test = ratings.filter(x => x._1 >= 8).values.cache()

    val numTraining = training.count()

    val numValidation = validation.count()

    val numTest = test.count()

    println("Training: " + numTraining + ", validation: " + numValidation + ",test: " + numTest)

    val ranks = List(8, 12)

    val lambdas = List(1.0, 10.0)

    val numIters = List(10, 20)

    var bestModel: Option[MatrixFactorizationModel] = None

    var bestValidationRmse = Double.MaxValue

    var bestRank = 0

    var bestLambda = -1.0

    var bestNumIter = -1

    for (rank <- ranks; lambda <- lambdas; numIter <- numIters) {

      val model = ALS.train(training, rank, numIter, lambda)

      val validationRmse = computeRmse(model, validation, numValidation)

      println("RMSE (validation) = " + validationRmse + " for the model trained with rank = "

        + rank + ", lambda = " + lambda + ", and numIter = " + numIter + ".")

      if (validationRmse < bestValidationRmse) {

        bestModel = Some(model)

        bestValidationRmse = validationRmse

        bestRank = rank

        bestLambda = lambda

        bestNumIter = numIter

      }

    }

    val testRmse = computeRmse(bestModel.get, test, numTest)

    println("The best model was trained with rank = " + bestRank + " and lambda = " + bestLambda +", and numIter = " + bestNumIter + ", and its RMSE on the test set is " +  testRmse + "."
    )

    val myRatedMovieIds = myRating.map(t => t.product)

    val candidates = sc.parallelize(movies.filterKeys(key => myRatedMovieIds.filter(t => t == key).count() == 0).map(m => m._1).toSeq)

    val recommendations = bestModel.get.predict(candidates.map((0, _)))

      .collect()

      .sortBy(t => t.rating)

      .take(100)

    var i = 1

    println("Movies recommended for you:")

    recommendations.foreach { r =>

      println("%2d".format(i) + ": " + movies(r.product))

      i += 1

    }
    // clean up

    sc.stop()

  }

  def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating], n: Long):

  Double = {

    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user,

      x.product)))

    val predictionsAndRatings = predictions.map(x => ((x.user, x.product), x.rating))

      .join(data.map(x => ((x.user, x.product), x.rating)))

      .values

    math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 -

      x._2)).reduce(_ + _) / n)

  }

}