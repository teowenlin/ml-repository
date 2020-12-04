## Reference code obtained from https://towardsdatascience.com/build-recommendation-system-with-pyspark-
# ## using-alternating-least-squares-als-matrix-factorisation-ebe1ad2e7679

from pyspark import SparkContext, SparkConf
from pyspark.sql import *
from pyspark.sql.types import StructType, IntegerType, FloatType
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.sql.functions import isnan, when, count, col

# def get_mat_sparsity(ratings):
#     # Count the total number of ratings in the dataset
#     count_nonzero = ratings.select("rating").count()
#
#     # Count the number of distinct Reviewers and distinct Titles
#     total_elements = ratings.select("src").distinct().count() * ratings.select("dst").distinct().count()
#
#     # Divide the numerator by the denominator
#     sparsity = (1.0 - (count_nonzero *1.0)/total_elements)*100
#     print("The ratings dataframe is ", "%.2f" % sparsity + "% sparse.")

def main():
    cnfg = SparkConf().setAppName("CustomerApplication").setMaster("local[*]")
    cnfg.set("spark.driver.memory", "15g")
    sc = SparkContext(conf=cnfg)
    spark = SparkSession(sc)

    ratingsPath = "C:\\Python\\BDEWAProject\\ratingsdf.csv"

    schema = (StructType().
        add("src", IntegerType()).
        add("dst", IntegerType()).
        add("rating", FloatType())
        )

    ratingsdf = (spark.read.schema(schema = schema).csv(ratingsPath, header=True))
    ratingsdf.printSchema()
    ratingsdf.show(20)
    print("shape of ratingsdf")
    print((ratingsdf.count(), len(ratingsdf.columns)))
    print("count of null and missing values in ratingsdf")
    ratingsdf.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) \
                      for c in ratingsdf.columns]).show()

    get_mat_sparsity(ratingsdf)

    # Collaborative filtering algorithm

    # Create train and test set
    (training, test) = ratingsdf.randomSplit([0.8, 0.2])
    print("training dataset")
    training.show(20)
    print("shape of training df")
    print((training.count(), len(training.columns)))

    print("test dataset")
    test.show(20)
    print("shape of test df")
    print((test.count(), len(test.columns)))

    # Build the recommendation model using Alternating Least Squares
    als = ALS(userCol="src", itemCol="dst", ratingCol="rating",
              coldStartStrategy="drop", nonnegative = True,
              implicitPrefs=False)

    # Tune model using ParamGridBuilder
    param_grid = ParamGridBuilder() \
                 .addGrid(als.rank, [22, 24, 25]) \
                 .addGrid(als.maxIter, [20, 22, 25]) \
                 .addGrid(als.regParam, [.10, .30, .35]) \
                 .build()

    # Define evaluator as RMSE
    evaluator = RegressionEvaluator(metricName = "rmse", labelCol="rating",
                                    predictionCol="prediction")

    # Build cross validation using TrainValidationSplit
    tvs = TrainValidationSplit(
        estimator=als,
        estimatorParamMaps=param_grid,
        evaluator=evaluator)

    # Fit ALS model to training data
    model = tvs.fit(training)

    # Extract best model from the tuning exercise using ParamGridBuilder
    best_model = model.bestModel

    # Generate predictions and evaluate using RMSE
    predictions = best_model.transform(test)
    rmse = evaluator.evaluate(predictions)

    # Print evaluation metrics and model parameters
    print("RMSE = " + str(rmse))
    print("** Best Model **")
    print("  Rank:", best_model.rank)
    print("  MaxIter:", best_model._java_obj.parent().getMaxIter())
    print("  RegParam:", best_model._java_obj.parent().getRegParam())

if __name__ == '__main__':
    main()