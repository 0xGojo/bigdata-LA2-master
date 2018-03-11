from __future__ import print_function
import sys
if sys.version >= '3':
    long = int
from pyspark.sql import SparkSession

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import udf, mean, avg, lit


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("thach")\
        .getOrCreate()
    lines = spark.read.text("./sample_movielens_ratings.txt").rdd
    parts = lines.map(lambda row: row.value.split("::"))
    ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                         rating=float(p[2]), timestamp=long(p[3])))
    ratings = spark.createDataFrame(ratingsRDD)
    
    global_mean_value = (ratings.select(mean(ratings['rating']))).head()[0]
    user_mean_global = ratings.groupBy('userId').agg(mean("rating").alias("user_mean"))
    item_mean_global = ratings.groupBy('movieId').agg(mean("rating").alias("item_mean"))
    table_all_Columns = ratings.join(user_mean_global, "userId").join(item_mean_global, "movieId").withColumn('global_mean', lit(global_mean_value))
  
    def get_user_item_interaction(rating, user_mean, item_mean, global_mean):
        return rating - (user_mean + item_mean - global_mean) 
    
    user_item_interaction = udf(get_user_item_interaction, DoubleType())

    user_item_interaction_data = table_all_Columns.select('userId', 'movieId', 'rating', 'user_mean', 'item_mean', user_item_interaction('rating', 'user_mean', 'item_mean', 'global_mean').alias("user-item-interaction"))

    (training, test) = user_item_interaction_data.randomSplit([0.8, 0.2], seed=sys.argv[1])   
    als = ALS(maxIter=5, rank=70, coldStartStrategy="drop", regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="user-item-interaction")
    model = als.fit(training)
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="user-item-interaction",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print(str(rmse))
    spark.stop()