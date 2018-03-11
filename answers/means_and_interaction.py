from __future__ import print_function
import sys
if sys.version >= '3':
    long = int
from pyspark.sql import SparkSession

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf, mean, avg
from pyspark.sql.functions import lit

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
    (training, test) = ratings.randomSplit([0.8, 0.2], seed=sys.argv[1])   
    global_mean_value = (training.select(mean(training['rating']))).head()[0]
    user_mean_global = training.groupBy('userId').agg(mean("rating").alias("user_mean"))
    item_mean_global = training.groupBy('movieId').agg(mean("rating").alias("item_mean"))
    table_all_Columns = training.join(user_mean_global, "userId").join(item_mean_global, "movieId").withColumn('global_mean', lit(global_mean_value))
  
    def get_user_item_interaction(rating, user_mean, item_mean, global_mean):
        return rating - (user_mean + item_mean - global_mean) 
    
    user_item_interaction = udf(get_user_item_interaction)
    number_result = sys.argv[2]
    result = table_all_Columns.select('userId', 'movieId', 'rating', 'user_mean', 'item_mean', user_item_interaction('rating', 'user_mean', 'item_mean', 'global_mean').alias("user-item-interaction")).take(17)

    for elem in result:
        print(elem)