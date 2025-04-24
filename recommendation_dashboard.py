
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, avg, row_number
from pyspark.sql.window import Window

# Start Spark
spark = SparkSession.builder.getOrCreate()

# Load and prepare data (based on provided code)
sample_df = spark.read.csv("gs://dataproc-staging-europe-west2-348416659726-ml0mzsp9/2019-10-01.csv", header=True, inferSchema=True)

from pyspark.sql.functions import when, sum as spark_sum
weighted_df = sample_df.withColumn(
    "event_strength",
    when(col("event_type") == "view", 1)
    .when(col("event_type") == "cart", 5)
    .when(col("event_type") == "purchase", 10)
    .otherwise(0)
)
ratings_df = weighted_df.groupBy("user_id", "product_id").agg(spark_sum("event_strength").alias("rating"))

from pyspark.ml.feature import StringIndexer
user_indexer = StringIndexer(inputCol="user_id", outputCol="user_index")
item_indexer = StringIndexer(inputCol="product_id", outputCol="product_index")
user_indexed = user_indexer.fit(ratings_df).transform(ratings_df)
final_df = item_indexer.fit(user_indexed).transform(user_indexed).select("user_index", "product_index", "rating")
training, test = final_df.randomSplit([0.8, 0.2], seed=42)

from pyspark.ml.recommendation import ALS
als = ALS(
    userCol="user_index",
    itemCol="product_index",
    ratingCol="rating",
    rank=30,
    regParam=0.05,
    alpha=20,
    maxIter=15,
    implicitPrefs=True,
    coldStartStrategy="drop"
)
als_model = als.fit(training)
als_top_k = als_model.recommendForAllUsers(10)

# Popularity-based model
pop_scores = final_df.groupBy("product_index").agg(spark_sum("rating").alias("total_rating"))
max_pop = pop_scores.agg({"total_rating": "max"}).first()[0]
pop_scores = pop_scores.withColumn("pop_score", col("total_rating") / max_pop)

# Ensemble recommendations
from pyspark.sql.functions import explode
als_expanded = als_top_k.selectExpr("user_index", "explode(recommendations) as rec")     .select("user_index", col("rec.product_index"), col("rec.rating").alias("als_score"))
ensemble_df = als_expanded.join(pop_scores, "product_index", "left").fillna(0.0)
ensemble_df = ensemble_df.withColumn("hybrid_score", 0.9 * col("als_score") + 0.1 * col("pop_score"))
window_spec = Window.partitionBy("user_index").orderBy(col("hybrid_score").desc())
ranked = ensemble_df.withColumn("rank", row_number().over(window_spec))
hybrid_top_k = ranked.filter(col("rank") <= 10).groupBy("user_index").agg(collect_list("product_index").alias("predicted"))

# Evaluation
actual_items = test.groupBy("user_index").agg(collect_list("product_index").alias("actual"))
from pyspark.sql.types import FloatType
from pyspark.sql.functions import udf

def precision_at_k(predicted, actual, k=10):
    if not predicted or not actual:
        return 0.0
    return len(set(predicted[:k]) & set(actual)) / float(k)

def recall_at_k(predicted, actual, k=10):
    if not predicted or not actual:
        return 0.0
    return len(set(predicted[:k]) & set(actual)) / float(len(actual))

precision_udf = udf(precision_at_k, FloatType())
recall_udf = udf(recall_at_k, FloatType())

eval_df = hybrid_top_k.join(actual_items, on="user_index")
eval_df = eval_df.withColumn("precision_at_10", precision_udf(col("predicted"), col("actual")))                  .withColumn("recall_at_10", recall_udf(col("predicted"), col("actual")))
eval_df = eval_df.dropna(subset=["precision_at_10", "recall_at_10"])

# Convert popularity into top-N per user
pop_top_k = final_df.groupBy("user_index", "product_index")     .agg({"rating": "sum"}).withColumnRenamed("sum(rating)", "pop_score")

window_spec = Window.partitionBy("user_index").orderBy(col("pop_score").desc())
pop_ranked = pop_top_k.withColumn("rank", row_number().over(window_spec))
pop_top_k = pop_ranked.filter(col("rank") <= 10)     .groupBy("user_index").agg(collect_list("product_index").alias("pop_predicted"))

# Streamlit UI
st.title("Recommendation Dashboard")
user_input = st.text_input("Enter User ID")

if user_input:
    user_id = user_input
    user_index_row = final_df.filter(col("user_id") == user_id).select("user_index").distinct().collect()

    if user_index_row:
        user_index = user_index_row[0]["user_index"]

        als_preds = als_top_k.filter(col("user_index") == user_index).select("recommendations.product_index").collect()
        als_list = als_preds[0][0] if als_preds else []

        pop_preds = pop_top_k.filter(col("user_index") == user_index).select("pop_predicted").collect()
        pop_list = pop_preds[0]["pop_predicted"] if pop_preds else []

        hybrid_preds = hybrid_top_k.filter(col("user_index") == user_index).select("predicted").collect()
        hybrid_list = hybrid_preds[0]["predicted"] if hybrid_preds else []

        st.subheader("Top-N Recommendations")
        st.write("**ALS:**", als_list)
        st.write("**Popularity:**", pop_list)
        st.write("**Hybrid (Ensemble):**", hybrid_list)

        user_eval = eval_df.filter(col("user_index") == user_index).select("precision_at_10", "recall_at_10").collect()
        if user_eval:
            st.metric("Precision@10", f"{user_eval[0]['precision_at_10']:.2f}")
            st.metric("Recall@10", f"{user_eval[0]['recall_at_10']:.2f}")
        else:
            st.info("No evaluation data available for this user.")
    else:
        st.error("User ID not found in dataset.")
