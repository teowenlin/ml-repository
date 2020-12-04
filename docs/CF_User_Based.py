from pyspark import SparkContext, SparkConf
from pyspark.sql import *
from pyspark.sql.functions import first
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.sql.types import StructType, IntegerType, FloatType, StringType
from graphframes import *
# from org.neo4j.spark import *

def main():
    cnfg = SparkConf().setAppName("project").setMaster("local[*]")
    cnfg.set("spark.driver.memory", "18g")
    sc = SparkContext(conf=cnfg)
    spark = SparkSession(sc)

    ratingsPath = "C:\\Python\\BDEWAProject\\ratingsdf.csv"
    vertexPath = "C:\\Python\\BDEWAProject\\vertexdf.csv"
    moviesPath = "C:\\Python\\BDEWAProject\\moviereviews.csv"

    vertexdf = (spark.read
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(vertexPath))

    schema_r = (StructType().
              add("src", IntegerType()).
              add("dst", IntegerType()).
              add("rating", FloatType())
              )

    schema_m = (StructType().
                add("src", IntegerType()).
                add("src_name", StringType()).
                add("dst", IntegerType()).
                add("dst_name", StringType()).
                add("rating", FloatType())
                )
    vertexdf = vertexdf.withColumnRenamed("Reviewer", "name")

    ratingsdf = (spark.read.schema(schema=schema_r).csv(ratingsPath, header=True))
    ratingsdf.printSchema()

    moviesdf = (spark.read.schema(schema=schema_m).csv(moviesPath, header=True))
    moviesdf.printSchema()

    adj_df = (ratingsdf.groupBy("src")
              .pivot("dst")
              .agg(first("rating"))
              .fillna(0))

    rdd = adj_df.rdd.map(list)
    adj_mat = RowMatrix(rdd)

    # Calculate similarities
    exact = adj_mat.columnSimilarities()

    # Output
    sim_df = exact.entries.toDF(["src", "dst", "sim"])

    # Analyse with graphframes
    g = GraphFrame(vertexdf, sim_df) # creates directed graph
    print(g)
    print("Vertices of graph")
    g.vertices.show(20)
    print("Edges of graph")
    g.edges.show(20)
    numhigh = g.edges.filter("sim > 0.5").count()
    print("The number of pairs of reviewers with similarity more than 0.5 is ", numhigh)

    print("Critics who are more than 0.5 similar")

    motifs = g.find("(Reviewer1)-[similarity]->(Reviewer2)").filter("similarity.sim > 0.5")
    motifs.show(20)

    # Find critics who are > 0.5 similar; neighbours who are > 0.5 similar

    print("Second degree neighbours who are more than 0.3 similar to the middle neighbour")

    motifs2 = g.find("(a)-[e]->(b); (b)-[e2]->(c)")\
        .filter("e.sim > 0.3 AND e2.sim > 0.3")
    motifs2.show(20)

    # Write to csv files
    # sim_df.write.format("csv").save("C:\\Python\\BDEWAProject\\similarity_matrix")
    # print("Similarity matrix saved to file")

if __name__ == '__main__':
    main()