from pyspark import SparkContext, SparkConf
from pyspark.sql import *
# from pyspark.sql import SQLContext
# from functools import reduce
# from pyspark.sql.functions import col, lit, when
from pyspark.sql.types import StructType, IntegerType, FloatType, StringType
from graphframes import *


def main():
    cnfg = SparkConf().setAppName("project").setMaster("local[*]")
    cnfg.set("spark.executor.memory", "10g")
    sc = SparkContext(conf=cnfg)
    spark = SparkSession(sc)

    edgesPath = "C:\\Python\\BDEWAProject\\edgedf_graph.csv"
    vertexPath = "C:\\Python\\BDEWAProject\\vertexdf.csv"
    moviesPath = "C:\\Python\\BDEWAProject\\moviereviews.csv"

    edgedf = (spark.read
              .option("header", "true")
              .option("inferSchema", "true")
              .csv(edgesPath))

    vertexdf = (spark.read
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(vertexPath))

    schema_m = (StructType().
                add("src", IntegerType()).
                add("src_name", StringType()).
                add("dst", IntegerType()).
                add("dst_name", StringType()).
                add("rating", FloatType())
                )
    moviesdf = (spark.read.schema(schema=schema_m).csv(moviesPath, header=True))
    moviesdf.printSchema()

    vertexdf = vertexdf.withColumnRenamed("Reviewer", "name")


    g = GraphFrame(vertexdf, edgedf)
    print(g)
    print("Vertices of graph")
    g.vertices.show(20)
    print("Edges of graph")
    g.edges.show(20)

    print("In degrees - Movies sorted by number of reviews")
    g.inDegrees.sort(['inDegree'], ascending=[0]).show(20)

    print("Out degrees")
    g.outDegrees.sort(['outDegree'], ascending=[0]).show(20)

    print('Degrees')
    g.degrees.sort(['degree'], ascending=[0]).show(20)

    numhigh = g.edges.filter("rating > 8").count()
    print("The number of reviews with a score more than 8 is", numhigh)

    # motifs = g.find("(a)-[e]->(b)")  # (c2)-[r2]->(m1)
    # # motifs.filter("c1.id != c2.id").show(20)
    # motifs.show(20)


if __name__ == '__main__':
    main()
