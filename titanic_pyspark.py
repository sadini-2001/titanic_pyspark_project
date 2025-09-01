from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, when, round
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

class TitanicSurvivalML:
    def __init__(self):
        """Initialize Spark session"""
        self.spark = SparkSession.builder \
            .appName("TitanicSurvivalPrediction") \
            .config("spark.sql.shuffle.partitions", "8") \
            .getOrCreate()

        self.dataset_path = "titanic.csv"

    def load_data(self):
        """Load Titanic dataset"""
        self.df = self.spark.read.csv(self.dataset_path, header=True, inferSchema=True)
        return self.df

    def run_eda(self):
        """Exploratory Data Analysis"""
        df = self.df

        print("\nDataset Overview:")
        print(f"Total Rows: {df.count()}, Columns: {len(df.columns)}")

        print("\nSurvival Rate by Sex:")
        df.groupBy("Sex").agg(round(avg("Survived"), 2).alias("SurvivalRate")).show()

        print("\nSurvival Rate by Passenger Class:")
        df.groupBy("Pclass").agg(round(avg("Survived"), 2).alias("SurvivalRate")).show()

        print("\nAverage Age by Survival Status:")
        df.groupBy("Survived").agg(round(avg("Age"), 2).alias("AvgAge")).show()

        print("\nFamily Size vs Survival:")
        df.withColumn("FamilySize", col("Siblings/Spouses Aboard") + col("Parents/Children Aboard")) \
          .groupBy("FamilySize").agg(round(avg("Survived"), 2).alias("SurvivalRate"),
                                     count("*").alias("Count")).orderBy("FamilySize").show()

    def preprocess(self):
        """Data cleaning & feature engineering"""
        df = self.df

        # Fill missing values
        df = df.fillna({"Age": 30, "Fare": df.approxQuantile("Fare", [0.5], 0.25)[0]})

        # Encode categorical feature: Sex
        sex_indexer = StringIndexer(inputCol="Sex", outputCol="Sex_idx")
        df = sex_indexer.fit(df).transform(df)

        # Rename columns for consistency
        df = df.withColumnRenamed("Siblings/Spouses Aboard", "SibSp") \
               .withColumnRenamed("Parents/Children Aboard", "Parch")

        # Feature assembler
        feature_cols = ["Pclass", "Sex_idx", "Age", "SibSp", "Parch", "Fare"]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

        self.df = assembler.transform(df).select("features", col("Survived").alias("label"))
        return self.df

    def build_model(self):
        """Train & evaluate ML model"""
        train, test = self.df.randomSplit([0.7, 0.3], seed=42)

        rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100)
        model = rf.fit(train)
        preds = model.transform(test)

        evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
        auc = evaluator.evaluate(preds)

        return model, auc, preds


if __name__ == "__main__":
    project = TitanicSurvivalML()

    # Load
    df = project.load_data()

    # Run EDA
    project.run_eda()

    # Preprocess & ML
    df = project.preprocess()
    model, auc, preds = project.build_model()

    print(f"\nRandomForest Titanic AUC: {auc:.3f}")
    preds.select("features", "label", "prediction", "probability").show(10, truncate=False)

