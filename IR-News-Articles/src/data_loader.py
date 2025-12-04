import pandas as pd
from pathlib import Path
from .config import (
    rawArticlesFilePath, datasetDirectory,
    articleTextColumn, articleTitleColumn, articleDateColumn,
    articleCategoryColumn, maxArticlesToIndex
)

# data loading which handles the encoding of the CSV file safely

def safeReadCSV(filePath):
    """
    Attempts to read a CSV file using multiple encodings
    (UTF-8, Latin-1, Windows-1252) until one works.
    """
    encodingsToTry = ["utf-8", "latin1", "windows-1252"]

    for encodingName in encodingsToTry:
        try:
            print(f"Trying encoding: {encodingName}")
            return pd.read_csv(filePath, encoding=encodingName)
        except UnicodeDecodeError:
            print(f"Encoding failed: {encodingName}")

    raise UnicodeDecodeError("All encodings failed.")


# Although the CSV is inspected in exploration.ipynb, following code handles the scenario where the file presence is confirmed.

def inspectRawDataset():
    """Prints dataset path, available columns, and sample rows."""
    print(f"\nDataset at: {rawArticlesFilePath}")

    if not rawArticlesFilePath.exists():
        raise FileNotFoundError(
            f"CSV not found at {rawArticlesFilePath}. "
            "Place 'Articles.csv' inside /data."
        )

    dataframe = safeReadCSV(rawArticlesFilePath)

    print("\nDataset Columns:")
    print(dataframe.columns.tolist())

    print("\nSample Rows:")
    print(dataframe.head(5))


# Build clean dataset

def buildCleanDataset():
    """
    Converts raw CSV into a standard dataset with:
        documentId, title, text, date, category
    """
    print("\nLoading raw dataset...")
    dataframe = safeReadCSV(rawArticlesFilePath)

    # Maximum articles to index is set to None in config.py, else, limiting documents for debugging or development mode
    if maxArticlesToIndex is not None:
        dataframe = dataframe.iloc[:maxArticlesToIndex].copy()
        print(f"Using first {maxArticlesToIndex} articles for development.")

    dataframe["documentId"] = range(len(dataframe)) # unique document ID creation

    requiredColumns = [articleTextColumn, articleTitleColumn]
    for col in requiredColumns:
        if col not in dataframe.columns:
            raise ValueError(f"Required column '{col}' not found in dataset.") # to ensure that the required columns exist

    # Rename main fields for internal consistency
    dataframe = dataframe.rename(columns={
        articleTextColumn: "text",
        articleTitleColumn: "title"
    })

    # Optional fields
    dataframe["date"] = (
        dataframe[articleDateColumn] 
        if articleDateColumn in dataframe.columns 
        else None
    )

    dataframe["category"] = (
        dataframe[articleCategoryColumn] 
        if articleCategoryColumn in dataframe.columns 
        else None
    )

    dataframe = dataframe[["documentId", "title", "text", "date", "category"]].copy() # keeping the standardized structure

    dataframe = dataframe.dropna(subset=["text"]) # removal of the rows with missing text

    cleanedDataPath = datasetDirectory / "clean_data.csv"
    dataframe.to_csv(cleanedDataPath, index=False)

    print(f"\nClean data saved to: {cleanedDataPath}")
    print(f"Total articles processed: {len(dataframe)}")

    return dataframe


# Main function

if __name__ == "__main__":
    print("\nInspecting raw dataset!")
    inspectRawDataset()

    print("\nBuilding clean dataset!")
    buildCleanDataset()
