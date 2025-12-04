from pathlib import Path

projectRootPath = Path(__file__).resolve().parents[1]
datasetDirectory = projectRootPath / "data"
artifactsDirectory = projectRootPath / "artifacts"

artifactsDirectory.mkdir(exist_ok=True)

rawArticlesFilePath = datasetDirectory / "Articles.csv"

articleTextColumn = "Article"        
articleTitleColumn = "Heading"      
articleIColumn = None # since the dataset does not contain unique IDs
articleDateColumn = "Date"      
articleCategoryColumn = "NewsType"  

maxArticlesToIndex = None
