# Wikipedia Quality Assessment

The library provides a set of tools for quality assessment of Wikipedia articles. The library is based on the paper Assessing the Quality of Infomration on Wikipedia: A Deep Learning Approch by [Wang & Li, 2019](https://doi.org/10.1002/asi.24210) with some modifications.


For better results based on mediawiki please check out [articlequality](https://github.com/wikimedia/articlequality)


## Installation
clone the repository and install the requirements
```bash
git clone https://github.com/MohammadSakhnini/WikipediaTools.git
cd WikipediaTools
pip install -r requirements.txt
```

## Predicting the quality of a Wikipedia article
```python
from wikipediatools.pagequality import QualityPredictor, FeatureSets, ModelType

# Create the best model
predictor = QualityPredictor()

# Select a feature set and a model
# predictor = Wikipredictor(FeaturesSet.STRUCTURE_FEATURES, ModelType.DeepNeuralNetwork)

# Or just the feature set, which will automatically select the best model
# predictor = Wikipredictor(FeaturesSet.READABILITY_SCORES)

# Predict using the title, which will use the the latest revision.
print(predictor.predict_quality("SpongeBob SquarePants"))
>>> High

# Or predict using the revision ID.
print(predictor.predict_quality(1171275655))
>>> High

```

# Generate new feature set
```python
from wikipediatools.data import SampleGenerator, FeatureSets
from zipfile import ZipFile
import pandas as pd

with ZipFile('dataset.zip', 'r') as file:
    df = pd.read_csv(file.open("raw/dataset.csv"))

# Generate an evenly distributed dataset that contains 50 articles per grading
generator = SampleGenerator(grading_dataset=df, size_per_grading=50, random_state=99)
generator.get_features("path/to/save", FeatureSets.TEXT_STATISTICS)
# for multiple feature sets
# generator.get_features("path/to/save", [FeatureSets.TEXT_STATISTICS, FeatureSets.STRUCTURE_FEATURES])
# or simply use FeaturesSet.ALL
```
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

