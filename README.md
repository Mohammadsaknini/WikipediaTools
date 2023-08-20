# Wikipedia Quality Assessment

The library provides a set of tools for quality assessment of Wikipedia articles. The library is based on the paper Assessing the Quality of Infomration on Wikipedia: A Deep Learning Approch by [Wang & Li, 2019](https://doi.org/10.1002/asi.24210) with some modifications.


For better results based on mediawiki please check out [articlequality](https://github.com/wikimedia/articlequality)


## Installation
clone the repository and install the requirements
```bash
git clone https://github.com/MohammadSakhnini/Wikipedia-Quality-Assessment.git
cd Wikipedia-Quality-Assessment
pip install -r requirements.txt
```

## Usage
```python
from wikipredictor import Wikipredictor

predictor = Wikipredictor()
print(predictor.predict_quality("SpongeBob SquarePants"))

>>> High
```

To select a model and a feature set, use the following:
```python
from wikipredictor import Wikipredictor, FeaturesSet, ModelType
predictor = Wikipredictor(FeaturesSet.STRUCTURE_FEATURES, ModelType.DeepNeuralNetwork)
print(predictor.predict_quality("SpongeBob SquarePants"))

>>> High
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

