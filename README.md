# sensecnn
Integrating word senses into downstream NLP applications

```
python topic_categorization/__main__.py bbc word 5
python topic_categorization/__main__.py bbc supersense_wn 5
python topic_categorization/__main__.py bbc wn 5
python topic_categorization/__main__.py bbc supersense_bn 5
python topic_categorization/__main__.py bbc bn 5

python topic_categorization/__main__.py ohsumed word 23 --vocabsize 40000
python topic_categorization/__main__.py ohsumed supersense_wn 23 --vocabsize 40000
python topic_categorization/__main__.py ohsumed wn 23 --vocabsize 40000
python topic_categorization/__main__.py ohsumed supersense_bn 23 --vocabsize 40000
python topic_categorization/__main__.py ohsumed bn 23 --vocabsize 40000

python sentiment_analysis/__main__.exp.py PL04 word
python sentiment_analysis/__main__.exp.py PL04 wn
python sentiment_analysis/__main__.exp.py PL04 supersense_wn
python sentiment_analysis/__main__.exp.py PL04 bn
python sentiment_analysis/__main__.exp.py PL04 supersense_bn
```

# Datasets (compiled)

## Polarity detection:

* [RTC](https://drive.google.com/drive/folders/0Bz40_IukD5qDdnlGSTl5LW4wcVE?usp=sharing)
* [PL04](https://drive.google.com/drive/folders/0Bz40_IukD5qDZUhVbXRHeTcwTFU?usp=sharing)
* [IMDB](https://drive.google.com/drive/folders/0Bz40_IukD5qDRDJ5NHhSUTBEUnc?usp=sharing)

## Topic categorization

* [Ohsumed](https://drive.google.com/drive/folders/0Bz40_IukD5qDcmVKcm5rNVA1LWs?usp=sharing)
* [BBC](https://drive.google.com/drive/folders/0Bz40_IukD5qDX3Y2c3RvRWhPenM?usp=sharing)
* [20K](https://drive.google.com/drive/folders/0Bz40_IukD5qDbURfbEdTUnZELWc?usp=sharing)


# Embeddings

* [Deconf WordNet embeddings](https://drive.google.com/a/di.uniroma1.it/file/d/0Bz40_IukD5qDbzJFSmppN0Y2djg/view?usp=sharing)
* [Nasari Deconf embeddings (.bin)](https://drive.google.com/a/di.uniroma1.it/file/d/0Bz40_IukD5qDZ3lsemhSVTBLeFU/view?usp=sharing)
* [Nasari Deconf embeddings (.txt)](https://drive.google.com/a/di.uniroma1.it/file/d/0Bz40_IukD5qDLWdCNUJhQ3gtMzA/view?usp=sharing)

Please note that these embeddings live in the same semantic space of [Word2vec trained on the Google News dataset](https://code.google.com/archive/p/word2vec/). 

# Mapping files

* [BabelNet supersense mappings](https://drive.google.com/a/di.uniroma1.it/file/d/0Bz40_IukD5qDbTNQWGxFdDdZNnM/view?usp=sharing)
* [WordNet supersense mappings](https://drive.google.com/file/d/0B-ZmZ8m4R_BaNWFBaEhIMmRha1U/view?usp=sharing)


