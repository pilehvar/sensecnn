# sensecnn
Integrating word senses into downstream NLP applications

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
