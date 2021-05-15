# quickumls_pred

This is the repo accompanying the blog post "Doing almost as much with much less: a case study in biomedical named entity recognition". In short, we use [QuickUMLS](https://github.com/Georgetown-IR-Lab/QuickUMLS) to classify ambiguous named entities from text into a pre-specified set of semantic types. While the baseline performance of QuickUMLS is pretty good, we show that it can be a lot better with some extra tweaks.

For the full information, please take a look at the blog post [here]().

## Usage

Install all the requirements in `requirements.txt`. The version numbers for `numpy` and `spacy` have been omitted. This code should work with either spacy 2 or 3.

Then, using QuickUMLS, you should create a QuickUMLS installation somewhere. For instructions on how to do that, see [here](https://github.com/Georgetown-IR-Lab/QuickUMLS#installation). Once you've done this, you're all set.

To reproduce our results, you can download the Medmentions corpus [here](https://github.com/chanzuckerberg/MedMentions). If you put the MedMentions corpus files in `data/`, the file `run.py` should run out of the box.  

## Author

Stéphan Tulkens

## License

MIT
