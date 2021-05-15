from typing import Tuple, List

Annotation = Tuple[int, int, str, str, str]
Article = Tuple[str, str, List[Annotation]]


def parse_article_data(data: List[str]) -> Article:
    """Parse an article.

    :param data: A list of strings. The first item is always the text of the article.
    :return A tuple containing the article ID, the article text, and a list of annotations.

    """
    text = []
    split = data[0].strip().split("|")
    identifier = split[0]
    text.extend(split[2:])
    text.extend(data[1].strip().split("|")[2:])
    text = " ".join(text)
    annotations = []
    for item in data[2:]:
        _, s, e, txt, sty, cui = item.strip().split("\t")
        # Only take the first label for now.
        label = sty.split(",")[0]
        if sty == "UnknownType":
            continue
        annotations.append(((int(s), int(e)), label))
    return identifier, text, annotations


def read_pubtator(path: str) -> List[Article]:
    """
    Reads a file in pubtator format and returns articles.

    An article is a triple with (id, text, annotations), where
    annotations is a list of tuples with (start index, end index, text, label, label).

    :param path: The path to the pubtator file to read.
    :returns: A list of articles.

    """
    articles = []
    with open(path) as f:
        article = []
        for line in f:
            if not line.strip():
                articles.append(parse_article_data(article))
                article = []
                continue
            article.append(line)
        else:
            if article:
                articles.append(parse_article_data(article))

    return articles
