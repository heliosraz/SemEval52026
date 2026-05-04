import nltk
from nltk import NLTKWordTokenizer
import json
from collections import Counter
import pandas as pd

nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")
_nltk_tokenizer = NLTKWordTokenizer()


def load_tagset(path="data/treebank_tagset.tsv"):
    df = pd.read_csv(path, sep="\t", names=["tag", "name", "example"])
    return set(df["tag"].values)


def tok_span_and_tag(text: str):
    spans = list(_nltk_tokenizer.span_tokenize(text))
    tokens = [text[s:e] for s, e in spans]  # slice tokens from spans directly
    tags = nltk.tag.pos_tag(tokens)
    return spans, [t[1] for t in tags]


def tok_span(text: str):
    return list(_nltk_tokenizer.span_tokenize(text))


def tok_tag(text: str):
    return nltk.tag.pos_tag(nltk.tokenize.word_tokenize(text))


if __name__ == "__main__":
    homonym_counts = Counter()
    with open("data/test.json") as f:
        test_json = json.load(f)
        for inst in test_json:
            for field in [
                "judged_meaning",
                "precontext",
                "sentence",
                "ending",
                "example_sentence",
            ]:
                tags = tok_tag(test_json[inst][field])
                spans = tok_span(test_json[inst][field])
                test_json[inst][field] = {
                    "content": test_json[inst][field],
                    "tags": {str(span): tok_tag for tok_tag, span in zip(tags, spans)},
                }

            hom = test_json[inst]["homonym"]
            # print(hom)
            # get homonym tag from sentence and put it in the homonym field
            # there's definitely a smarter way to do this w/o list comp but i cant think of it rn
            test_json[inst]["homonym"] = [
                x for x in test_json[inst]["sentence"]["tags"].values() if x[0] == hom
            ][0]
            homonym_counts[test_json[inst]["homonym"][1]] += 1

    with open("data/test_tags.json", "w") as f:
        json.dump(test_json, f, indent=4)
    print(homonym_counts, homonym_counts.total())
