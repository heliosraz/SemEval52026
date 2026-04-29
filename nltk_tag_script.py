import nltk
import json
from collections import Counter

nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")


def tok_tag(text: str):
    return nltk.tag.pos_tag(nltk.tokenize.word_tokenize(text))


if __name__ == "__main__":
    homonym_counts = Counter()
    with open("test.json") as f:
        test_json = json.load(f)
        for inst in test_json:
            for field in [
                "judged_meaning",
                "precontext",
                "sentence",
                "ending",
                "example_sentence",
            ]:
                test_json[inst][field] = tok_tag(test_json[inst][field])
            hom = test_json[inst]["homonym"]
            # print(hom)
            # get homonym tag from sentence and put it in the homonym field
            # there's definitely a smarter way to do this w/o list comp but i cant think of it rn
            test_json[inst]["homonym"] = [
                x for x in test_json[inst]["sentence"] if x[0] == hom
            ][0]
            homonym_counts[test_json[inst]["homonym"][1]] += 1

    with open("test_tags.json", "w") as f:
        json.dump(test_json, f)
    print(homonym_counts, homonym_counts.total())
