import matplotlib.pyplot as plt
from wordcloud import WordCloud


def plot_class_distribution(labels, out_path=None, show=False):
    counts = labels.value_counts()
    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    if out_path:
        plt.savefig(out_path)
    if show:
        plt.show()


def plot_length_histogram(texts, out_path=None, show=False):
    lengths = [len(t.split()) for t in texts]
    plt.figure(figsize=(10, 5))
    plt.hist(lengths, bins=50)
    plt.title("Histogram of Text Lengths")
    plt.xlabel("Num Words")
    plt.ylabel("Frequency")
    if out_path:
        plt.savefig(out_path)
    if show:
        plt.show()


def build_wordcloud_from_texts(texts, out_path=None, show=False):
    text = " ".join(texts)
    wc = WordCloud(width=1200, height=600).generate(text)
    plt.figure(figsize=(14, 7))
    plt.imshow(wc)
    plt.axis("off")
    if out_path:
        plt.savefig(out_path)
    if show:
        plt.show()
