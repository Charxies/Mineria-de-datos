from wordcloud import WordCloud
from collections import Counter
import matplotlib.pyplot as plt


def open_file(path: str) -> str:
    content = ""
    with open(path, "r",encoding="utf8") as f:
        content = f.readlines()
    return " ".join(content)


all_words = ""
frase = open_file("artistas.txt")
palabras = frase.rstrip().split(" ")

Counter(" ".join(palabras).split()).most_common(10)
# looping through all incidents and joining them to one text, to extract most common words
for arg in palabras:
    tokens = arg.split()
    all_words += " ".join(tokens) + " "

print(all_words)
wordcloud = WordCloud(
    background_color="white", min_font_size=5
).generate(all_words)

# print(all_words)
# plot the WordCloud image
plt.close()
plt.figure(figsize=(5, 5), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

# plt.show()
plt.savefig("practica10.png")
plt.close()