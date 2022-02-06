# Import the wordcloud library
from wordcloud import WordCloud
from PIL import Image
import matplotlib.pyplot as plt
#load the text data
import data_manager as dm
religion_misc_df = dm.data_manager("talk_religion_misc")
religion_misc_df.load()
# Join the different processed titles together.
long_string = ','.join(list(religion_misc_df.output()['text'].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()