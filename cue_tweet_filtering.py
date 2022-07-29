import numpy as np
import pandas as pd

#put name of the file containing cue tweets
cue_tweets = "data.csv"

df = pd.read_csv(cue_tweets)

df = df[(df.text.str.contains('i was')) | (df.text.str.contains('i am')) |
      (df.text.str.contains('she was')) | (df.text.str.contains('he was')) |
      (df.text.str.contains('she is')) | (df.text.str.contains('he is')) |
      (df.text.str.contains('she\'s')) | (df.text.str.contains('he\'s')) |
      (df.text.str.contains('you\'re')) | (df.text.str.contains('you are')) |
      (df.text.str.contains('they were')) | (df.text.str.contains('you were')) |
      (df.text.str.contains('they are')) | (df.text.str.contains('they\'re'))]

df.to_csv("filtered.csv", index=False)

