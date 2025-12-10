#IMPORT RELEVANT PACKAGES
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

#LINE 15 IS SOLE RELEVANT LINE IN get_jvke() FOR sentiment.py
def get_jvke(limit=500):
    artist = genius.search_artist(artist_name, max_songs=limit, sort="title")
    for song in artist.songs:
        sentiment = sia.polarity_scores(song.lyrics)
        data.append({
            "Title": song.title,
            "Lyrics": song.lyrics,
            "Score": sentiment['compound']
        })
    time.sleep(1)
    df = pd.DataFrame(data)
    df.to_csv("jvke2.csv", index=False, encoding="utf-8")
    print("Saved:", len(df), "songs")
    return df
get_jvke(360)

df = pd.read_csv("jvke2.csv")

#CLASSIFY VADER SCORES INTO POS/NEG/NEU CLASSES
def emotions(score):
    if score > 0.3:
        return 'POSITIVE'
    elif score < -0.3:
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'
df['Emotional Scale'] = df['Score'].apply(emotions)
df.to_csv("jvke_song_emotions.csv", index=False)

#ANALYZE IF POS/NEG/NEU SPLIT IS BALANCED
label_map = {"NEGATIVE": 0, "POSITIVE": 1, "NEUTRAL": 2}
df["Label"] = df["Emotional Scale"].map(label_map)
train_df, val_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["Label"])
print("Train label distribution")
print(train_df["Label"].value_counts(),"/n")
print("Validation label distribution")
print(val_df["Label"].value_counts())
