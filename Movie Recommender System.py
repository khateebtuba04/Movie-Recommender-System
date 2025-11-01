# 🎥 Bollywood Movie Recommender System (SRK Edition)
# Author: Tuba Mariyam

# ==============================
# 🔧 Step 1: Import Libraries
# ==============================
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# 📥 Step 2: Create Dataset
# ==============================
data = {
    'title': [
        'Chennai Express', 'Dilwale', 'Raees', 'Jab Tak Hai Jaan', 
        'Pathaan', 'My Name is Khan', 'Om Shanti Om', 
        'Kal Ho Naa Ho', 'Kabhi Khushi Kabhie Gham', 'Veer-Zaara'
    ],
    'description': [
        'A man’s journey from Mumbai to Rameswaram turns into an unexpected adventure with a South Indian woman.',
        'Two lovers are separated by family rivalry and reunite years later amid chaos and comedy.',
        'A bootlegger becomes a powerful man who challenges corruption and fights for justice.',
        'An army officer falls in love with a woman, but destiny separates them in a tale of love and sacrifice.',
        'An undercover spy returns to protect his country and take revenge against enemies.',
        'A Muslim man with Asperger’s syndrome embarks on a journey to meet the President of America.',
        'A struggling actor falls in love with a superstar and discovers the mystery behind her death.',
        'A cheerful young man spreads happiness while hiding his illness from his loved ones.',
        'A rich family’s relationships are tested through love, pride, and forgiveness.',
        'An Indian Air Force officer falls in love with a Pakistani woman during a mission.'
    ]
}

df = pd.DataFrame(data)
print("✅ Dataset Loaded Successfully!\n")
print(df)

# ==============================
# 🧹 Step 3: Convert Text to Numbers
# ==============================
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])

# ==============================
# 🧮 Step 4: Calculate Similarity
# ==============================
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ==============================
# 🔍 Step 5: Recommendation Function
# ==============================
def recommend_movie(movie_title):
    if movie_title not in df['title'].values:
        return "❌ Movie not found! Try another one."
    
    idx = df[df['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_movies = [df.iloc[i[0]]['title'] for i in sim_scores[1:6]]
    return top_movies

# ==============================
# 🎬 Step 6: Try It Out
# ==============================
print("\n🎥 Welcome to the SRK Movie Recommender System!\n")
movie_name = input("Enter a movie name: ")

recommendations = recommend_movie(movie_name)
print(f"\n📽️ Because you liked '{movie_name}', you might also enjoy:\n")

if isinstance(recommendations, list):
    for i, movie in enumerate(recommendations, start=1):
        print(f"{i}. {movie}")
else:
    print(recommendations)
