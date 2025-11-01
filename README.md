# Movie-Recommender-System
# 🎬 SRK Movie Recommendation System
👩‍💻 **Author:** Tuba Mariyam  
📊 **Project Type:** Machine Learning (NLP - Natural Language Processing)
---
## 🧠 Overview
A fun and beginner-friendly **Bollywood Movie Recommender System** that suggests similar movies based on their **storylines** and **descriptions** — specially featuring **Shah Rukh Khan’s iconic films** ❤️  

This project uses **TF-IDF Vectorization** and **Cosine Similarity** from Scikit-Learn to understand how close movie plots are to each other and recommend the **Top 5 most similar movies**.
---
## 🧰 Technologies Used
| Tool | Purpose |
|------|----------|
| **Python** | Programming Language |
| **Pandas** | For data handling |
| **Scikit-Learn** | For text vectorization and similarity |
| **VS Code** | For writing and running the code |
---
## 🎬 Dataset
This project uses a small, custom Bollywood dataset with 10 famous SRK movies:  

- Chennai Express  
- Dilwale  
- Raees  
- Jab Tak Hai Jaan  
- Pathaan  
- My Name is Khan  
- Om Shanti Om  
- Kal Ho Naa Ho  
- Kabhi Khushi Kabhie Gham  
- Veer-Zaara  

Each movie has a short **description** used to calculate similarity.

---

## 💻 Code Implementation

```python
# 🎥 SRK Movie Recommender System
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
````

---

## 🧮 Example Output

```
🎥 Welcome to the SRK Movie Recommender System!

Enter a movie name: Chennai Express

📽️ Because you liked 'Chennai Express', you might also enjoy:
1. Dilwale
2. Raees
3. Pathaan
4. Jab Tak Hai Jaan
5. Om Shanti Om
```

---

## 🚀 How to Run the Project

1. **Clone this repository**

   ```bash
   git clone https://github.com/TubaMariyam/SRK-Movie-Recommendation-System.git
   ```
2. **Install the required libraries**

   ```bash
   pip install pandas scikit-learn
   ```
3. **Run the program**

   ```bash
   python bollywood_movie_recommender.py
   ```

---

## 💡 Key Concepts

| Concept                                                  | Explanation                                                                              |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **TF-IDF (Term Frequency - Inverse Document Frequency)** | Converts text into numbers based on how important each word is in the movie description. |
| **Cosine Similarity**                                    | Measures how similar two movie descriptions are (closer to 1 = more similar).            |

---

## 🌟 Insights

* Movies with similar *themes and emotions* are grouped together.
* The model easily identifies *romantic*, *action*, and *drama* patterns in movie plots.
* You can expand this system with a bigger dataset from IMDb or Kaggle.

---

## 🏁 Future Scope

✅ Add 100+ Bollywood movies.
✅ Include **genre**, **actors**, and **ratings** for better results.
✅ Create a web app using **Streamlit** or **Flask** with movie posters.

---

## 📈 Expected Accuracy

Since this is a **content-based system**, it doesn’t give an accuracy score —
instead, it provides **similarity-based recommendations** that improve as the dataset grows.

---

## 🧩 Project Summary

| Step               | What Happens                    | Why It's Needed                           |
| ------------------ | ------------------------------- | ----------------------------------------- |
| Import Libraries   | Brings ML tools into Python     | To use text/vectorization features        |
| Create Dataset     | Adds movie names & descriptions | To give the model something to learn from |
| TF-IDF             | Turns words into numbers        | Computers understand numbers, not text    |
| Cosine Similarity  | Compares two movies             | Finds which are most alike                |
| Recommend Function | Builds logic for user input     | Displays top 5 similar movies             |
| Output             | Shows recommendation list       | Final user interaction step               |

---

## 🌈 About the Author

**Tuba Mariyam**
🎓 Data Science & UI/UX Enthusiast
💻 Passionate about learning, designing, and building intelligent systems
🌐 GitHub: [github.com/TubaMariyam](https://github.com/TubaMariyam)
---
## 🎉 End Note
Thank you for exploring this project!
Every line of code is written with learning, love, and lots of Shah Rukh Khan magic ❤️

✨ *“Don’t underestimate the power of coding... or Shah Rukh Khan!”* ✨
----
