from flask import Flask, request, render_template_string
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# 1. Load your dataset and the trained DBSCAN model
try:
    df = pd.read_csv('movies_metadata.csv')
    with open('model.pkl', 'rb') as f:
        dbscan = pickle.load(f)
    
    # Attach the trained cluster labels directly to the dataframe
    df['cluster'] = dbscan.labels_[cite: 1]
except Exception as e:
    print(f"Error loading files, utilizing fallback mock data: {e}")
    # Fallback mock data for demonstration if files aren't present yet
    df = pd.DataFrame({
        'title': ['Inception', 'Interstellar', 'The Dark Knight', 'The Hangover', 'Superbad'],
        'cluster': [0, 0, 0, 1, 1],
        'imdb_score': [8.8, 8.6, 9.0, 7.7, 7.6]
    })

def get_recommendations(movie_title, top_n=5):
    # Find the movie in the dataset (case-insensitive check)
    movie_row = df[df['title'].str.lower() == movie_title.lower()]
    
    if movie_row.empty:
        return None, "Movie not found in the database."
    
    # Get the cluster of the selected movie
    movie_cluster = movie_row.iloc[0]['cluster']
    movie_id = movie_row.index[0]
    
    # Handle noise (-1 by DBSCAN) or valid clusters[cite: 1]
    if movie_cluster == -1:[cite: 1]
        similar_movies = df[df['cluster'] != -1][cite: 1]
    else:
        similar_movies = df[df['cluster'] == movie_cluster]
        
    # Exclude the input movie itself
    similar_movies = similar_movies.drop(movie_id, errors='ignore')
    
    # Sort recommendations by quality
    sort_column = 'imdb_score' if 'imdb_score' in df.columns else df.columns[1]
    similar_movies = similar_movies.sort_values(by=sort_column, ascending=False)
    
    return similar_movies[['title', 'cluster']].head(top_n).to_dict(orient='records'), None

# HTML/Tailwind CSS Frontend Blueprint
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CineMatch | AI Movie Recommendations</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <style>
        body { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); }
    </style>
</head>
<body class="text-slate-100 min-h-screen font-sans">
    <div class="container mx-auto px-4 py-12 max-w-4xl">
        <header class="text-center mb-12">
            <h1 class="text-5xl font-black tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-fuchsia-500 mb-2">
                CineMatch
            </h1>
            <p class="text-slate-400 text-lg">DBSCAN Clustering Powered Recommendations</p>
        </header>

        <div class="bg-slate-900/60 backdrop-blur-md border border-slate-800 p-8 rounded-2xl shadow-xl mb-8">
            <form action="/recommend" method="POST" class="space-y-6">
                <div>
                    <label for="movie_title" class="block text-sm font-medium text-slate-300 mb-2">
                        Type or Select a Movie
                    </label>
                    <input 
                        list="movie-options" 
                        id="movie_title" 
                        name="movie_title" 
                        placeholder="e.g. Inception..." 
                        value="{{ selected_movie if selected_movie else '' }}"
                        class="w-full bg-slate-950/80 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-cyan-500 placeholder-slate-500 transition-all"
                        required
                    >
                    <datalist id="movie-options">
                        {% for movie in movies %}
                            <option value="{{ movie }}">
                        {% endfor %}
                    </datalist>
                </div>
                <button type="submit" class="w-full bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-semibold py-3 px-6 rounded-xl transition duration-200 shadow-lg shadow-cyan-500/20 cursor-pointer">
                    Discover Similar Movies
                </button>
            </form>

            {% if error %}
                <div class="mt-4 bg-red-900/30 border border-red-500/40 text-red-200 px-4 py-3 rounded-xl text-sm">
                    {{ error }}
                </div>
            {% endif %}
        </div>

        {% if recommendations %}
            <div class="space-y-6">
                <h2 class="text-2xl font-bold tracking-wide border-b border-slate-800 pb-2 flex items-center gap-2">
                    <span>Results for:</span> 
                    <span class="text-cyan-400 italic">"{{ selected_movie }}"</span>
                </h2>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {% for rec in recommendations %}
                        <div class="bg-slate-900/40 border border-slate-800/80 hover:border-cyan-500/50 p-5 rounded-xl transition-all flex items-center justify-between group">
                            <div>
                                <h3 class="font-bold text-lg text-slate-200 group-hover:text-white transition-colors">
                                    {{ rec.title }}
                                </h3>
                                <p class="text-xs text-slate-500 mt-1">Matched Cluster ID: {{ rec.cluster }}</p>
                            </div>
                            <div class="bg-cyan-950 text-cyan-400 text-xs px-3 py-1.5 rounded-full font-semibold uppercase tracking-wider">
                                Match
                            </div>
                        </div>
                    {% endfor %}
                </div>
            </div>
        {% elif selected_movie and not error %}
            <div class="text-center py-12 text-slate-500">
                No sister movies found in this specific cluster segment.
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    movie_list = df['title'].dropna().unique().tolist()
    return render_template_string(HTML_TEMPLATE, movies=movie_list)

@app.route('/recommend', methods=['POST'])
def recommend():
    selected_movie = request.form.get('movie_title')
    movie_list = df['title'].dropna().unique().tolist()
    
    if not selected_movie:
        return render_template_string(HTML_TEMPLATE, error="Please select a movie.", movies=movie_list)
        
    recommendations, error = get_recommendations(selected_movie)
    
    return render_template_string(HTML_TEMPLATE, 
                                  movies=movie_list, 
                                  selected_movie=selected_movie, 
                                  recommendations=recommendations, 
                                  error=error)

if __name__ == '__main__':
    app.run(debug=True)
