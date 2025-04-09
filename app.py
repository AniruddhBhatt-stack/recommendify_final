from flask import Flask, redirect, request, session, render_template
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import os
import spotipy
import pickle
import pandas as pd
import re
from datetime import datetime, timedelta
import logging

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")
app.logger.setLevel(logging.INFO)

DEFAULT_GENRES = ['pop', 'indie', 'hip hop', 'rap', 'desi hip hop', 'indian indie']
GENRE_CACHE = {'genres': [], 'expires_at': None}

# 🎯 Get OAuth instance using a single shared cache file (.cache)
def get_spotify_oauth():
    cache_file = ".cache"
    return SpotifyOAuth(
        client_id=os.getenv("CLIENT_ID"),
        client_secret=os.getenv("CLIENT_SECRET"),
        redirect_uri=os.getenv("REDIRECT_URI"),
        scope="user-top-read,playlist-modify-private,user-read-private",
        cache_path=cache_file,
        show_dialog=True
    )

def refresh_genre_cache(sp):
    try:
        seeds = sp.recommendation_genre_seeds().get('genres', [])
        GENRE_CACHE.update({'genres': seeds, 'expires_at': datetime.now() + timedelta(hours=1)})
    except Exception as e:
        app.logger.warning(f"Genre cache refresh failed: {e}")

def get_available_genres():
    return GENRE_CACHE.get('genres', DEFAULT_GENRES)

def get_valid_genres(sp):
    if not GENRE_CACHE['expires_at'] or datetime.now() >= GENRE_CACHE['expires_at']:
        refresh_genre_cache(sp)
    return get_available_genres()

def get_token():
    token_info = session.get('token_info')
    if not token_info:
        return None
    sp_oauth = get_spotify_oauth()
    if sp_oauth.is_token_expired(token_info):
        try:
            token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
            session['token_info'] = token_info
            sp_oauth._save_token_info(token_info)
        except Exception as e:
            app.logger.warning(f"Token refresh failed: {e}")
            session.pop('token_info', None)
            return None
    return token_info

def extract_all_genres():
    try:
        df = pd.read_csv("song.csv")
        df.columns = df.columns.str.lower().str.strip()
        col = 'genres' if 'genres' in df.columns else 'genre' if 'genre' in df.columns else None
        if not col:
            return DEFAULT_GENRES
        all_g = set()
        for entry in df[col].dropna().astype(str):
            parts = re.split(r'[,/;\\|]', entry)
            for g in parts:
                g = g.strip().lower()
                if g:
                    all_g.add(g)
        return sorted(all_g)
    except Exception as e:
        app.logger.warning(f"Failed to load genres from CSV: {e}")
        return DEFAULT_GENRES

def enrich_with_spotify(sp, name, artist):
    try:
        res = sp.search(q=f"track:{name} artist:{artist}", type="track", limit=1)
        items = res.get('tracks', {}).get('items', [])
        if items:
            item = items[0]
            img = item['album']['images'][0]['url'] if item['album']['images'] else None
            url = item['external_urls']['spotify']
            uri = item['uri']
            return img, url, uri
    except Exception as e:
        app.logger.warning(f"Spotify search failed for {name} by {artist}: {e}")
    return None, None, None

@app.route("/")
def home():
    # Home page with the login prompt
    session.clear()
    return render_template("index.html")

@app.route("/login")
def login():
    # Delete any existing .cache file before starting a new login
    cache_file = ".cache"
    if os.path.exists(cache_file):
        try:
            os.remove(cache_file)
            app.logger.info(f"Removed previous cache file: {cache_file}")
        except Exception as e:
            app.logger.warning(f"Failed to remove previous cache file {cache_file}: {e}")
    
    session.clear()
    sp_oauth = get_spotify_oauth()
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route("/callback")
def callback():
    code = request.args.get('code')
    sp_oauth = get_spotify_oauth()
    token_info = sp_oauth.get_access_token(code)
    session["token_info"] = token_info

    sp = spotipy.Spotify(auth=token_info['access_token'])
    user_profile = sp.current_user()

    session["user_id"] = user_profile.get('id')
    session["display_name"] = user_profile.get('display_name', 'User')

    # Create a new cache file with the token info for the new session
    get_spotify_oauth()._save_token_info(token_info)

    return redirect("/dashboard")

@app.route("/dashboard")
def dashboard():
    try:
        token_info = get_token()
        if not token_info:
            return redirect("/login")

        sp = spotipy.Spotify(auth=token_info['access_token'])

        top_tracks = sp.current_user_top_tracks(limit=10, time_range='medium_term')
        top_songs = [{
            'name': item['name'],
            'artist': item['artists'][0]['name'],
            'url': item['external_urls']['spotify'],
            'image': item['album']['images'][0]['url'] if item['album']['images'] else None
        } for item in top_tracks['items']]

        return render_template("dashboard.html",
                               top_songs=top_songs,
                               display_name=session.get('display_name', 'User'),
                               genres=extract_all_genres())
    except Exception as e:
        app.logger.error(f"Dashboard error: {e}")
        return redirect("/login")

@app.route("/recommend", methods=["POST"])
def recommend():
    token_info = get_token()
    if not token_info:
        return redirect("/login")
    sp = spotipy.Spotify(auth=token_info['access_token'])

    selected = [request.form.get(f"genre{i}") for i in range(1, 6)]
    selected = [g.strip().lower() for g in selected if g]

    predicted = None
    if os.path.exists("model.pkl") and selected:
        try:
            with open("model.pkl", "rb") as f:
                data = pickle.load(f)
                model, mlb = data['model'], data['mlb']
            vec = mlb.transform([selected])
            pred = model.predict(vec)[0]
            predicted = mlb.inverse_transform([pred])[0][0]
        except Exception as e:
            app.logger.error(f"Model predict error: {e}")

    df = pd.read_csv("song.csv") if os.path.exists("song.csv") else pd.DataFrame()
    df.columns = df.columns.str.lower().str.strip()
    if 'name' in df.columns:
        df = df.rename(columns={'name': 'track_name'})
    if {'track_name', 'artists', 'genres'}.issubset(df.columns):
        df['genres_list'] = df['genres'].fillna('').apply(
            lambda x: [g.strip().lower() for g in re.split(r'[,/;\\|]', x)]
        )
    else:
        df = pd.DataFrame(columns=['track_name', 'artists', 'genres_list'])

    pred_songs, seen = [], set()
    if predicted:
        mask = df['genres_list'].apply(lambda gl: predicted in gl)
        for _, r in df[mask].iterrows():
            pred_songs.append({'track_name': r['track_name'], 'artists': r['artists']})

    seen = {(s['track_name'], s['artists']) for s in pred_songs}
    sel_songs = []
    for g in selected:
        mask = df['genres_list'].apply(lambda gl: g in gl)
        for _, r in df[mask].iterrows():
            key = (r['track_name'], r['artists'])
            if key not in seen:
                sel_songs.append({'track_name': r['track_name'], 'artists': r['artists']})
                seen.add(key)

    combined = (pred_songs + sel_songs)[:10]
    recs, uris = [], []
    for s in combined:
        img, url, uri = enrich_with_spotify(sp, s['track_name'], s['artists'])
        recs.append({
            'name': s['track_name'],
            'artist': s['artists'],
            'album_image': img,
            'external_url': url
        })
        if uri:
            uris.append(uri)

    playlist_url = None
    try:
        user_id = sp.me()['id']
        playlist = sp.user_playlist_create(user_id, name="aniruddh's love", public=False)
        if uris:
            sp.playlist_add_items(playlist['id'], uris)
        playlist_url = playlist['external_urls']['spotify']
    except Exception as e:
        app.logger.error(f"Playlist creation failed: {e}")

    top_tracks = []
    try:
        results = sp.current_user_top_tracks(limit=10, time_range='medium_term')
        for item in results['items']:
            img = item['album']['images'][0]['url'] if item['album']['images'] else None
            url = item['external_urls']['spotify']
            top_tracks.append({
                'name': item['name'],
                'artist': item['artists'][0]['name'],
                'album_image': img,
                'external_url': url
            })
    except Exception as e:
        app.logger.warning(f"Could not fetch top tracks: {e}")

    return render_template("dashboard.html",
                           top_tracks=top_tracks,
                           genres=extract_all_genres(),
                           selected_genres=selected,
                           prediction=predicted,
                           tracks=recs,
                           playlist_url=playlist_url,
                           display_name=session.get('display_name', 'User'))

@app.route("/logout", methods=["POST"])
def logout():
    # Clear session data
    session.clear()

    # Delete the .cache file associated with the session
    cache_file = ".cache"
    if os.path.exists(cache_file):
        try:
            os.remove(cache_file)
            app.logger.info(f"Deleted cache file: {cache_file}")
        except Exception as e:
            app.logger.warning(f"Cache cleanup failed for {cache_file}: {e}")

    # Redirect to login/index page
    return redirect("/")

if __name__ == "__main__":
    app.run(host="localhost", port=5003, debug=True)
