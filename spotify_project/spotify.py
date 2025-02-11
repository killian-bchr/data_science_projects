import spotipy
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import warnings
from requests.exceptions import JSONDecodeError
from scipy import stats
from datetime import datetime
from wordcloud import WordCloud
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from config import (
    api_key_lastfm,
    client_secret_lastfm,
    client_id,
    client_secret,
    redirect_uri
)


### Spotify Authentication
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id = client_id,
    client_secret = client_secret,
    redirect_uri = redirect_uri,
    scope = "user-top-read user-read-private user-library-read playlist-read-private user-read-playback-state user-read-recently-played"
))

### LastFM functions
def get_artist_details(artist, api_key, method):
    url = 'http://ws.audioscrobbler.com/2.0/?'

    endpoint = f'method={method}&artist={artist}&api_key={api_key}&format=json'

    response = requests.get(url+endpoint)

    if response.status_code == 200:
        try:
            data = response.json()
        except JSONDecodeError:
            print("Erreur de décodage JSON, réponse API:", response.text)
            return {}
        return data
    else:
        print('error :', response.status_code, response.text)

def get_track_details(track, artist, api_key, method):
    url = 'http://ws.audioscrobbler.com/2.0/?'

    endpoint = f'method={method}&api_key={api_key}&artist={artist}&track={track}&format=json'

    response = requests.get(url+endpoint)

    if response.status_code == 200:
        try:
            data = response.json()
        except JSONDecodeError:
            print("Erreur de décodage JSON, réponse API:", response.text)
            return {}
        return data
    else:
        print('error :', response.status_code, response.text)


### Spotify functions
def search_track(track_name, artist_name=None):
    query = f"track:{track_name}"
    if artist_name:
        query += f" artist:{artist_name}"

    results = sp.search(q=query, type="track", limit=1)

    if results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        return {
            "album_id": track["album"]["id"],
            "album_name": track["album"]["name"],
            "release_date": track["album"]["release_date"],
            "album_artists_id": ", ".join(str(artist["id"]) for artist in track["album"]["artists"]),
            "album_artists_name": ", ".join(artist["name"] for artist in track["album"]["artists"]),
            "duration": track['duration_ms']/1000,
            "track_id": track["id"],
            "track_name": track["name"],
            "popularity": track["popularity"],
            "track_artists_id": ", ".join(str(artist["id"]) for artist in track["artists"]),
            "track_artists_name": ", ".join(artist["name"] for artist in track["artists"])
        }
    else:
        return None

def get_recent_tracks(numbers):
    """
    numbers : number of last tracks you want to get
    Return a dataframe with the last tracks listened with further informations for each one
    """
    all_tracks = []
    after = None

    while len(all_tracks) < numbers:
        limit = min(50, numbers - len(all_tracks))
        response = sp.current_user_recently_played(limit=limit, after=after)
        tracks = response.get('items', [])

        if not tracks:
            break

        all_tracks.extend(tracks)
        after = tracks[-1]['played_at']
        time.sleep(0.5)

    recent_tracks_df = pd.DataFrame()

    for track in all_tracks:
        album_track_df = pd.json_normalize(track['track']['album'])[['id', 'name', 'release_date']]
        album_track_df.rename(columns={'id':'album_id', 'name':'album_name'}, inplace=True)

        album_artist_str = ", ".join([artist['name'] for artist in track['track']['album']['artists']])
        album_artist_id_str = ", ".join([artist['id'] for artist in track['track']['album']['artists']])

        album_track_df['album_artists_id'] = album_artist_id_str
        album_track_df['album_artists_name'] = album_artist_str

        track_df = pd.json_normalize(track['track'])[['duration_ms', 'id', 'name', 'popularity']]
        track_df['duration_ms'] = track_df['duration_ms']/1000
        track_df.rename(columns={'id':'track_id', 'name':'track_name', 'duration_ms':'duration'}, inplace=True)

        track_details = get_track_details(
            track=track['track']['name'], 
            artist=track['track']['artists'][0]['name'], 
            api_key=api_key_lastfm, 
            method='track.getInfo'
        )

        if "error" in track_details:
            track_details_df = pd.DataFrame(columns=['track_listeners', 'track_playcount'])
        else:
            track_details_df = pd.json_normalize(track_details.get('track', {})).get(['listeners', 'playcount'], pd.DataFrame())
            track_details_df.rename(columns={'listeners': 'track_listeners', 'playcount': 'track_playcount'}, inplace=True)

        track_artist_str = ", ".join([artist['name'] for artist in track['track']['artists']])
        track_artist_id_str = ", ".join([artist['id'] for artist in track['track']['artists']])

        track_df['track_artists_id'] = track_artist_id_str
        track_df['track_artists_name'] = track_artist_str

        ### Get similar artists and musical styles
        similar_artists_list = []
        track_tags_list = []
        for artist in track['track']['artists']:
            current_artist = artist['name']

            artist_details = get_artist_details(artist=current_artist, api_key=api_key_lastfm, method='artist.getInfo')

            if "error" in artist_details:
                similar_artists_list.append(None)
            else:
                similar_artists = artist_details.get('artist', {}).get('similar', {}).get('artist', [])
                similar_artists_list.extend([similar_artist['name'] for similar_artist in similar_artists])

            if "error" in artist_details:
                track_tags_list.append(None)
            else:
                tags = artist_details.get('artist', {}).get('tags', {}).get('tag', [])
                track_tags_list.extend([tag['name'] for tag in tags])
            
        similar_artists_str = ", ".join([artist for artist in set(similar_artists_list) if artist is not None])     # Use of set in order to avoid duplicates
        track_tags_str = ", ".join([track for track in set(track_tags_list) if track is not None])      # Use of set in order to avoid duplicates

        track_df['similar_artists'] = similar_artists_str
        track_df['track_tags'] = track_tags_str

        played_at_df = pd.json_normalize(track)[['played_at']]

        context_df = pd.json_normalize(track['context'])[['type']]
        context_df.rename(columns={'type': 'context_type'}, inplace=True)

        album_track_df = pd.concat([album_track_df, track_df, track_details_df, played_at_df, context_df], axis=1)
        recent_tracks_df = pd.concat([recent_tracks_df, album_track_df], axis=0)
        time.sleep(0.1)

    recent_tracks_df.reset_index(inplace=True, drop=True)
    recent_tracks_df['release_date'] = pd.to_datetime(recent_tracks_df['release_date'], errors='coerce')
    recent_tracks_df['played_at'] = pd.to_datetime(recent_tracks_df['played_at'], errors='coerce')
    return recent_tracks_df


def vectorize_recent_tracks(df):
    """
    df : dataframe with last tracks (returned by get_recent_track)
    Return a dataframe with vectorized data
    """
    # Define the encoder
    encoder = LabelEncoder()

    df_vect = df[['popularity', 'duration', 'track_listeners', 'track_playcount']].copy()

    # Encode all the ids (so we don't need anymore name and titles)
    df_vect['album_id'] = encoder.fit_transform(df['album_id'])
    df_vect['album_artists_id'] = encoder.fit_transform(df['album_artists_id']) ### Corriger avec MultiLabelBinarizer
    df_vect['track_id'] = encoder.fit_transform(df['track_id'])
    df_vect['track_artists_id'] = encoder.fit_transform(df['track_artists_id']) ### Corriger avec MultiLabelBinarizer
    df_vect['context_type'] = encoder.fit_transform(df['context_type'])


    # Vectorize release_date : compare lifeitme since reference date
    reference_date = datetime.strptime('01-01-2000', '%d-%m-%Y')
    df_vect['release_date'] = (df['release_date'] - reference_date).dt.days


    # Vectorize played date : conserve cyclic relation with cos and sin

    df_vect['year'] = df['played_at'].dt.year
    df_vect['month'] = df['played_at'].dt.month
    df_vect['day'] = df['played_at'].dt.day
    df_vect['dayofweek'] = df['played_at'].dt.dayofweek  # 0 = Monday, 6 = Sunday
    df_vect['hour'] = df['played_at'].dt.hour

    df_vect['month_sin'] = np.sin(2 * np.pi * df_vect['month'] / 12)
    df_vect['month_cos'] = np.cos(2 * np.pi * df_vect['month'] / 12)
    df_vect['dayofweek_sin'] = np.sin(2 * np.pi * df_vect['dayofweek'] / 7)
    df_vect['dayofweek_cos'] = np.cos(2 * np.pi * df_vect['dayofweek'] / 7)
    df_vect['day_sin'] = np.sin(2 * np.pi * df_vect['day'] / 31)
    df_vect['day_cos'] = np.cos(2 * np.pi * df_vect['day'] / 31)


    # Vectorize similar_artists and track_tags
    unique_artists = df['similar_artists'].str.split(', ').explode().dropna()
    unique_artists = unique_artists[unique_artists != ''].unique().tolist()

    unique_tags = df['track_tags'].str.split(', ').explode().dropna()
    unique_tags = unique_tags[unique_tags != ''].unique().tolist()

    def vectorize_column(column, unique_values):
        return column.fillna('').str.split(', ').apply(lambda x: pd.Series(unique_values).isin(x).astype(int).tolist())
    
    similar_artists_vect = vectorize_column(df['similar_artists'], unique_artists)
    track_tags_vect = vectorize_column(df['track_tags'], unique_tags)

    ### Transform vector in float number
    pca = PCA(n_components=1)

    df_vect['similar_artists'] = pd.DataFrame(pca.fit_transform(similar_artists_vect.apply(pd.Series).fillna(0)))
    df_vect['track_tags'] = pd.DataFrame(pca.fit_transform(track_tags_vect.apply(pd.Series).fillna(0)))

    return df_vect


### Visualisations

def scatter_plot(X, Y=None):
    """
    X : one features from which we want to get a scatter plot ! it need to be a vector features ! 
    Y : one other features we want to observe the repartition for the plot (None by default)
    Return a scatter plot, using PCA method to divide data
    """
    X = X.apply(pd.Series).fillna(0)

    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(12, 6))

    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=Y, palette="viridis")
    plt.title(f"Scatter Plot {X.columns} PCA")

    plt.tight_layout()
    plt.show()

def pairplot(df, Y=None):
    """
    df : features for a tracklist
    Y : features we want to observe repartition of the plot (None by default)
    Return a pairplot corresponding to the dataframe df
    """
    plt.figure(figsize=(12, 6))
    sns.pairplot(data=df, hue=Y, diag_kind='hist')
    plt.show()

def histplot(X):
    """ 
    X : one specific features of a tracklist
    Return a histplot of this features
    """
    sns.histplot(X, kde=True)
    plt.title(f"{X.columns} histogram")
    plt.show()

def correlation_heatmap(df):
    """
    df : features of a tracklist
    Return a correlation heatmap of the df's features
    """
    corr = df.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Features Correlation Heatmap")
    plt.show()


### Clustering

### Using K-Means
def clustering(df, epsilon=0.1):
    """
    df : vectorized features of a trakclist
    """

    df.columns = df.columns.astype(str)

    ### Pour éviter d'utiliser dropna et garder le même nombre de lignes
    imputer = SimpleImputer(strategy="mean")
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_imputed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inertia = []
        K_range = range(2, 11)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)

        differences = np.diff(inertia)
        optimal_k = K_range[np.argmax(differences < epsilon)]

        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)

        return kmeans.fit_predict(X_scaled)


def visualize_clustering(df, epsilon=0.1):
    """
    df : vectorized features of a trakclist
    """

    df.columns = df.columns.astype(str)

    ### Pour éviter d'utiliser dropna et garder le même nombre de lignes
    imputer = SimpleImputer(strategy="mean")
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_imputed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inertia = []
        K_range = range(2, 11)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)

        plt.plot(K_range, inertia, marker='o')
        plt.title("Méthode du coude pour déterminer le nombre optimal de clusters")
        plt.xlabel("Nombre de clusters")
        plt.ylabel("Inertie")
        plt.show()

        differences = np.diff(inertia)
        optimal_k = K_range[np.argmax(differences < epsilon)]

        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans.fit_predict(X_scaled), palette="viridis")
        plt.title("Clustering (PCA Projection)")
        plt.show()


def get_tracklist(df, limit=3):
    """"
    df: dataframe of recent tracks listened with column 'cluster' after clustering
    limit: limit for similar songs (to reduce execution time)

    Return a dictionnary of tracklist's dataframes for each cluster
    """
    n_clusters = df['cluster'].nunique()
    cluster_dfs = {}

    for i in range(n_clusters):
        cluster = df[df['cluster']==i]
        cluster = cluster[['album_id', 'album_name', 'release_date', 'album_artists_id', 'album_artists_name', 'duration',
                'track_id', 'track_name', 'popularity', 'track_artists_id', 'track_artists_name',
                'track_listeners', 'track_playcount', 'similar_artists', 'track_tags']]
        l=len(cluster)

        for j in range(l):

            similar_tracks = get_track_details(
                track=cluster['track_name'].iloc[j],
                artist=cluster['track_artists_name'].iloc[j].split(",")[0], 
                api_key=api_key_lastfm, 
                method='track.getSimilar'
            )

            if "error" not in similar_tracks:
                similar_tracks = similar_tracks.get('similartracks', {}).get('track', {})[:limit]
                for track in similar_tracks:
                    new_track = pd.json_normalize(search_track(track['name'], track['artist']['name']))
                    new_track_details = get_track_details(
                        track=track['name'], 
                        artist=track['artist']['name'], 
                        api_key=api_key_lastfm, 
                        method='track.getInfo'
                    )

                    if "error" in new_track_details:
                        new_track_details_df = pd.DataFrame(columns=['track_listeners', 'track_playcount'])
                    else:
                        new_track_details_df = pd.json_normalize(new_track_details.get('track', {})).get(['listeners', 'playcount'], pd.DataFrame())
                        new_track_details_df.rename(columns={'listeners': 'track_listeners', 'playcount': 'track_playcount'}, inplace=True)
                    
                    similar_artists_list = []
                    track_tags_list = []

                    artist_details = get_artist_details(artist=track['artist']['name'], api_key=api_key_lastfm, method='artist.getInfo')

                    if "error" in artist_details:
                        similar_artists_list.append(None)
                    else:
                        similar_artists = artist_details.get('artist', {}).get('similar', {}).get('artist', [])
                        similar_artists_list.extend([similar_artist['name'] for similar_artist in similar_artists])

                    if "error" in artist_details:
                        track_tags_list.append(None)
                    else:
                        tags = artist_details.get('artist', {}).get('tags', {}).get('tag', [])
                        track_tags_list.extend([tag['name'] for tag in tags])
                            
                    similar_artists_str = ", ".join([artist for artist in set(similar_artists_list) if artist is not None])     # Use of set in order to avoid duplicates
                    track_tags_str = ", ".join([track for track in set(track_tags_list) if track is not None])      # Use of set in order to avoid duplicates

                    new_track_details_df['similar_artists'] = similar_artists_str
                    new_track_details_df['track_tags'] = track_tags_str

                    new_track = pd.concat([new_track, new_track_details_df], axis=1)

                    if not new_track.empty:
                        cluster = pd.concat([cluster, new_track], axis=0)
            
        cluster.drop_duplicates(subset=['track_name'], inplace=True)
        cluster['release_date'] = pd.to_datetime(cluster['release_date'], errors='coerce')
        cluster.reset_index(drop=True, inplace=True)
            
        cluster_dfs[i] = cluster if not cluster.empty else pd.DataFrame()

    return cluster_dfs


def vectorize_tracklist(cluster_dfs):
    cluster_vect_dfs = {}

    for cluster_name, cluster in cluster_dfs.items():
        encoder = LabelEncoder()
        cluster_vect = cluster[['popularity', 'duration', 'track_listeners', 'track_playcount']].copy()

        ### Encode all the ids
        cluster_vect['album_id'] = encoder.fit_transform(cluster['album_id'])
        cluster_vect['track_id'] = encoder.fit_transform(cluster['track_id'])

        cluster['track_artists_id'] = cluster['track_artists_id'].fillna('').apply(lambda x: x.split(', '))
        cluster['album_artists_id'] = cluster['album_artists_id'].fillna('').apply(lambda x: x.split(', '))

        mlb = MultiLabelBinarizer()
        encoded_artists = pd.DataFrame(mlb.fit_transform(cluster['track_artists_id']))
        encoded_albums = pd.DataFrame(mlb.fit_transform(cluster['album_artists_id']))

        cluster_vect = pd.concat([cluster_vect, encoded_artists, encoded_albums], axis=1)

        reference_date = datetime.strptime('01-01-2000', '%d-%m-%Y')
        cluster_vect['release_date'] = (cluster['release_date'] - reference_date).dt.days

        # Vectorize similar_artists and track_tags
        unique_artists = cluster['similar_artists'].str.split(', ').explode().dropna()
        unique_artists = unique_artists[unique_artists != ''].unique().tolist()

        unique_tags = cluster['track_tags'].str.split(', ').explode().dropna()
        unique_tags = unique_tags[unique_tags != ''].unique().tolist()

        def vectorize_column(column, unique_values):
            return column.fillna('').str.split(', ').apply(lambda x: pd.Series(unique_values).isin(x).astype(int).tolist())
            
        similar_artists_vect = vectorize_column(cluster['similar_artists'], unique_artists)
        track_tags_vect = vectorize_column(cluster['track_tags'], unique_tags)

        ### On convertit les vecteurs en float
        pca = PCA(n_components=1)

        cluster_vect['similar_artists'] = pd.DataFrame(pca.fit_transform(similar_artists_vect.apply(pd.Series).fillna(0)))
        cluster_vect['track_tags'] = pd.DataFrame(pca.fit_transform(track_tags_vect.apply(pd.Series).fillna(0)))

        cluster_vect['track_artists_id'] = pd.DataFrame(pca.fit_transform(encoded_artists.fillna(0)))
        cluster_vect['album_artists_id'] = pd.DataFrame(pca.fit_transform(encoded_albums.fillna(0)))

        cluster_vect_dfs[cluster_name] = cluster_vect if not cluster_vect.empty else pd.DataFrame()

    return cluster_vect_dfs


def compute_cosine_matrices(cluster_vect_dfs):
    cosine_matrices = {}

    for num_cluster, cluster in cluster_vect_dfs.items():
        cosine_matrices[num_cluster] = cosine_similarity(cluster.fillna(0))

    return cosine_matrices


def search_similar_tracks(df, cluster_dfs, cosine_matrices, track_pos, nb_tracks=5):
    track = df.iloc[track_pos]
    num_cluster = track['cluster']

    cluster = cluster_dfs[num_cluster]
    cosine_sim_matrix = cosine_matrices[num_cluster]

    # Trouver la position du track dans le cluster
    current_cluster = df[df['cluster'] == num_cluster].reset_index(drop=True)
    new_track_pos = current_cluster[current_cluster['track_name'] == track['track_name']].index[0]

    # Calcul des similarités
    similarities = list(enumerate(cosine_sim_matrix[new_track_pos]))
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    # Sélection des tracks les plus proches (hors track lui-même)
    top_similar_tracks = [cluster.iloc[i[0]]['track_name'] for i in similarities[1:nb_tracks+1] if i[0] < len(cluster)]
    top_similar_tracks = list(dict.fromkeys(top_similar_tracks))
    top_similar_tracks_str = ", ".join(top_similar_tracks)

    return f"Tracks similaires à '{track['track_name']}' : {top_similar_tracks_str}"


def search_similar_tracks_all(df, track_pos, nb_tracks=5, limit=3):
    ### Clustering
    df_vect = vectorize_recent_tracks(df)
    df['cluster'] = clustering(df_vect)

    cluster_dfs = get_tracklist(df, limit=limit)
    cluster_dfs_vect = vectorize_tracklist(cluster_dfs)

    ### Retrieve the track in the dataframe
    track = df.iloc[track_pos]
    ### Retrieve the track's cluster
    num_cluster = track['cluster']

    ### Get the right cluster
    cluster = cluster_dfs_vect[num_cluster]
    
    ## Récupérer la nouvelle position du track dans le cluster
    current_cluster = df[df['cluster'] == num_cluster].reset_index(drop=True)
    new_track_pos = current_cluster[current_cluster['track_name'] == track['track_name']].index[0]

    ### Get the cosine matrix
    cosine_sim_matrix = cosine_similarity(cluster.fillna(0))  ### Utilisation de fillna pour garder le même nombre de lignes (et ainsi les mêmes positions(track_pos))

    similarities = list(enumerate(cosine_sim_matrix[new_track_pos]))  # on cherche pour la position du track souhaité
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)  # Trier par similarité décroissante

    # Récupérer les indices des 5 tracks les plus proches (en excluant le track lui-même)
    top_similar_tracks = [cluster_dfs[num_cluster].iloc[i[0]]['track_name'] for i in similarities[1:nb_tracks+1]if i[0] < len(cluster_dfs[num_cluster])]
    top_similar_tracks = list(dict.fromkeys(top_similar_tracks))
    top_similar_tracks_str = ", ".join(top_similar_tracks)

    return f"Tracks similaires à '{track['track_name']}' : {top_similar_tracks_str}"

### Analyse clusters

def wordcloud(df):
    for cluster in df["cluster"].unique():
        text = " ".join(df[df["cluster"] == cluster]["track_tags"].dropna())
        
        wordcloud = WordCloud(width=400, height=200, background_color="white").generate(text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word cloud for different music styles in clusters {cluster}")
        plt.show()

def counter(df):
    for cluster in df["cluster"].unique():
        all_artists = ",".join(df[df["cluster"] == cluster]["track_artists_id"].dropna()).split(", ")
        
        artist_counts = Counter(all_artists)
        most_common_artists = artist_counts.most_common(10)

        if len(most_common_artists) == 0:
            print(f"No artists found in cluster {cluster}")
            continue
        
        artists, counts = zip(*most_common_artists)
        
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(counts), y=list(artists), palette="viridis")
        plt.title(f"Top 10 artists - Cluster {cluster}")
        plt.xlabel("Occurrences")
        plt.ylabel("Artists")
        plt.show()